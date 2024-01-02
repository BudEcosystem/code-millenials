from typing import Optional, Tuple
import torch
import math
import torch.nn.functional as F

from .lambda_attention import lambda_matmul


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(vec, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    vec_embed = (vec * cos) + (rotate_half(vec) * sin)
    return vec_embed


# Efficient implementation using `models/lambda_attention.py`
def attn_forward_factory(
    self, local_branch, global_branch, limit_distance
):

    def limited_distance_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]

        # New: here we change the code to store the un-rotated key and value
        # states, as they are useful for stationary attention.
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            key_position_ids = torch.cat([past_key_value[2], position_ids], dim=1)
            kv_seq_len += past_key_value[0].shape[-2]
        else:
            key_position_ids = position_ids

        past_key_value = (key_states, value_states, key_position_ids) if use_cache else None

        # inv_freq controls the dtype of rotation phase, which can be large
        self.rotary_emb.inv_freq = self.rotary_emb.inv_freq.to(torch.float32)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        rot_query_states = apply_rotary_pos_emb(
            query_states, cos, sin, position_ids)
        rot_key_states = apply_rotary_pos_emb(
            key_states, cos, sin, key_position_ids)

        if limit_distance is None:
            stationary_key_states = rot_key_states
            stationary_query_states = rot_query_states
        else:
            stationary_key_states = key_states
            effective_limit_distance = min(limit_distance, kv_seq_len-1)
            stationary_query_states = \
                (query_states * cos[0, 0, effective_limit_distance]) + \
                (rotate_half(query_states) * sin[0, 0, effective_limit_distance])

        
        headwise_limit = 33000  # magic number set for A100 GPU
        if q_len > headwise_limit:
            for head_i in range(self.num_heads):
                query_states[:, head_i] = (
                    lambda_matmul(
                        rot_key_states[:, head_i],
                        stationary_key_states[:, head_i],
                        rot_query_states[:, head_i],
                        stationary_query_states[:, head_i],
                        local_branch, global_branch
                    ) / math.sqrt(self.head_dim)
                ).softmax().matmul(value_states[:, head_i])
        else:
            query_states = (
                lambda_matmul(
                    rot_key_states,
                    stationary_key_states,
                    rot_query_states,
                    stationary_query_states,
                    local_branch, global_branch
                ) / math.sqrt(self.head_dim)
                ).softmax().matmul(value_states)

        
        attn_output = query_states
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return limited_distance_forward


def convert_llama_model(model, local_branch, global_branch):
    for layer_i, hidden_layer in enumerate(model.model.layers):
        attn = hidden_layer.self_attn
        attn.forward = attn_forward_factory(
            attn, local_branch, global_branch, local_branch
        )
    return model