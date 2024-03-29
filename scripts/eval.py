import itertools
from pathlib import Path
import os

import fire
import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompt import get_prompt


def get_mbpp_raw_problems():
    problems = get_mbpp_plus()
    return list(problems.values())

def get_humaneval_raw_problems():
    problems = get_human_eval_plus()
    return list(problems.values())

def map_mbpp_problem(p: dict):
    id = p["task_id"]
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    prompt = prompt[start_index + 3 : end_index]
    assert_index = prompt.index("assert")
    instruction = prompt[:assert_index].strip()
    if not instruction.endswith("."):
        instruction += "."
    assertion = prompt[assert_index:].strip()
    instruction = f"""{instruction} Your code should satisfy the following assertion:
```python
{assertion}
```"""
    response_prefix = f"""```python"""
    
    return {
        'id': str(id),
        'instruction': instruction,
        'response_prefix': response_prefix
    }

def map_humaneval_problem(p: dict):
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()

    instruction = f"""Write a solution to the following problem:
```python
{prompt}
```"""
    response_prefix = f"""```python
{prompt}"""
    return {
        'id': id,
        'instruction': instruction,
        'response_prefix': response_prefix
    }

def chunked(seq, n):
    return (seq[i : i + n] for i in range(0, len(seq), n))

def get_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer

def pad_sequences(
    sequences,
    pad_value: int,
    padding_side,
    dtype: torch.dtype = torch.long,
    padding_length = None,
) -> torch.Tensor:
    tensors = [torch.tensor(sequence, dtype=dtype) for sequence in sequences]
    max_len = max(len(sequence) for sequence in sequences)
    if padding_length is not None:
        assert padding_length >= max_len, "padding_length must be >= max_len"
        max_len = padding_length
    if padding_side == "right":
        result = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_value
        )
        remaining_length = max_len - result.shape[-1]
        # padding matrix of (batch_size * remaining_length)
        shape = result.shape[:-1] + (remaining_length,)
        padding_matrix = torch.full(shape, pad_value, dtype=dtype)
        result = torch.cat([result, padding_matrix], dim=-1)
    else:
        padded_tensors: list[torch.Tensor] = []
        for tensor in tensors:
            n_pad_values = max_len - len(tensor)
            padded_values = torch.full((n_pad_values,), pad_value, dtype=dtype)
            padded_tensor = torch.cat([padded_values, tensor], dim=0)
            assert len(padded_tensor) == max_len
            padded_tensors.append(padded_tensor)
        result = torch.stack(padded_tensors, dim=0)
    assert result.shape == torch.Size([len(sequences), max_len])
    return result

def get_response(model, tokenizer, prompts):
    
    input_ids = tokenizer(prompts, add_special_tokens=False)
    input_ids = input_ids['input_ids']
    
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    bos_token_ids = [bos_token_id] if bos_token_id else []
    input_ids = [
        bos_token_ids + input_id for input_id in input_ids
    ]

    input_ids = pad_sequences(
        sequences=input_ids,
        pad_value=tokenizer.pad_token_id,
        padding_side="left",
    )
    
    input_ids = input_ids.to(model.device)
    input_len = input_ids.shape[1]
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    generation_config = GenerationConfig(
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
        num_beams=4
    )

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )

    output_ids = outputs[:, input_len:]
    
    output_strings = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    return {
        "raw_inputs": input_ids,
        "raw_outputs": output_ids,
        "decoded_outputs": output_strings
    }

def main(
    base_model,
    save_path = '',
    dataset = 'humaneval',
    n_problems_per_batch = 16,
    n_batches = 1,
    n_samples_per_problem = 1,
    
):

    save_path=f"evalplus-{os.path.basename(base_model)}-{dataset}.jsonl"

    # raw_problems = get_humaneval_raw_problems()
    # problems = list(map(map_humaneval_problem, raw_problems))

    raw_problem_fn, map_problem_fn = (
        (get_humaneval_raw_problems, map_humaneval_problem)
        if dataset == "humaneval"
        else (get_mbpp_raw_problems, map_mbpp_problem)
    )
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    model, tokenizer = get_model(base_model)
    
    problems_chunked = list(chunked(list(problems), n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(n_batches))
    n_total = len(problems_chunked) * n_batches

    Path(save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        prompts = [
            get_prompt(problem["instruction"], problem["response_prefix"])
            for problem in problems
        ]
        print("PROMPT")
        print(prompts[-1])
        all_prompts = prompts * n_samples_per_problem
        all_task_ids = task_ids * n_samples_per_problem
        response = get_response(model, tokenizer, all_prompts)
        completions = response['decoded_outputs']
        assert len(problems) <= n_problems_per_batch
        assert len(completions) == len(problems) * n_samples_per_problem
        print("COMPLETION")
        print(completions[-1])
        samples = [
            dict(
                task_id=task_id,
                completion=completion[
                    : index
                    if (index := completion.find("```")) != -1
                    else len(completion)
                ],
            )
            for task_id, completion in zip(all_task_ids, completions)
        ]
        write_jsonl(save_path, samples, append=True)


if __name__ == "__main__":
    fire.Fire(main)
