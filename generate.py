
import os
import sys

import fire
import gradio as gr
import torch

from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompt import get_prompt

def main(
    base_model: str = "",
):
    device = "cuda"
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.half()
    model.eval()

    def evaluate(
        instruction,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = get_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens
        )

        decoded = [
            tokenizer.decode(_o[_a.sum():], skip_special_tokens=True)
            for _a, _o in zip(attention_mask, generation_output)
        ]

        yield decoded[0]

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Query",
                placeholder="Ask me anything",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=8192, step=1, value=128, label="Max tokens"
            )
        ],
        outputs=[
            gr.components.Textbox(
                lines=15,
                label="Output",
            )
        ],
        title="Code Millenials",
        description="An instruction finetuned model",
    ).queue().launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    fire.Fire(main)
