import itertools
from pathlib import Path
import os

import fire
import torch
from evalplus.data import get_human_eval_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompt import get_prompt


def get_humaneval_raw_problems():
    problems = get_human_eval_plus()
    return list(problems.values())

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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return model, tokenizer

def get_response(model, tokenizer, prompts):
    
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = input_ids['input_ids'].to(model.device)
    input_len = input_ids.shape[1]
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    generation_config = GenerationConfig(
        # temperature=0.0,
        top_p=1,
        num_beams=4,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=0
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
    n_problems_per_batch = 16,
    n_batches = 1,
    n_samples_per_problem = 1,
    
):

    save_path=f"evalplus-{os.path.basename(base_model)}-humaneval.jsonl"

    raw_problems = get_humaneval_raw_problems()
    problems = list(map(map_humaneval_problem, raw_problems))

    model, tokenizer = get_model(base_model)
    
    problems_chunked = list(chunked(list(problems), n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(n_batches))
    n_total = len(problems_chunked) * n_batches

    Path(save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        prompts = [
            get_prompt(problem["instruction"]) + problem["response_prefix"]
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
