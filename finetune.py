
import torch
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM

from utils.common import parse_args, load_data
from utils.preprocess_data import prepare_data
from utils.data_collator import DynamicDataCollatorWithPadding
from model.llama import convert_llama_model


def main():

    local_branch = 2048
    global_branch = 10

    model_args, data_args, training_args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.float16,
        # device_map=device_map,
    )
    # model = convert_llama_model(model, local_branch, global_branch)

    model.print_trainable_parameters()

    dataset = load_data(data_args)

    trainer_data = prepare_data(dataset, data_args, tokenizer)
    data_collator = DynamicDataCollatorWithPadding(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **trainer_data
    )

    train_result = trainer.train()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
