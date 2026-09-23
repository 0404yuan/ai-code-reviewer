import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="bigcode/starcoder",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    use_auth_token: bool = field(
        default=True,
        metadata={"help": "Whether to use the token for authentication to the hub."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_column: str = field(
        default="content", metadata={"help": "Column name of the dataset to train on."}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for tokenization."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.use_auth_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.use_auth_token,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    def tokenize(examples):
        return tokenizer(
            examples[data_args.data_column],
            truncation=True,
            max_length=data_args.max_seq_length,
        )

    tokenized = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
