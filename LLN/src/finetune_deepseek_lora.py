import argparse
import gc
import json
import logging
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class LoRATrainer:
    def __init__(self, model_name: str, output_dir: str, local_rank: int = -1):
        self.model_name = model_name
        self.output_dir = output_dir
        self.local_rank = local_rank
        self.model = None
        self.tokenizer = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _is_main_process(self) -> bool:
        return self.local_rank in (-1, 0)

    def setup_model_and_tokenizer(self):
        if self._is_main_process():
            self.logger.info("Loading base model: %s", self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = None if self.local_rank != -1 else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )

        if self.local_rank != -1:
            self.model = self.model.to(f"cuda:{self.local_rank}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)

        if self._is_main_process():
            self.model.print_trainable_parameters()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preprocess_data(self, data_file: str, max_length: int) -> Dataset:
        if self._is_main_process():
            self.logger.info("Preprocessing training data: %s", data_file)

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        def format_prompt(sample):
            instruction = sample["instruction"]
            input_text = sample["input"]
            output_text = sample["output"]
            return f"User: {instruction}\n\n{input_text}\n\nAssistant: {output_text}"

        dataset = Dataset.from_list(
            [{"text": format_prompt(sample)} for sample in data]
        )

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None,
            )

        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

    def train(self, args):
        if self._is_main_process():
            self.logger.info("Starting LoRA fine-tuning")

        dataset = self.preprocess_data(args.training_data, args.max_length)
        train_val_split = dataset.train_test_split(test_size=args.validation_split)
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]

        if self._is_main_process():
            self.logger.info("Training set size: %d", len(train_dataset))
            self.logger.info("Validation set size: %d", len(val_dataset))

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            local_rank=self.local_rank,
            save_safetensors=True,
            ignore_data_skip=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()

        if self._is_main_process():
            trainer.save_model()
            self.tokenizer.save_pretrained(args.output_dir)
            self.logger.info("Fine-tuned adapter saved to: %s", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek with LoRA.")
    parser.add_argument("--model_name", default="deepseek-ai/deepseek-llm-7b-chat")
    parser.add_argument("--training_data", required=True)
    parser.add_argument("--output_dir", default="LLN/models/lora_model")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.empty_cache()

    trainer = LoRATrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        local_rank=local_rank,
    )
    trainer.setup_model_and_tokenizer()
    trainer.train(args)


if __name__ == "__main__":
    main()
