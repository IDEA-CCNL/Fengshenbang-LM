from curses import flash
import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the eval data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)

def pad(ids, pad_id, max_length):
    if len(ids) > max_length:
        return ids[:max_length]
    return ids + [pad_id] * (max_length - len(ids))

prompt_without_output = "[human]:{prompt}\n[bot]:"

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        user_tokens=[39408],
        assistant_tokens=[39409],
    ):
        super(SupervisedDataset, self).__init__()
        # self.data = json.load(open(data_path))
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print(item, flush=True)
        print("input:", self.tokenizer.decode(item["input_ids"]), flush=True)
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels), flush=True)

    def __len__(self):
        return len(self.data)

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r') as file:
            for line in file.readlines():
                data.append(json.loads(line))
        return data
        
    def preprocessing(self, example):
        input_ids_list = []
        labels_list = []
        # print('example:\n', example)
        prompt_cnt = min(len(example["prompt"]), len(example["output"]))
        input_ids = []
        labels_ids = []
        for i in range(prompt_cnt):
            prompt_input_ids = self.tokenizer(prompt_without_output.format_map(
                {"prompt": example["prompt"][i].strip()}), add_special_tokens=False).input_ids
            output_ids = self.tokenizer(example["output"][i].strip(), add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
            
            input_ids += prompt_input_ids
            input_ids += output_ids
            labels_ids += [-100] * (len(prompt_input_ids)) + output_ids
        
        # PAD dynamic not here, but data_collector
        # max_len = min(self.model_max_length, max(len(input_ids), len(labels_ids)))
        max_len = self.model_max_length
        input_ids = pad(input_ids, self.tokenizer.pad_token_id, max_len)
        labels_ids = pad(labels_ids, -100, max_len)
        input_ids_list = torch.tensor(input_ids).clone()
        labels_list = torch.tensor(labels_ids).clone()
        attention_mask = input_ids_list.ne(self.tokenizer.pad_token_id)
        padding_mask = attention_mask

        model_inputs = {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask,
            'labels': labels_list,
        }
        return model_inputs

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # use_flash_attention_2=True,
        use_cache = False,
    )
    model.config.use_cache = False
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )
    tokenizer.pad_token = '</s>'
    tokenizer.pad_token_id = 2
    
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    eval_dataset = SupervisedDataset(
        data_args.eval_data_path, tokenizer, training_args.model_max_length
    )
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=collator
    )
    trainer.train()
    # trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()