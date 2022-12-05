import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
    DataCollatorForLanguageModeling
)
import random
import numpy as np
import logging

from tasks.superglue.dataset import SuperGlueDataset

from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForMultipleChoice(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        print(batch)
        input_list = [sample['input_ids'] for sample in batch]

        choice_nums = list(map(len, input_list))
        max_choice_num = max(choice_nums)

        def pad_choice_dim(data, choice_num):
            if len(data) < choice_num:
                data = np.concatenate([data] + [data[0:1]] * (choice_num - len(data)))
            return data

        for i, sample in enumerate(batch):
            for key, value in sample.items():
                if key != 'label':
                    sample[key] = pad_choice_dim(value, max_choice_num)
                else:
                    sample[key] = value
            # sample['loss_mask'] = np.array([1] * choice_nums[i] + [0] * (max_choice_num - choice_nums[i]),
            #                                dtype=np.int64)

        return batch


class SuperGlueDatasetForRecord(SuperGlueDataset):
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        raw_datasets = load_dataset("super_glue", data_args.dataset_name)
        self.tokenizer = tokenizer
        self.data_args = data_args
        #labels
        self.multiple_choice = data_args.dataset_name in ["copa", "record"]

        if not self.multiple_choice:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        self.label_to_id = None

        if self.label_to_id is not None:
            self.label2id = self.label_to_id
            self.id2label = {id: label for label, id in self.label2id.items()}
        elif not self.multiple_choice:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}


        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

            self.train_dataset = self.train_dataset.map(
                self.prepare_train_dataset,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on train dataset",
            )
            
        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

            self.eval_dataset = self.eval_dataset.map(
                self.prepare_eval_dataset,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on validation dataset",
            )
            
        self.metric = load_metric("super_glue", data_args.dataset_name)

        self.data_collator = DataCollatorForMultipleChoice(tokenizer)
        # if data_args.pad_to_max_length:
        #     self.data_collator = default_data_collator
        # elif training_args.fp16:
        #     self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    def preprocess_function(self, examples):
        results = {
            "input_ids": list(),
            "attention_mask": list(),
            "token_type_ids": list(),
            "label": list()
        }
        for passage, query, entities, answers in zip(examples["passage"], examples["query"], examples["entities"], examples["answers"]):
            passage = passage.replace("@highlight\n", "- ")

            input_ids = []
            attention_mask = []
            token_type_ids = []
            
            for _, ent in enumerate(entities):
                question = query.replace("@placeholder", ent)
                result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
                
                input_ids.append(result["input_ids"])
                attention_mask.append(result["attention_mask"])
                if "token_type_ids" in result: token_type_ids.append(result["token_type_ids"])
                label = 1 if ent in answers else 0
            
            result["label"].append()

        return results


    def prepare_train_dataset(self, examples, max_train_candidates_per_question=10):
        entity_shuffler = random.Random(44)
        results = {
            "input_ids": list(),
            "attention_mask": list(),
            "token_type_ids": list(),
            "label": list()
        }
        for passage, query, entities, answers in zip(examples["passage"], examples["query"], examples["entities"], examples["answers"]):
            passage = passage.replace("@highlight\n", "- ")
            
            for answer in answers:
                input_ids = []
                attention_mask = []
                token_type_ids = []
                candidates = [ent for ent in entities if ent not in answers]
                # if len(candidates) < max_train_candidates_per_question - 1:
                #     continue
                if len(candidates) > max_train_candidates_per_question - 1:
                    entity_shuffler.shuffle(candidates)
                    candidates = candidates[:max_train_candidates_per_question - 1]
                candidates = [answer] + candidates

                for ent in candidates:
                    question = query.replace("@placeholder", ent)
                    result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
                    input_ids.append(result["input_ids"])
                    attention_mask.append(result["attention_mask"])
                    if "token_type_ids" in result: token_type_ids.append(result["token_type_ids"])

                results["input_ids"].append(input_ids)
                results["attention_mask"].append(attention_mask)
                if len(token_type_ids) > 0: results["token_type_ids"].append(token_type_ids)
                results["label"].append(0)

        return results
            

    def prepare_eval_dataset(self, examples):

        results = {
            "input_ids": list(),
            "attention_mask": list(),
            "token_type_ids": list(),
            "label": list()
        }
        for passage, query, entities, answers in zip(examples["passage"], examples["query"], examples["entities"], examples["answers"]):
            passage = passage.replace("@highlight\n", "- ")
            for answer in answers:
                input_ids = []
                attention_mask = []
                token_type_ids = []

                for ent in entities:
                    question = query.replace("@placeholder", ent)
                    result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
                    input_ids.append(result["input_ids"])
                    attention_mask.append(result["attention_mask"])
                    if "token_type_ids" in result: token_type_ids.append(result["token_type_ids"])

                results["input_ids"].append(input_ids)
                results["attention_mask"].append(attention_mask)
                if len(token_type_ids) > 0: results["token_type_ids"].append(token_type_ids)
                results["label"].append(0)

        return results
