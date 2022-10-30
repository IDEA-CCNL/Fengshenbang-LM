# -*- encoding: utf-8 -*-
'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   qa_dataset.py
@Time    :   2022/10/28 19:57
@Author  :   He Junqing
@Version :   1.0
@Contact :   hejunqing@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''
# here put the import lib

from dataclasses import dataclass
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fengshen.data.t5_dataloader.t5_gen_datasets import DialogDataset


class T5StyleDataset(DialogDataset):

    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group("Dataset")
        parser.add_argument("--max_seq_length", default=512, type=int)
        parser.add_argument("--max_knowledge_length", default=128, type=int)
        parser.add_argument("--max_target_length", default=128, type=int)
        return parent_args

    def regular_tokenize(self, sample):
        """
        sample.keys:question:str,context:stc, answer:[],idx:int,ans_span:[]
        """
        plain_text = (
            "question:"
            + sample["question"]
            + "knowledge:"
            + sample["context"][: self.max_knowledge_length]
        )
        l_text = len(plain_text)

        ctx_len = self.max_seq_length - l_text - 1
        if ctx_len > 0 and "history" in sample:
            context = "[SEP]".join(sample["history"])
            plain_text += "context:" + context

        res_prefix = self.tokenizer.encode("answer:", add_special_tokens=False)
        # res_prefix.tolist()
        l_rp = len(res_prefix)

        tokenized = self.tokenizer.encode(
            plain_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length - 2 - l_rp,
        )
        # tokenized.tolist()
        tokenized += res_prefix
        # add maskid
        mask_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        tokenized.append(mask_id)
        tokenized.append(self.eos_token_id)
        # print(tokenized)

        target_ids = self.tokenizer.encode(
            "<extra_id_0>" + sample["answer"][0],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_target_length,
        )

        # print(target_ids)
        tokenized_sample = {}
        tokenized_sample["input_ids"] = np.array(tokenized, dtype=np.int32)
        tokenized_sample["attention_mask"] = np.ones(len(tokenized), dtype=np.int8)
        tokenized_sample["labels"] = np.array(target_ids, dtype=np.int32)
        tokenized_sample["idx"] = sample["idx"]
        # print(tokenized_sample)
        return tokenized_sample


@dataclass
class TextGenCollator:
    '''
    '''
    config: None
    pad_token_id: -100
    decoder_start_token_id: 0
    formator: str = 't5style'

    def setup(self):
        pass

    def __call__(self, samples):
        batch = {
            k: [
                torch.tensor(samples[i][k], dtype=torch.int64)
                for i in range(len(samples))
            ]
            for k in ["input_ids", "attention_mask", "labels"]
        }
        batch["idx"] = torch.tensor([samples[i]["idx"] for i in range(len(samples))])

        # print(batch)
        for k, v in batch.items():
            if k != "labels" and k != "idx":
                batch[k] = pad_sequence(
                    v, batch_first=True, padding_value=self.pad_token_id
                )
            elif k == "labels":
                batch[k] = pad_sequence(v, batch_first=True, padding_value=-100)

        batch["decoder_input_ids"] = torch.tensor(
            self.shift_tokens_right(
                batch["labels"], self.pad_token_id, self.decoder_start_token_id
            ),
            dtype=torch.long,
        )
        return batch

    def shift_tokens_right(
        self, input_ids: np.array, pad_token_id: int, decoder_start_token_id: int
    ) -> np.ndarray:
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = np.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id

        shifted_input_ids = np.where(
            shifted_input_ids == -100, pad_token_id, shifted_input_ids
        )
        return shifted_input_ids


if __name__ == "__main__":
    # test
    import argparse

    total_parser = argparse.ArgumentParser("DATASET parser")
    total_parser.add_argument(
        "--tokenizer_type",
        default="t5_tokenizer",
        choices=["bert_tokenizer", "t5_tokenizer"],
    )
    total_parser.add_argument("--preprocessing_num_workers", default="4", type=int)
    total_parser.add_argument(
        "--new_vocab_path",
        default=None,
        type=str,
    )

    total_parser.add_argument(
        "--pretrained_model_path",
        default="YOUR DOWNLOAD MODEL PATH",
    )
    total_parser.add_argument("--train_split_size", default=0.995, type=int)
    total_parser.add_argument(
        "--formator", default="t5style", choices=["t5style", "squad", "dialog"]
    )
    total_parser = TextGenCollator.add_data_specific_args(total_parser)
    args = total_parser.parse_args()
    args.train_data_path = "cmrc"
    ds = T5StyleDataset("cmrc", args, "dev")
    print(len(ds))
    for i in range(10):
        print(ds[i])

    dl = TextGenCollator(args)
    for i in range(5):
        for batch in dl.val_dataloader():
            print(batch)
            print(batch["input_ids"])
            print(batch["no_answer"])
            print(batch["decoder_input_ids"])
            print(batch["labels"])
