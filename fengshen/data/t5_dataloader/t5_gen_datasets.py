# -*- encoding: utf-8 -*-
'''
@File    :   t5_gen_datasets.py
@Time    :   2022/10/24 19:29
@Author  :   He Junqing
@Version :   1.0
@Contact :   hejunqing@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''

from logging import exception
from transformers import (
    BertTokenizer,
    MT5Config,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import numpy as np
import sys

sys.path.append("../../")

special_token_dict = {
    "additional_special_tokens": [
        "[CTSTART]",
        "[CTEND]",
        "[SEP]",
        "[KNSTART]",
        "[KNEND]",
    ]
}


class DialogDataset(Dataset):
    def __init__(self, data_path, args, data, load_data_type=1) -> None:
        super().__init__()

        if args.tokenizer_type == "t5_tokenizer":
            self.tokenizer = MT5Tokenizer.from_pretrained(
                args.pretrained_model_path)
            if len(self.tokenizer) == 32596:
                self.tokenizer.add_special_tokens(special_token_dict)
                print(
                    "add special tokens to tokenizer,vocab size:",
                    len(self.tokenizer)
                )
                self.model = MT5ForConditionalGeneration.from_pretrained(
                    args.pretrained_model_path
                )
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.save_pretrained(args.new_vocab_path)
                self.tokenizer.save_pretrained(
                    args.new_vocab_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                args.pretrained_model_path)

        self.load_data_type = load_data_type
        self.data_split = data
        self.num_workers = args.preprocessing_num_workers
        self.max_seq_length = args.max_seq_length
        self.max_knowledge_length = args.max_knowledge_length
        self.max_target_length = args.max_target_length

        # tokenizer config
        self.config = MT5Config.from_pretrained(args.pretrained_model_path)
        self.decoder_start_token_id = self.config.decoder_start_token_id
        self.eos_token_id = self.config.eos_token_id
        self.vocab_size = self.config.vocab_size
        # print(self.tokenizer.decode([2]))

        # load from raw data or hf dataset

        if self.load_data_type == 0:
            self.data = self.load_data(data_path)
        elif self.load_data_type == 1:
            self.data = self.load_packed_data(data_path)
        else:  # for testing
            self.data = data_path

    def load_packed_data(self, data_path):
        from fengshen.data.fs_datasets import load_dataset

        samples = load_dataset(data_path,
                               num_proc=self.num_workers)[self.data_split]
        tokenized_samples = samples.map(
            self.regular_tokenize, batched=False,
            num_proc=self.num_workers
        )

        return tokenized_samples

    def load_data(self, data_path):
        """
        load data from raw data
        return untokoenized data
        """
        from datasets import load_dataset

        ds = load_dataset("json", data_files=data_path)['train']
        samples = ds.map(self.regular_tokenize, batched=False, num_proc=self.num_workers
                         )
        return samples

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def regular_tokenize(self, sample):
        # print(len(sample['context']))
        context_ids = self.tokenizer(
            sample["context"],
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=True,
        )

        context_types = self.get_token_type(
            sample["context"], context_ids["token_type_ids"]
        )
        # print('context',sample['context'])
        # print('context_ids',context_ids['input_ids'])
        knowledge_ids = self.tokenizer.encode(
            sample["knowledge"], add_special_tokens=False
        )
        # print('knowledge_ids',knowledge_ids)
        if isinstance(knowledge_ids, int):
            knowledge_ids = [knowledge_ids]
        target_ids = self.tokenizer.encode(
            sample["target"],
            add_special_tokens=False,
            max_length=self.max_target_length - 1,
            truncation=True,
        )
        # print('target',sample['target'])
        # print('target_ids',target_ids)
        # print('decode target',self.tokenizer.decode(target_ids))
        # truncate

        knowledge_ids = (
            [self.tokenizer.convert_tokens_to_ids("[KNSTART]")]
            + knowledge_ids[: self.max_knowledge_length - 2]
            + [self.tokenizer.convert_tokens_to_ids("[KNEND]")]
        )
        l_kn = len(knowledge_ids)
        knowledge_types = [2] * l_kn

        flatten_context = []
        for line in context_ids["input_ids"]:
            flatten_context.extend(line)
        l_ct = min(len(flatten_context), self.max_seq_length - l_kn - 2)
        context_ids = (
            [self.tokenizer.convert_tokens_to_ids("[CTSTART]")]
            + flatten_context[-l_ct:]
            + [self.tokenizer.convert_tokens_to_ids("[CTEND]")]
        )

        context_types = context_types[-l_ct:] + [0]
        context_types.insert(0, context_types[0])
        assert len(context_ids) == len(
            context_types
        ), "len of context ids and token types unmatch, context:{},ids:{} types:{},len {}:{}".format(
            sample["context"],
            context_ids,
            context_types,
            len(context_ids),
            len(context_types),
        )

        try:
            target_ids = target_ids + [self.eos_token_id]
        except exception:
            print(sample["target"], target_ids, self.eos_token_id)

        tokenized = {}
        tokenized["input_ids"] = np.array(context_ids + knowledge_ids, dtype=np.int32)
        tokenized["token_types"] = np.array(
            context_types + knowledge_types, dtype=np.int32
        )
        tokenized["attention_mask"] = np.ones(
            len(context_types + knowledge_types), dtype=np.int8
        )
        tokenized["labels"] = np.array(target_ids, dtype=np.int32)

        return tokenized

    def get_token_type(self, context, tokentypes=None):
        # token_type fail in tokenizer, all zero
        context_token_types = []
        for i, line in enumerate(context):
            if tokentypes:
                if i % 2 == 0:
                    token_type = [0] * len(tokentypes[i])
                else:
                    token_type = [1] * len(tokentypes[i])
            else:
                if i % 2 == 0:
                    token_type = [0] * (1 + len(line))
                else:
                    token_type = [1] * (1 + len(line))

            context_token_types.extend(token_type)

        return context_token_types


class DialogDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group("SuperviseT5DataModel")
        parser.add_argument("--dataset_num_workers", default=8, type=int)
        parser.add_argument("--dataloader_num_workers", default=4, type=int)
        parser.add_argument("--train_data_path", default="dialog_4g_test", type=str)
        parser.add_argument(
            "--valid_data_path", default="wudao_180g_mt5_tokenized", type=str
        )
        parser.add_argument("--train_batchsize", default=2, type=int)
        parser.add_argument("--valid_batchsize", default=2, type=int)
        parser.add_argument("--max_seq_length", default=512, type=int)
        parser.add_argument("--max_knowledge_length", default=128, type=int)
        parser.add_argument("--max_target_length", default=128, type=int)

        return parent_args

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.load_data(args)
        self.epochs = args.max_epochs

    def load_data(self, args):
        if args.train_split_size is not None:
            from fengshen.data.fs_datasets import load_dataset

            data_splits = load_dataset(
                args.train_data_path, num_proc=args.dataset_num_workers
            )
            train_split = data_splits['train']
            test_split = data_splits['test']
            print('train:', train_split, '\ntest_data:', test_split)
            self.train_dataset = DialogDataset(
                args.train_data_path, args, load_data_type=1, data="train"
            )
            self.test_dataset = DialogDataset(
                args.train_data_path, args, load_data_type=1, data="test"
            )
        else:
            self.train_data = DialogDataset(
                args.train_data_path, args, load_data_type=1
            )

        self.config = MT5Config.from_pretrained(args.pretrained_model_path)
        self.pad_token_id = self.config.pad_token_id
        self.decoder_start_token_id = self.config.decoder_start_token_id
        print("bos id:", self.decoder_start_token_id)

    def collate_fn(self, samples):
        batch = {
            k: [
                torch.tensor(samples[i][k], dtype=torch.int64)
                for i in range(len(samples))
            ]
            for k in ["input_ids", "token_types", "attention_mask", "labels"]
        }

        # print(batch)
        for k, v in batch.items():
            if k != "labels":
                batch[k] = pad_sequence(
                    v, batch_first=True, padding_value=self.pad_token_id
                )
            else:
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

    def train_dataloader(self):
        from fengshen.data.universal_datamodule.universal_sampler import (
            PretrainingRandomSampler,
        )
        from fengshen.data.universal_datamodule.universal_datamodule import (
            get_consume_samples,
        )

        # 采用自定义的sampler，确保继续训练能正确取到数据
        consumed_samples = get_consume_samples(self)
        batch_sampler = PretrainingRandomSampler(
            epoch=self.epochs,
            total_samples=len(self.train_dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.hparams.train_batchsize,
            data_parallel_rank=self.trainer.global_rank,  # gpu idx
            data_parallel_size=self.trainer.world_size,  # gpu num
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            pin_memory=True,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.test_dataset, shuffle=False
        )
        return DataLoader(
            self.test_dataset,
            sampler=sampler,
            shuffle=False,
            batch_size=self.hparams.valid_batchsize,
            pin_memory=True,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.test_dataset, shuffle=False
        )
        return DataLoader(
            self.test_dataset,
            sampler=sampler,
            shuffle=False,
            batch_size=self.hparams.valid_batchsize,
            pin_memory=True,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    # test
    import argparse

    total_parser = argparse.ArgumentParser("DATASET parser")
    total_parser.add_argument(
        "--tokenizer_type",
        default="t5_tokenizer",
        choices=["bert_tokenizer", "t5_tokenizer"],
    )
    total_parser.add_argument("--preprocessing_num_workers", default="10", type=int)
    total_parser.add_argument(
        "--new_vocab_path",
        default="/cognitive_comp/hejunqing/projects/Dialog_pretrain/randeng_t5_newvocab_784M",
        type=str,
    )
    total_parser.add_argument("--train_split_size", default=0.995, type=int)
    total_parser.add_argument(
        "--pretrained_model_path",
        default="/cognitive_comp/hejunqing/projects/Dialog_pretrain/randeng_t5_newvocab_784M",
    )
    total_parser = DialogDataModel.add_data_specific_args(total_parser)
    args = total_parser.parse_args()
    dl = DialogDataModel(args)

    for i in range(5):
        for batch in dl.train_dataloader():
            print(batch)
            print(batch["input_ids"])
            print(batch["token_types"])
            print(batch["decoder_input_ids"])
            print(batch["labels"])

    print("test finish")
