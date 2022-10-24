# encoding: utf8

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import MT5Config

from fengshen.data.t5_dataloader.t5_gen_datasets import DialogDataModel, DialogDataset


class T5StyleDataset(DialogDataset):
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


class TextGenDataModel(DialogDataModel):
    def load_data(self, args):
        if args.train_split_size is not None:
            from fengshen.data.fs_datasets import load_dataset

            data_splits = load_dataset(
                args.train_data_path, num_proc=args.dataset_num_workers
            )
            if args.do_eval_only:
                test_split = data_splits["test"]
                print("\ntest_data:", test_split)
            else:
                train_split = data_splits["train"]
                dev_split = data_splits["dev"]
                print("train:", train_split, "\ndev_data:", dev_split)

            if not args.do_eval_only:
                self.train_dataset = T5StyleDataset(
                    args.train_data_path, args, load_data_type=1, data="train"
                )
                self.dev_dataset = T5StyleDataset(
                    args.train_data_path, args, load_data_type=1, data="dev"
                )
            else:
                self.test_dataset = T5StyleDataset(
                    args.train_data_path, args, load_data_type=1, data="test"
                )
        else:

            self.train_dataset = T5StyleDataset(
                args.train_data_path, args, load_data_type=1, data="train"
            )

        self.config = MT5Config.from_pretrained(args.pretrained_model_path)
        self.pad_token_id = self.config.pad_token_id
        self.decoder_start_token_id = self.config.decoder_start_token_id
        self.formator = args.formator
        print("bos id:", self.decoder_start_token_id)

    def val_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.dev_dataset, shuffle=False
        )
        return DataLoader(
            self.dev_dataset,
            sampler=sampler,
            shuffle=False,
            batch_size=self.hparams.valid_batchsize,
            pin_memory=True,
            num_workers=self.hparams.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
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

    def collate_fn(self, samples):
        if self.formator == "t5style":
            batch = {
                k: [
                    torch.tensor(samples[i][k], dtype=torch.int64)
                    for i in range(len(samples))
                ]
                for k in ["input_ids", "attention_mask", "labels"]
            }
        else:
            batch = {
                k: [
                    torch.tensor(samples[i][k], dtype=torch.int64)
                    for i in range(len(samples))
                ]
                for k in ["input_ids", "token_types", "attention_mask", "labels"]
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
        default="/cognitive_comp/hejunqing/projects/Dialog_pretrain/randeng_t5_newvocab_784M",
        type=str,
    )

    total_parser.add_argument(
        "--pretrained_model_path",
        default="/cognitive_comp/hejunqing/projects/Dialog_pretrain/randeng_t5_newvocab_784M",
    )
    total_parser.add_argument("--train_split_size", default=0.995, type=int)
    total_parser.add_argument(
        "--formator", default="t5style", choices=["t5style", "squad", "dialog"]
    )
    total_parser = TextGenDataModel.add_data_specific_args(total_parser)
    args = total_parser.parse_args()
    args.train_data_path = "cmrc"
    ds = T5StyleDataset("cmrc", args, "dev")
    print(len(ds))
    for i in range(10):
        print(ds[i])

    dl = TextGenDataModel(args)
    for i in range(5):
        for batch in dl.val_dataloader():
            print(batch)
            print(batch["input_ids"])
            print(batch["no_answer"])
            print(batch["decoder_input_ids"])
            print(batch["labels"])
