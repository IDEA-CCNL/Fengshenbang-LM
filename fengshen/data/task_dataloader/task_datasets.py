# coding=utf8
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import torch
import pytorch_lightning as pl
import os


class LCSTSDataset(Dataset):
    '''
    Dataset Used for LCSTS summary task.
    '''

    def __init__(self, data_path, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, use_fast=False)
        self.data = self.load_data(data_path)
        self.prompt = args.prompt
        self.max_enc_length = args.max_enc_length
        self.max_dec_length = args.max_dec_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index])

    def load_data(self, data_path):
        with open(data_path, "r", encoding='utf8') as f:
            lines = f.readlines()
        samples = []
        for line in tqdm(lines):
            obj = json.loads(line)
            source = obj['text']
            target = obj['summary']
            samples.append({
                "text": source,
                "summary": target
            })
        return samples

    def cal_data(self, data_path):
        with open(data_path, "r", encoding='utf8') as f:
            lines = f.readlines()
        samples = []
        enc_sizes = []
        dec_sizes = []
        for line in tqdm(lines):
            obj = json.loads(line.strip())
            source = obj['text']
            target = obj['summary']
            enc_input_ids = self.tokenizer.encode(source)
            target = self.tokenizer.encode(target)
            enc_sizes.append(len(enc_input_ids))
            dec_sizes.append(len(target)-1)
            samples.append({
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)
        import numpy as np
        # mean of len(enc_input_ids): 74.68041911345998
        # mean of len(dec_input_ids): 14.02265483791283
        # max of len(enc_input_ids): 132
        # max of len(dec_input_ids): 31
        print('mean of len(enc_input_ids):', np.mean(enc_sizes),
              'mean of len(dec_input_ids):', np.mean(dec_sizes),
              'max of len(enc_input_ids):', max_enc_len,
              'max of len(dec_input_ids):', max_dec_len)
        return samples

    def encode(self, item):
        encode_dict = self.tokenizer.encode_plus(
            self.prompt + item['text'],
            max_length=self.max_enc_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        decode_dict = self.tokenizer.encode_plus(
            item['summary'],
            max_length=self.max_dec_length,
            padding='max_length',
            truncation=True)

        target = decode_dict['input_ids']
        # print('encode_dict shape:', encode_dict['input_ids'].shape)
        labels = torch.tensor(target)
        labels[target == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": encode_dict['input_ids'].squeeze(),
            "attention_mask": encode_dict['attention_mask'].squeeze(),
            "labels": labels.squeeze(),
            "text": item['text'],
            "summary": item['summary']
        }


class LCSTSDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('LCSTSDataModel')
        parser.add_argument(
            '--data_dir', default='/cognitive_comp/ganruyi/data_datasets_LCSTS_LCSTS/', type=str)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_data', default='train.jsonl', type=str)
        parser.add_argument('--valid_data', default='valid.jsonl', type=str)
        parser.add_argument('--test_data', default='test_public.jsonl', type=str)
        parser.add_argument('--train_batchsize', default=128, type=int)
        parser.add_argument('--valid_batchsize', default=128, type=int)
        parser.add_argument('--max_enc_length', default=128, type=int)
        parser.add_argument('--max_dec_length', default=30, type=int)
        parser.add_argument('--prompt', default='summarize:', type=str)
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        if not args.do_eval_only:
            self.train_data = LCSTSDataset(os.path.join(
                args.data_dir, args.train_data), args)
        self.valid_data = LCSTSDataset(os.path.join(
            args.data_dir, args.valid_data), args)
        self.test_data = LCSTSDataset(os.path.join(
            args.data_dir, args.test_data), args)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          shuffle=True,
                          batch_size=self.train_batchsize,
                          pin_memory=False,
                          num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data,
                          shuffle=False,
                          batch_size=self.valid_batchsize,
                          pin_memory=False,
                          num_workers=self.args.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data,
                          shuffle=False,
                          batch_size=self.valid_batchsize,
                          pin_memory=False,
                          num_workers=self.args.num_workers)
