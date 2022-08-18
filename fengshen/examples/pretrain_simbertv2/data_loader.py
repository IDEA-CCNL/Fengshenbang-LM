import os
import re
from pathlib import Path
import glob
from tqdm import tqdm
from contextlib import ExitStack
import datasets
import multiprocessing
from typing import cast, TextIO
from itertools import chain
import json
from concurrent.futures import ProcessPoolExecutor
from random import shuffle
from pytorch_lightning import LightningDataModule
from typing import Optional
import torch
import jieba
import numpy as np
from torch.utils.data import DataLoader


# _SPLIT_DATA_PATH = '/data1/datas/wudao_180g_split/test'
_SPLIT_DATA_PATH = '/raid/wuziwei/data_test/simsentence'
_CACHE_SPLIT_DATA_PATH = '/raid/wuziwei/data_test/simsentence_FS'

# feats = datasets.Features({"text": datasets.Value('string')})

class BertDataGenerate(object):

    def __init__(self,
                 data_files=_SPLIT_DATA_PATH,
                 save_path=_CACHE_SPLIT_DATA_PATH,
                 train_test_validation='950,49,1',
                 num_proc=1,
                 cache=True):
        self.data_files = Path(data_files)
        if save_path:
            self.save_path = Path(save_path)
        else:
            self.save_path = self.file_check(
                Path(self.data_files.parent, self.data_files.name+'_FSDataset'),
                'save')
        self.num_proc = num_proc
        self.cache = cache
        self.split_idx = self.split_train_test_validation_index(train_test_validation)
        if cache:
            self.cache_path = self.file_check(
                Path(self.save_path.parent, 'FSDataCache', self.data_files.name), 'cache')
        else:
            self.cache_path = None

    @staticmethod
    def file_check(path, path_type):
        print(path)
        if not path.exists():
            path.mkdir(parents=True)
        print(f"Since no {path_type} directory is specified, the program will automatically create it in {path} directory.")
        return str(path)

    @staticmethod
    def split_train_test_validation_index(train_test_validation):
        split_idx_ = [int(i) for i in train_test_validation.split(',')]
        idx_dict = {
            'train_rate': split_idx_[0]/sum(split_idx_),
            'test_rate': split_idx_[1]/sum(split_idx_[1:])
        }
        return idx_dict

    def process(self, index, path):
        print('saving dataset shard {}'.format(index))

        ds = (datasets.load_dataset('json', data_files=str(path),
                                    cache_dir=self.cache_path,
                                    features=None))
        # ds = ds.map(self.cut_sent,input_columns='text')
        # print(d)
        # print('!!!',ds)
        ds = ds['train'].train_test_split(train_size=self.split_idx['train_rate'])
        ds_ = ds['test'].train_test_split(train_size=self.split_idx['test_rate'])
        ds = datasets.DatasetDict({
            'train': ds['train'],
            'test': ds_['train'],
            'validation': ds_['test']
        })
        # print('!!!!',ds)
        ds.save_to_disk(Path(self.save_path, path.name))
        return 'saving dataset shard {} done'.format(index)

    def generate_cache_arrow(self) -> None:
        '''
        生成HF支持的缓存文件，加速后续的加载
        '''
        data_dict_paths = self.data_files.rglob('*')
        p = ProcessPoolExecutor(max_workers=self.num_proc)
        res = list()

        for index, path in enumerate(data_dict_paths):
            res.append(p.submit(self.process, index, path))

        p.shutdown(wait=True)
        for future in res:
            print(future.result(), flush=True)


def load_dataset(num_proc=4, **kargs):
    cache_dict_paths = Path(_CACHE_SPLIT_DATA_PATH).glob('*')
    ds = []
    res = []
    p = ProcessPoolExecutor(max_workers=num_proc)
    for path in cache_dict_paths:
        res.append(p.submit(datasets.load_from_disk,
                            str(path), **kargs))

    p.shutdown(wait=True)
    for future in res:
        ds.append(future.result())
        # print(future.result())
    train = []
    test = []
    validation = []
    for ds_ in ds:
        train.append(ds_['train'])
        test.append(ds_['test'])
        validation.append(ds_['validation'])
    # ds = datasets.concatenate_datasets(ds)
    # print(ds)
    return datasets.DatasetDict({
        'train': datasets.concatenate_datasets(train),
        'test': datasets.concatenate_datasets(test),
        'validation': datasets.concatenate_datasets(validation)
    })


class BertDataModule(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_batchsize', default=4, type=int)
        parser.add_argument('--val_batchsize', default=4, type=int)
        parser.add_argument('--test_batchsize', default=4, type=int)
        parser.add_argument('--datasets_name', type=str)
        # parser.add_argument('--datasets_name', type=str)
        parser.add_argument('--train_datasets_field', type=str, default='train')
        parser.add_argument('--val_datasets_field', type=str, default='validation')
        parser.add_argument('--test_datasets_field', type=str, default='test')
        return parent_args

    def __init__(
        self,
        tokenizer,
        collate_fn,
        args,
        **kwargs,
    ):
        super().__init__()
        self.datasets = load_dataset(num_proc=args.num_workers)
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.save_hyperparameters(args)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train = DataLoader(
            self.datasets[self.hparams.train_datasets_field],
            batch_size=self.hparams.train_batchsize,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
        self.val = DataLoader(
            self.datasets[self.hparams.val_datasets_field],
            batch_size=self.hparams.val_batchsize,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
        self.test = DataLoader(
            self.datasets[self.hparams.test_datasets_field],
            batch_size=self.hparams.test_batchsize,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
        return

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test


class DataCollate(object):

    def __init__(self, tokenizer, max_length, mask_rate=0.15, max_ngram=3, if_padding=True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.word_cuter = jieba.cut
        self.vocab_length = len(tokenizer)
        self.mask_rate = mask_rate
        self.ignore_labels = -100
        self.ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
        self.pvals = pvals
        self.padding = if_padding

    def token_process(self, token_id):
        rand = np.random.random()
        if rand <= 0.8:
            return self.tokenizer.mask_token_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(1, self.vocab_length)

    def __call__(self, samples):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        batch_labels = []
        origin_text = []
        # print('^-^ batch size :',len(samples))
        # print(samples[1])
        samples_ = []
        sim_labels = []
        for i in samples:
            # print(i)
            samples_.append({'text':i['text_a'],'sim':i['label']})
            samples_.append({'text':i['text_b'],'sim':i['label']})
        
        for sample in samples_:
            origin_text.append(sample['text'])
            sim_labels.append(sample['sim'])
            word_list = list(self.word_cuter(sample['text']))
            mask_ids, labels = [], []

            record = []
            for i in range(len(word_list)):
                rands = np.random.random()
                if i in record:
                    continue
                word = word_list[i]
                if rands > self.mask_rate and len(word) < 4:
                    word = word_list[i]
                    word_encode = self.tokenizer.encode(word, add_special_tokens=False)
                    for token in word_encode:
                        mask_ids.append(token)
                        labels.append(self.ignore_labels)
                    record.append(i)
                else:
                    n = np.random.choice(self.ngrams, p=self.pvals)
                    for index in range(n):
                        ind = index + i
                        if ind in record or ind >= len(word_list):
                            continue
                        record.append(ind)
                        word = word_list[ind]
                        word_encode = self.tokenizer.encode(word, add_special_tokens=False)
                        for token in word_encode:
                            mask_ids.append(self.token_process(token))
                            labels.append(token)
            if self.padding:
                if len(mask_ids) > self.max_length:
                    input_ids.append(mask_ids[:self.max_length])
                    batch_labels.append(labels[:self.max_length])
                else:
                    lenght = len(mask_ids)
                    mask_ids.extend([0]*(self.max_length-lenght))
                    labels.extend([-100]*(self.max_length-lenght))
                    input_ids.append(mask_ids)
                    batch_labels.append(labels)
            attention_mask.append([1]*self.max_length)
            token_type_ids.append([0]*self.max_length)
        #     print('sentence:',sample['text'])
        #     print('input_ids:',mask_ids)
        #     print('decode inputids:',self.tokenizer.decode(mask_ids))
        #     print('labels',labels)
        #     print('decode labels:',self.tokenizer.decode(labels))
        #     print('*'*20)
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(batch_labels),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids)
            # 'text':origin_text,
            # 'sim_label':sim_labels
        }


if __name__ == '__main__':

    dataset = BertDataGenerate(_SPLIT_DATA_PATH, num_proc=32)
    dataset.generate_cache_arrow()
    # import argparse
    # from transformers import BertTokenizer

    # args_parser = argparse.ArgumentParser()
    # args_parser = BertDataModule.add_data_specific_args(args_parser)
    # args = args_parser.parse_args()
    # tokenizer = BertTokenizer.from_pretrained('/raid/wuziwei/pretrained_model_hf/bert_base4wudao')
    # collate_fn = DataCollate(tokenizer, 512)
    # data_module = BertDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    # data_module.setup()
    # dtrain = data_module.test_dataloader()
    # for idx,batch in enumerate(dtrain):
    #     if idx == 1:
    #         break
    #     print(batch['input_ids'].shape)
    #     print(batch['text'][0],batch['sim_label'][0])
    #     # print(tokenizer.decode(batch['input_ids'][0]))
    #     print(batch['text'][1],batch['sim_label'][1])
    #     # print(tokenizer.decode(batch['input_ids'][1]))

