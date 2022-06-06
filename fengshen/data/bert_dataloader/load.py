import os
import re
from pathlib import Path
import glob
from tqdm import tqdm
from contextlib import ExitStack
import datasets
import multiprocessing
from typing import cast,TextIO
from itertools import chain
import json
from concurrent.futures import ProcessPoolExecutor
from random import shuffle
from pytorch_lightning import LightningDataModule
from typing import Optional

from torch.utils.data import DataLoader


# _SPLIT_DATA_PATH = '/data1/datas/wudao_180g_split/test'
_SPLIT_DATA_PATH = '/data1/datas/wudao_180g_split'
_CACHE_SPLIT_DATA_PATH = '/data1/datas/wudao_180g_FSData'

# feats = datasets.Features({"text": datasets.Value('string')})

class BertDataGenerate(object):

    def __init__(self,
            data_files=_SPLIT_DATA_PATH,
            save_path=_CACHE_SPLIT_DATA_PATH,
            train_test_validation = '950,49,1',
            num_proc=1,
            cache=True):
        self.data_files = Path(data_files)
        if save_path:
            self.save_path = Path(save_path)
        else:
            self.save_path = self.file_check( 
                Path(self.data_files.parent,self.data_files.name+'_FSDataset'),
                'save')
        self.num_proc = num_proc
        self.cache = cache
        self.split_idx = self.split_train_test_validation_index(train_test_validation)
        if cache:
            self.cache_path = self.file_check(
                Path(self.save_path.parent,'FSDataCache',self.data_files.name),'cache')
        else:
            self.cache_path = None
    
    @staticmethod
    def file_check(path,path_type):
        print(path)
        if not path.exists():
            path.mkdir(parents=True)
        print(f"Since no {path_type} directory is specified, the program will automatically create it in {path} directory.")
        return str(path)
    
    @staticmethod
    def split_train_test_validation_index(train_test_validation):
        split_idx_ = [int(i) for i in train_test_validation.split(',')]
        idx_dict = {
            'train_rate':split_idx_[0]/sum(split_idx_),
            'test_rate':split_idx_[1]/sum(split_idx_[1:])
        }
        return idx_dict

    def process(self,index, path):
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
            'train':ds['train'],
            'test':ds_['train'],
            'validation':ds_['test']
        })
        # print('!!!!',ds)
        ds.save_to_disk(Path(self.save_path,path.name))
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
        'train':datasets.concatenate_datasets(train),
        'test':datasets.concatenate_datasets(test),
        'validation':datasets.concatenate_datasets(validation)
    })


class BertDataModule(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_batchsize', default=32, type=int)
        parser.add_argument('--val_batchsize', default=32, type=int)
        parser.add_argument('--test_batchsize', default=32, type=int)
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
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
        self.test = DataLoader(
            self.datasets[self.hparams.test_datasets_field],
            batch_size=self.hparams.test_batchsize,
            shuffle=False,
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

if __name__ == '__main__':
    # pre = PreProcessing(_SPLIT_DATA_PATH)
    # pre.processing()

    dataset = BertDataGenerate(_SPLIT_DATA_PATH,num_proc=16)
    dataset.generate_cache_arrow()
