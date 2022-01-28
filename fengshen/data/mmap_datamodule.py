import argparse
from ast import Num
from typing import List, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from fengshen.data.mmap_index_dataset import MMapIndexDataset


class MMapDataModule(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('MMAP DataModule')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_batchsize', default=32, type=int)
        parser.add_argument('--eval_batchsize', default=32, type=int)
        parser.add_argument('--test_batchsize', default=32, type=int)
        parser.add_argument('--train_datas', default=[
            './train_datas'
        ], type=str, nargs='+')
        parser.add_argument('--valid_datas', default=[
            './valid_datas'
        ], type=str, nargs='+')
        parser.add_argument('--test_datas', default=[
            './test_datas'],
            type=str, nargs='+')
        parser.add_argument('--input_tensor_name', default=['input_ids'], type=str, nargs='+')
        return parent_args

    def __init__(
        self,
        collate_fn,
        args,
        **kwargs,
    ):
        super().__init__()
        self.collate_fn = collate_fn
        self.train_dataset = MMapIndexDataset(args.train_datas, args.input_tensor_name)
        self.valid_dataset = MMapIndexDataset(args.valid_datas, args.input_tensor_name)
        self.test_dataset = MMapIndexDataset(args.test_datas, args.input_tensor_name)
        self.save_hyperparameters(args)

    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batchsize,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.eval_batchsize,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.test_batchsize,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
