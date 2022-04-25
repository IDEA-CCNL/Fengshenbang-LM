from pytorch_lightning import LightningDataModule
from typing import Optional

from torch.utils.data import DataLoader


class UniversalDataModule(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_batchsize', default=32, type=int)
        parser.add_argument('--val_batchsize', default=32, type=int)
        parser.add_argument('--test_batchsize', default=32, type=int)
        parser.add_argument('--datasets_name', type=str)
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
        from ..fs_datasets import load_dataset
        self.datasets = load_dataset(
            args.datasets_name, num_proc=args.num_workers)
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
            batch_size=self.hparams.eval_batchsize,
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
