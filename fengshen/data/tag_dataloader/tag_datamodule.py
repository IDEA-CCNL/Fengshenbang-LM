from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from fengshen.data.tag_dataloader.tag_datasets import DataProcessor, TaskDataset
import pytorch_lightning as pl


class TaskDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group("DataModel")
        parser.add_argument("--data_dir", default="./data", type=str)
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--train_batchsize", default=16, type=int)
        parser.add_argument("--valid_batchsize", default=16, type=int)
        parser.add_argument("--max_seq_length", default=512, type=int)

        parser.add_argument("--decode_type", default="linear", choices=["linear", "crf", "biaffine", "span"], type=str)

        return parent_args

    def __init__(self, args, collate_fn, tokenizer):
        super().__init__()
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize

        self.tokenizer = tokenizer
        self.collator = collate_fn

        processor = DataProcessor(args.data_dir, args.decode_type)

        self.train_data = TaskDataset(processor=processor, mode="train")
        self.valid_data = TaskDataset(processor=processor, mode="test")
        self.test_data = TaskDataset(processor=processor, mode="test")

        self.save_hyperparameters(args)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, shuffle=True, batch_size=self.train_batchsize, pin_memory=False, collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data, shuffle=False, batch_size=self.valid_batchsize, pin_memory=False, collate_fn=self.collator,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_data, shuffle=False, batch_size=self.valid_batchsize, pin_memory=False, collate_fn=self.collator,
        )
