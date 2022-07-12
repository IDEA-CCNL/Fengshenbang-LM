# !deprecated this version 
# coding=utf8
 
import os
import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2Model
import pytorch_lightning as pl

os.environ["CUDA_VISIBLE_DEVICES"]="4,7"
class DuSincDataset(Dataset):
    '''
    Dataset Used for yuyuan medical qa task.
    Just surpport small datasets, when deal with large datasets it may be slowly.
    for large datasets please use mmapdatasets(doing)
    '''

    def __init__(self, data_path, name, task_id, max_seq_length=1024, is_test=False):
        super().__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained(
        #    args.pretrained_model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong-GPT2-3.5B')
        self.tokenizer.add_special_tokens({'pad_token': "[PAD]"}) #[PAD]
        self.tokenizer.add_special_tokens({'bos_token': "[SEP]"}) # <s>
        # self.tokenizer.add_special_tokens({'eos_token': "</s>"}) # </s>
        # self.tokenizer.add_special_tokens({'unk_token': "<unk>"}) # <unk>]
        self.data_size = os.path.getsize(data_path)/1024/1024/1024
        self.data_type_name = name
        self.data = self.load_data(data_path)
        self.task_id = task_id
        self.is_test = is_test

        # args
        self.max_seq_length = max_seq_length
        # self.max_src_length = args.max_src_length
        # self.max_tgt_length = args.max_tgt_length
        # self.max_kno_length = args.max_kno_length
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode_baseline(self.data[index])

    def load_data(self, data_path):
        # 有进度条展示
        with open(data_path, "rt", encoding='utf8') as f:
            lines = f.readlines()

        total_num = len(lines)
        data = []
        with tqdm(total=total_num, desc='处理进度', mininterval=0.3) as bar:
            for idx, line in enumerate(lines[1:]):
                data.append(line)
                #data.append(self.data_parse(line))
                bar.update()

        return data

    def _parse_text(self, src, token_type="src"):
        """Parse source sequence and return corresponding fields."""
        if src is None:
            return src

        # initialize 
        input_ids_list, token_type_ids_list, pos_ids_list = [],[],[]

        # get token type ids
        token_type_dict = {
            "src": 0,
            "tgt": 1,
            "kno": 2,
        }
        token_type_id = token_type_dict[token_type]
        max_length_dict = {
            "src" : int(self.max_seq_length / 3), 
            "tgt" : int(self.max_seq_length / 3), 
            "kno" : int(self.max_seq_length/ 3)
        }
        max_s_length = max_length_dict[token_type]
        max_s_length = self.max_seq_length

        # split sentence by [SEP]
        for s in src.split("[SEP]"):
            s = s.strip()

            # tokenize sentence
            input_dicts = self.tokenizer.encode_plus(
                s,
                max_length=max_s_length-2, # [bos] + xx + [eos] 
                padding="max_length",
                truncation=True, 
                return_tensors='pt'
            )
            
            # parse input dicts
            input_ids = torch.cat([
                    torch.tensor([self.tokenizer.bos_token_id]), 
                    input_dicts['input_ids'].squeeze(), 
                    torch.tensor([self.tokenizer.eos_token_id]
                )] , dim=0)
            
            pos_ids = torch.tensor(list(range(0, input_ids.shape[0])))
            token_type_ids = torch.tensor([token_type_id] * input_ids.shape[0])

            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            pos_ids_list.append(pos_ids)

            
        return {
            "token_ids": torch.cat(input_ids_list)[:max_s_length],
            "token_type_ids": torch.cat(token_type_ids_list)[:max_s_length],
            "pos_ids": torch.cat(pos_ids_list)[:max_s_length],
        }

    def _merge_input(self, dict1, dict2):
        """Merge(concatenate) two input dicts."""
        if dict1 is None:
            return dict2
        if dict2 is None:
            return dict1
        return {
            "token_ids": torch.cat([dict1["token_ids"], dict2["token_ids"]]),
            "token_type_ids": torch.cat([dict1["token_type_ids"], dict2["token_type_ids"]]),
            "pos_ids": torch.cat([dict1["pos_ids"], dict2["pos_ids"]]),
        }

    def encode_baseline(self, item):
        """
        将数据转换成模型训练的输入(模仿baseline但是baseline是encoder-decoder结构，这里只有decoder)
        """
        # test mode + dialogue : src + kno
        # test mode + query : src
        # train mode + dialogue : src + tgt + kno
        # train mode + query : src + tgt
        if "\t" in item[:-1]:
            outputs = item[:-1].split('\t')
        else:
            outputs = [item[:-1]]

        if self.is_test:
            tgt = None
            if self.task_id == 1:
                kno, src, tgt = None, outputs[0], None
            else:
                kno, src, tgt = outputs[0], outputs[1], None
            print(kno, src)
        else:
            if self.task_id == 1:
                kno, src, tgt = None, outputs[0], outputs[1]
            else:
                kno, src, tgt = outputs[0], outputs[1], outputs[2]

        src_input_dicts = self._parse_text(src, token_type="src")
        tgt_input_dicts = self._parse_text(tgt, token_type="tgt")
        kno_input_dicts = self._parse_text(kno, token_type="kno")
       
        # merge input dicts
        input_dicts = self._merge_input(src_input_dicts, kno_input_dicts)
        input_dicts = self._merge_input(input_dicts, tgt_input_dicts)  

        # use relative pos ids then remain
        # use continuous pos ids then do
        input_dicts["pos_ids"] = torch.tensor(list(range(0, input_dicts["token_ids"].shape[0])))
        input_dicts["labels"] = torch.tensor([-1]*input_dicts["token_ids"].shape[0])

        return input_dicts

    def encode(self,item):
        """Encoder for GPT"""
        # if '\t' in item:
        #     outputs = item.split('\t')

        # if not self.is_test:            
        #     kno, src, tgt = outputs[0], outputs[1], outputs[2]
        #     inputs = kno + src
        # else: 
        #     kno, src = outputs[0], outputs[1]

        encoded_input = self.tokenizer(
            item[:-1],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        encoded_label = self.tokenizer(
            tgt,
            max_length = self.max_seq_length,
            padding = "max_length",
            truncation = True,
            return_tensor = 'pt'
        )
        if not self.is_test:
            encoded_input["labels"] = encoded_label
        
        return encoded_input
            
def collate_fn(batch):

    x1, x2,  y = [], [],  []
    for unit in batch:
        x1.append(unit["input_ids"])
        x2.append(unit["attention_mask"])
        y.append(unit["labels"])
    print (x1,x2,y)
    return torch.cat(x1), torch.cat(x2), torch.cat(y)

class DuSincDataLoader(pl.LightningDataModule):
    def __init__(self, task="dialogue", train_batch_size=32, eval_batch_size=32, max_seq_length=128, data_dir="./data", **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.task_name = task
        self.task_id = 2 if task == "dialogue" else 1
        if self.task_id == 1:
            self.train_data, self.valid_data, self.test_data = "train_query.txt", "dev_query.txt", "test_query.txt"
        else:
            self.train_data, self.valid_data, self.test_data = "train_dial.txt", "dev_dial.txt", "test_dial.txt"

        # self.text_fields = self.task_text_field_map[task]
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        self.prepare_data()

    def prepare_data(self):
        self.dataset = {
            "train": DuSincDataset(os.path.join(self.data_dir, self.train_data), '训练集', task_id=self.task_id, max_seq_length=self.max_seq_length),
            "test": DuSincDataset(os.path.join(self.data_dir, self.valid_data), '验证集',task_id=self.task_id, max_seq_length=self.max_seq_length),
            "dev": DuSincDataset(os.path.join(self.data_dir, self.valid_data), '测试集',task_id=self.task_id, max_seq_length=self.max_seq_length),
        }

        print("Prepare data")
        print("Tokenizer data")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            #collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Return validation data"""
        return DataLoader(
            self.dataset["dev"], batch_size=self.eval_batch_size, 
            shuffle=False,
            #collate_fn=collate_fn
        )

    def test_dataloader(self):
        """Return test data"""
        return DataLoader(
            self.dataset["dev"], batch_size=self.eval_batch_size, 
            shuffle=False,
            #collate_fn=collate_fn
        ) 
        
class DuSincDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('DuSinc Data')
        parser.add_argument('--data_dir', type=str, required=True)
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--train_data', default='train.txt', type=str)
        parser.add_argument('--valid_data', default='valid.txt', type=str)
        parser.add_argument('--test_data', default='test.txt', type=str)
        parser.add_argument('--train_batchsize', type=int, required=True)
        parser.add_argument('--valid_batchsize', type=int, required=True)
        parser.add_argument('--max_seq_length', default=1024, type=int)
        parser.add_argument('--max_src_length', default=256, type=int)
        parser.add_argument('--max_tgt_length', default=256, type=int)
        parser.add_argument('--max_kno_length', default=512, type=int)
        parser.add_argument('--task_id', default=1, type=int) # if 1 is query else 0 is dialogue
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        if not args.do_eval_only:
            self.train_data = DuSincDataset(os.path.join(
                args.data_dir, args.train_data), '训练集', args)
            self.valid_data = DuSincDataset(os.path.join(
                args.data_dir, args.valid_data), '验证集', args)
        self.test_data = DuSincDataset(os.path.join(
            args.data_dir, args.test_data), '测试集', args, is_test=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, shuffle=True,
            batch_size=self.train_batchsize,
            collate_fn=collate_fn,
            pin_memory=False, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False,
                          batch_size=self.valid_batchsize,
                          collate_fn=collate_fn,
                          pin_memory=False, num_workers=self.args.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data, shuffle=False,
                          batch_size=self.valid_batchsize,
                          pin_memory=False,
                          collate_fn=collate_fn,
                          num_workers=self.args.num_workers)

#test dataset
import argparse
def testDuSincDatase():
    total_parser = argparse.ArgumentParser("QA Task")
    total_parser.add_argument('--do_eval_only', action='store_true', default=False)
    total_parser.add_argument('--pretrained_model_path', default='google/mt5-small', type=str)
    total_parser.add_argument('--output_save_path', default='./predict.json', type=str)
    total_parser = DuSincDataModel.add_data_specific_args(total_parser)
    args = total_parser.parse_args()

    # args.data_dir = "cognitive_comp/yangqi/data/DuSinc"
    # args.train_data = "train_query.txt"
    # args.valid_data = "dev_query.txt"
    # args.test_data = "test_query.txt"
    # args.max_seq_length =1024
    data_model = DuSincDataModel(args)
    # dataset = DuSincDataset(os.path.join(
    #     args.data_dir, args.train_data), '训练集', args)
    # dataset = DuSincDataset(os.path.join(
    #     args.data_dir, args.valid_data), '验证集', args)
    dataset = DuSincDataset(os.path.join(
        args.data_dir, args.test_data), '测试集', args, is_test=True)
    print(dataset[0])



# ===========================================
from pytorch_lightning import LightningDataModule
from typing import Optional

from torch.utils.data import DataLoader, DistributedSampler


def get_consume_samples(data_model: LightningDataModule) -> int:
    if hasattr(data_model.trainer.lightning_module, 'consumed_samples'):
        consumed_samples = data_model.trainer.lightning_module.consumed_samples
        print('get consumed samples from model: {}'.format(consumed_samples))
    else:
        world_size = data_model.trainer.world_size
        consumed_samples = max(0, data_model.trainer.global_step - 1) * \
            data_model.hparams.train_batchsize * world_size * data_model.hparams.accumulate_grad_batches
        print('calculate consumed samples: {}'.format(consumed_samples))
    return consumed_samples

class DusincDataModule(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--dataloader_workers', default=2, type=int)
        parser.add_argument('--train_batchsize', default=32, type=int)
        parser.add_argument('--val_batchsize', default=32, type=int)
        parser.add_argument('--test_batchsize', default=32, type=int)
        parser.add_argument('--datasets_name', type=str)
        parser.add_argument('--train_datasets_field', type=str, default='train')
        parser.add_argument('--val_datasets_field', type=str, default='validation')
        parser.add_argument('--test_datasets_field', type=str, default='test')
        parser.add_argument('--sampler_type', type=str,
                            choices=['single',
                                     'random'],
                            default='random')
        return parent_args

    def __init__(
        self,
        tokenizer,
        collate_fn,
        args,
        **kwargs,
    ):
        super().__init__()
        print('---------begin to load datasets {}'.format(args.datasets_name), flush=True)
        from ..fs_datasets import load_dataset
        self.datasets = load_dataset(
            args.datasets_name
        )
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.save_hyperparameters(args)
        print('---------ending load datasets {}'.format(args.datasets_name))

    def get_custom_sampler(self):
        from universal_datamodule.universal_sampler import PretrainingRandomSampler
        from universal_datamodule.universal_sampler import PretrainingSampler
        world_size = self.trainer.world_size
        consumed_samples = get_consume_samples(self)
        # use the user default sampler
        if self.hparams.sampler_type == 'random':
            return PretrainingRandomSampler(
                total_samples=len(self.datasets[self.hparams.train_datasets_field]),
                # consumed_samples cal by global steps
                consumed_samples=consumed_samples,
                micro_batch_size=self.hparams.train_batchsize,
                data_parallel_rank=self.trainer.global_rank,
                data_parallel_size=world_size,
                epoch=self.trainer.current_epoch,
            )
        elif self.hparams.sampler_type == 'single':
            return PretrainingSampler(
                total_samples=len(self.datasets[self.hparams.train_datasets_field]),
                # consumed_samples cal by global steps
                consumed_samples=consumed_samples,
                micro_batch_size=self.hparams.train_batchsize,
                data_parallel_rank=self.trainer.global_rank,
                data_parallel_size=world_size,
            )
        else:
            raise Exception('Unknown sampler type: {}'.format(self.hparams.sampler_type))

    def setup(self, stage: Optional[str] = None) -> None:
        return

    def train_dataloader(self):
        if self.hparams.replace_sampler_ddp is False:
            return DataLoader(
                self.datasets[self.hparams.train_datasets_field],
                batch_sampler=self.get_custom_sampler(),
                num_workers=self.hparams.dataloader_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
        return DataLoader(
            self.datasets[self.hparams.train_datasets_field],
            batch_size=self.hparams.train_batchsize,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[self.hparams.val_datasets_field],
            batch_size=self.hparams.val_batchsize,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            sampler=DistributedSampler(
                self.datasets[self.hparams.val_datasets_field], shuffle=False),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[self.hparams.test_datasets_field],
            batch_size=self.hparams.test_batchsize,
            shuffle=False,
            num_workers=self.hparams.dataloader_workers,
            collate_fn=self.collate_fn,
            sampler=DistributedSampler(
                self.datasets[self.hparams.test_datasets_field], shuffle=False),
            pin_memory=True,
        )

"""
python data/dusinc_dataloader/dusinc_dataset.py --data_dir="/cognitive_comp/yangqi/data/DuSinc_release" --train_data="train.txt"  --test_data="test_query_1.txt"  --valid_data="dev.txt"  --train_batchsize=32  --valid_batchsize=32 --max_seq_length=1024
python data/dusinc_dataloader/dusinc_dataset.py --data_dir="/cognitive_comp/yangqi/data/DuSinc" --train_data="train_query.txt"  --test_data="test_query.txt"  --valid_data="dev_query.txt"  --train_batchsize=32  --valid_batchsize=32 --max_seq_length=1024
python data/dusinc_dataloader/dusinc_dataset.py --data_dir="/cognitive_comp/yangqi/data/DuSinc" --train_data="train_dial.txt"  --test_data="test_dial.txt"  --valid_data="dev_dial.txt"  --train_batchsize=32  --valid_batchsize=32 --max_seq_length=256
"""