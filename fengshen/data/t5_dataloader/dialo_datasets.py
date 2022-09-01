# coding=utf8
import json
from json import decoder
import torch
import numpy as np
import sys
import pytorch_lightning as pl
sys.path.append('../../')
from torch.utils.data import Dataset, DataLoader,DistributedSampler
from tqdm import tqdm
from transformers import BertTokenizer, MT5Config, MT5Tokenizer, BatchEncoding
from torch.utils.data.dataset import ConcatDataset
from fengshen.data.t5_dataloader.t5_datasets import TaskT5DataModel, TaskT5Dataset
from fengshen.data.fs_datasets.load import load_dataset, list_datasets

class DialoT5DataModule(pl.LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('DialoT5 DataModule')
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
                            choices=['single','random','mixing'],
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
        self.datasets = load_dataset(args.datasets_name)
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.save_hyperparameters(args)
        print('---------ending load datasets {}'.format(args.datasets_name))

    def get_custom_sampler(self):
        from universal_datamodule.universal_sampler import PretrainingRandomSampler,PretrainingSampler
        from universal_datamodule.universal_datamodule import get_consume_samples
        from gpt_dataloader.mixing_sampler import PropMixingRandomSampler
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
        elif self.hparams.sampler_type == 'mixing':
            return PropMixingRandomSampler(
                total_samples=len(self.datasets[self.hparams.train_datasets_field]),
                # consumed_samples cal by global steps
                batch_size=self.hparams.train_batchsize
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

def truncate_sequence(document:str, max_num_tokens:int,reverse=False):
    total_length = len(document)
    if total_length <= max_num_tokens:
        return document
    else: 
        if reverse:
            return document[-1*max_num_tokens:]
        else:
            return document[:max_num_tokens]

def padding_to_maxlength(ids, max_length, pad_id):
    cur_len = len(ids)
    len_diff = max_length - len(ids)
    return ids + [pad_id] * len_diff, [1] * cur_len + [0] * len_diff


@dataclass
class DialoT5Collator:
    tokenizer_type: str = 't5_tokenizer'
    max_seq_length: int = 512 
    max_kno_length: int = 256
    max_src_length: int = 128
    max_tgt_length: int = 128

    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Wenzhong Text Filling Collator')
        parser.add_argument('--max_seq_length', default=512, type=int) #总序列最长多长
        parser.add_argument('--max_src_length', default=256, type=int) #总序列最长多长
        parser.add_argument('--max_kno_length', default=128, type=int) #知识最长多长
        parser.add_argument('--max_tgt_length', default=128, type=int) #回复最长多长
        return parent_args

    def __init__(self, args):
        self.args = args
        if args.tokenizer_type == 't5_tokenizer':
            self.tokenizer = MT5Tokenizer.from_pretrained(args.pretrained_model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
        self.max_seq_length = args.max_seq_length
        
    def encode(self, x, y):
        """
        参考 Unified QA 
        # https://github.com/allenai/unifiedqa/blob/master/bart/unified_data.py
        """
        # tokenize sentence
        x = self.tokenizer.bos_token + x + self.tokenizer.eos_token
        y = y + self.tokenizer.eos_token

        encoder_input = self.tokenizer.encode_plus(
            x,
            max_length=self.max_kno_length+ self.max_src_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'
        )
        decoder_input = self.tokenizer.encode_plus(
            y,
            max_length=self.max_seq_length+self.max_tgt_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'       
        )

        return [encoder_input, decoder_input]

    def __call__(self, samples):
        for s in samples:
            s["knowledge"] = s["kno"] # 兼容不同数据集键名

        input_ids,  attn_mask, decoder_input_ids, decoder_attn_mask = [],[],[],[]
        for s in samples:
            # 需要补充 prompt(2) bos(1), eos(1)，所以最长长度 -3
            # bos prompt [kno] prompt [src] eos
            s["knowledge"] = truncate_sequence(s["knowledge"],self.args.max_kno_length-3)
            s["src"] = truncate_sequence(s["src"],self.args.max_src_length-3, reverse=True) # 倒叙截取上下文问句，以提升对最近问句的相应
            s["tgt"] = truncate_sequence(s["tgt"],self.args.max_tgt_length-1)#后面要加 eos

            x_trunc = f'knowledge: {s["knowledge"]} context: {s["src"]}' #prompt
            y_trunc = f'{s["tgt"]}'
            encoder_input, decoder_input = self.encode(x_trunc,y_trunc)
            
            input_ids.append(encoder_input[0])
            attn_mask.append(encoder_input[1])
            decoder_input_ids.append(decoder_input[2])
            decoder_attn_mask.append(decoder_input[2])

        return {
            'input_ids': torch.cat(input_ids),
            'attn_mask': torch.cat(attn_mask),
            'labels': torch.cat(decoder_input_ids),
            'decoder_attn_mask':torch.cat(decoder_attn_mask),
            'knowledge': s["knowledge"],
            'question':s["src"]
        }

