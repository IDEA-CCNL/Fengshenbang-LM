from fengshen.data.fs_datasets.load import load_dataset, list_datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ConcatDataset
import torch
from typing import Optional
from dataclasses import dataclass

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


def get_consume_dataset(name,subname):
    if name is not "dial":
        return load_dataset(name)

    if subname is not "balance":
        return load_dataset(name,subname)

    avail_dataset = list_datasets(name)
    avail_dataset.remove("allmerge") # do not include all merge data

    datasets = [load_dataset(name, ds) for ds in avail_dataset]
    concat_dataset = {
        "train":ConcatDataset([ds["train"] for ds in datasets]),
        "test":ConcatDataset([ds["test"] for ds in datasets]),
        "dev":ConcatDataset([ds["dev"] for ds in datasets])
    }
    return concat_dataset
class MixingDataModule(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--dataloader_workers', default=2, type=int)
        parser.add_argument('--train_batchsize', default=32, type=int)
        parser.add_argument('--val_batchsize', default=32, type=int)
        parser.add_argument('--test_batchsize', default=32, type=int)
        parser.add_argument('--datasets_name', type=str)
        parser.add_argument('--datasets_subname', type=str)
        parser.add_argument('--train_datasets_field', type=str, default='train')
        parser.add_argument('--val_datasets_field', type=str, default='validation')
        parser.add_argument('--test_datasets_field', type=str, default='test')
        parser.add_argument('--sampler_type', type=str,
                            choices=['single',
                                     'random',
                                     'mixing'],
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
        self.datasets = get_consume_dataset(args.datasets_name)
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.save_hyperparameters(args)
        print('---------ending load datasets {}'.format(args.datasets_name))

    def get_custom_sampler(self):
        from universal_datamodule.universal_sampler import PretrainingRandomSampler
        from universal_datamodule.universal_sampler import PretrainingSampler
        from .mixing_sampler import PropMixingRandomSampler, TempMixingRandomSampler
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


# rewarite from pretrain_bart Text Filling Collator
# x = src + kno +  tgt
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
class DialoCollator:
    tokenizer: None
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

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = args.max_seq_length
        
    def generate_sample(self, x):
        # tokenize sentence
        x = self.tokenizer.bos_token + x + self.tokenizer.eos_token
        input_dicts = self.tokenizer.encode_plus(
            x,
            max_length=self.max_seq_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'
        )
        
        input_ids = input_dicts["input_ids"]
        attn_mask = input_dicts["attention_mask"]
        labels = input_ids

        return [input_ids, labels, attn_mask]

    def __call__(self, samples):
        for s in samples:
            s["knowledge"] = s["kno"]

        input_ids, labels, attn_mask = [],[],[]
        for s in samples:
            # 需要补充 bos , eos, 所以最长长度需要-2
            s["knowledge"] = truncate_sequence(s["knowledge"],self.args.max_kno_length-2)
            s["src"] = truncate_sequence(s["src"],self.args.max_src_length-2)
            s["tgt"] = truncate_sequence(s["tgt"],self.args.max_tgt_length-1)

            x_trunc = f'knowledge: {s["knowledge"]} context: {s["src"]} response:{s["tgt"]}' #prompt
         
            g = self.generate_sample(x_trunc)
            
            input_ids.append(g[0])
            labels.append(g[1])
            attn_mask.append(g[2])

        return {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': torch.cat(labels),
            "knowledge": s["knowledge"],
            "question":s["src"]
        }

@dataclass
class QueryCollator:
    tokenizer: None
    max_seq_length: int = 512 
    max_src_length: int = 496
    max_tgt_length: int = 16

    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Bart Text Filling Collator')
        parser.add_argument('--max_seq_length', default=512, type=int) #总序列最长多长
        parser.add_argument('--max_src_length', default=496, type=int) #总序列最长多长
        parser.add_argument('--max_tgt_length', default=16, type=int) #回复最长多长
        return parent_args

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = args.max_seq_length
        
    def generate_sample(self, x):
        # tokenize sentence
        x = self.tokenizer.bos_token + x + self.tokenizer.eos_token
        input_dicts = self.tokenizer.encode_plus(
            x,
            max_length=self.max_seq_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'
        )
        
        input_ids = input_dicts["input_ids"]
        attn_mask = input_dicts["attention_mask"]
        labels = input_ids

        return [input_ids, labels, attn_mask]

    def __call__(self, samples):
        input_ids, labels, attn_mask = [],[],[]
        for s in samples:
            # 需要补充 bos , eos, 所以最长长度需要-2
            s["src"] = truncate_sequence(s["src"],self.args.max_src_length-2)
            s["tgt"] = truncate_sequence(s["tgt"],self.args.max_tgt_length-1)

            x_trunc = f'context: {s["src"]} query:{s["tgt"]}' #prompt
         
            g = self.generate_sample(x_trunc)
            
            input_ids.append(g[0])
            labels.append(g[1])
            attn_mask.append(g[2])

        return {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': torch.cat(labels),
            "query": s["tgt"],
            "question":s["src"]
        }
