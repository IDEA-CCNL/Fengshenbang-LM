# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from fengshen.models.zen1 import ZenModel
from dataclasses import dataclass
from fengshen.models.megatron_t5 import T5Config
from fengshen.models.megatron_t5 import T5EncoderModel
from fengshen.models.roformer import RoFormerConfig
from fengshen.models.roformer import RoFormerModel
from fengshen.models.longformer import LongformerConfig
from fengshen.models.longformer import LongformerModel
import numpy as np
import os
from tqdm import tqdm
import json
import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import (
    BertModel,
    BertConfig,
    MegatronBertModel,
    MegatronBertConfig,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
)
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'


model_dict = {'huggingface-bert': BertModel,
              'fengshen-roformer': RoFormerModel,
              'huggingface-megatron_bert': MegatronBertModel,
              'fengshen-megatron_t5': T5EncoderModel,
              'fengshen-longformer': LongformerModel,
              'fengshen-zen1': ZenModel,
              'huggingface-auto': AutoModel,
              }


config_dict = {'huggingface-bert': BertConfig,
               'fengshen-roformer': RoFormerConfig,
               'huggingface-megatron_bert': MegatronBertConfig,
               'fengshen-megatron_t5': T5Config,
               'fengshen-longformer': LongformerConfig,
               'fengshen-zen1': BertConfig,
               'huggingface-auto': AutoConfig}


class TaskDataset(Dataset):
    def __init__(self, data_path, args, label2id):
        super().__init__()
        self.args = args
        self.label2id = label2id
        self.max_length = args.max_length
        self.data = self.load_data(data_path, args)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self, data_path, args):
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            samples = []
            for line in tqdm(lines):
                data = json.loads(line)
                text_id = int(data[args.id_name]
                              ) if args.id_name in data.keys() else 0
                texta = data[args.texta_name] if args.texta_name in data.keys(
                ) else ''
                textb = data[args.textb_name] if args.textb_name in data.keys(
                ) else ''
                labels = self.label2id[data[args.label_name]
                                       ] if args.label_name in data.keys() else 0
                samples.append({args.texta_name: texta, args.textb_name: textb,
                                args.label_name: labels, 'id': text_id})
        return samples


@dataclass
class TaskCollator:
    args = None
    tokenizer = None

    def __call__(self, samples):
        sample_list = []
        for item in samples:
            if item[self.args.texta_name] != '' and item[self.args.textb_name] != '':
                if self.args.model_type != 'fengshen-roformer':
                    encode_dict = self.tokenizer.encode_plus(
                        [item[self.args.texta_name], item[self.args.textb_name]],
                        max_length=self.args.max_length,
                        padding='max_length',
                        truncation='longest_first')
                else:
                    encode_dict = self.tokenizer.encode_plus(
                        [item[self.args.texta_name]+'[SEP]'+item[self.args.textb_name]],
                        max_length=self.args.max_length,
                        padding='max_length',
                        truncation='longest_first')
            else:
                encode_dict = self.tokenizer.encode_plus(
                    item[self.args.texta_name],
                    max_length=self.args.max_length,
                    padding='max_length',
                    truncation='longest_first')
            sample = {}
            for k, v in encode_dict.items():
                sample[k] = torch.tensor(v)
            sample['labels'] = torch.tensor(item[self.args.label_name]).long()
            sample_list.append(sample)
        return default_collate(sample_list)


class TaskDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--data_dir', default='./data', type=str)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--valid_data', default='dev.json', type=str)
        parser.add_argument('--test_data', default='test.json', type=str)
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--valid_batchsize', default=32, type=int)
        parser.add_argument('--max_length', default=128, type=int)

        parser.add_argument('--texta_name', default='text', type=str)
        parser.add_argument('--textb_name', default='sentence2', type=str)
        parser.add_argument('--label_name', default='label', type=str)
        parser.add_argument('--id_name', default='id', type=str)

        parser.add_argument('--dataset_name', default=None, type=str)

        return parent_args

    def __init__(self, args):
        super().__init__()
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path)
        self.collator = TaskCollator()
        self.collator.args = args
        self.collator.tokenizer = self.tokenizer
        if args.dataset_name is None:
            self.label2id, self.id2label = self.load_schema(os.path.join(
                args.data_dir, args.train_data), args)
            self.train_data = TaskDataset(os.path.join(
                args.data_dir, args.train_data), args, self.label2id)
            self.valid_data = TaskDataset(os.path.join(
                args.data_dir, args.valid_data), args, self.label2id)
            self.test_data = TaskDataset(os.path.join(
                args.data_dir, args.test_data), args, self.label2id)
        else:
            import datasets
            ds = datasets.load_dataset(args.dataset_name)
            self.train_data = ds['train']
            self.valid_data = ds['validation']
            self.test_data = ds['test']
        self.save_hyperparameters(args)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batchsize, pin_memory=False,
                          collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, batch_size=self.valid_batchsize, pin_memory=False,
                          collate_fn=self.collator)

    def predict_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.valid_batchsize, pin_memory=False,
                          collate_fn=self.collator)

    def load_schema(self, data_path, args):
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            label_list = []
            for line in tqdm(lines):
                data = json.loads(line)
                labels = data[args.label_name] if args.label_name in data.keys(
                ) else 0
                if labels not in label_list:
                    label_list.append(labels)

        label2id, id2label = {}, {}
        for i, k in enumerate(label_list):
            label2id[k] = i
            id2label[i] = k
        return label2id, id2label


class taskModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = config_dict[args.model_type].from_pretrained(
            args.pretrained_model_path)
        print('args mode type:', args.model_type)
        self.bert_encoder = model_dict[args.model_type].from_pretrained(
            args.pretrained_model_path)
        self.cls_layer = torch.nn.Linear(
            in_features=self.config.hidden_size, out_features=self.args.num_labels)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        if self.args.model_type == 'fengshen-megatron_t5':
            bert_output = self.bert_encoder(
                input_ids=input_ids, attention_mask=attention_mask)  # (bsz, seq, dim)
            encode = bert_output.last_hidden_state[:, 0, :]
        else:
            bert_output = self.bert_encoder(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # (bsz, seq, dim)
            encode = bert_output[1]
        logits = self.cls_layer(encode)
        if labels is not None:
            loss = self.loss_func(logits, labels.view(-1,))
            return loss, logits
        else:
            return 0, logits


class LitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--num_labels', default=2, type=int)

        return parent_args

    def __init__(self, args, num_data):
        super().__init__()
        self.args = args
        self.num_data = num_data
        self.model = taskModel(args)
        self.save_hyperparameters(args)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches
                self.total_steps = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))

    def training_step(self, batch, batch_idx):
        loss, logits = self.model(**batch)
        acc = self.comput_metrix(logits, batch['labels'])
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        loss, logits = self.model(**batch)
        acc = self.comput_metrix(logits, batch['labels'])
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def predict_step(self, batch, batch_idx):
        output = self.model(**batch)
        return output.logits

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)


class TaskModelCheckpoint:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./log/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)

        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=100, type=float)
        parser.add_argument('--save_weights_only', default=True, type=bool)

        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         every_n_train_steps=args.every_n_train_steps,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.dirpath,
                                         filename=args.filename)


def save_test(data, args, data_model):
    with open(args.output_save_path, 'w', encoding='utf-8') as f:
        idx = 0
        for i in range(len(data)):
            batch = data[i]
            for sample in batch:
                tmp_result = dict()
                label_id = np.argmax(sample.numpy())
                tmp_result['id'] = data_model.test_data.data[idx]['id']
                tmp_result['label'] = data_model.id2label[label_id]
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data+'\n')
                idx += 1
    print('save the result to '+args.output_save_path)


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser.add_argument('--pretrained_model_path', default='', type=str)
    total_parser.add_argument('--output_save_path',
                              default='./predict.json', type=str)
    total_parser.add_argument('--model_type',
                              default='huggingface-bert', type=str)

    # * Args for data preprocessing
    total_parser = TaskDataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = pl.Trainer.add_argparse_args(total_parser)
    total_parser = TaskModelCheckpoint.add_argparse_args(total_parser)

    # * Args for base model
    from fengshen.models.model_utils import add_module_args
    total_parser = add_module_args(total_parser)
    total_parser = LitModel.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    checkpoint_callback = TaskModelCheckpoint(args).callbacks
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[checkpoint_callback]
                                            )

    data_model = TaskDataModel(args)
    model = LitModel(args, len(data_model.train_dataloader()))

    trainer.fit(model, data_model)
    # result = trainer.predict(model, data_model)
    # save_test(result, args, data_model)


if __name__ == "__main__":
    main()
