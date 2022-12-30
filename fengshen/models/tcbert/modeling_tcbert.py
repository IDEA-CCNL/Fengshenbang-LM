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

from logging import basicConfig
import torch
from torch import nn
import json
from tqdm import tqdm
import os
import numpy as np
from transformers import BertTokenizer
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import trainer, loggers
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertForMaskedLM
from transformers import AutoConfig
from transformers.pipelines.base import Pipeline
from transformers import MegatronBertForMaskedLM
import argparse
import copy
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
import warnings
from transformers import TextClassificationPipeline as HuggingfacePipe


class TCBertDataset(Dataset):
    def __init__(self, data, tokenizer, args, prompt, label_classes):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.num_labels = args.num_labels
        self.data = data
        self.args = args
        self.label_classes = label_classes
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index])

    
    def encode(self, item, labeled=True):

        if labeled:
            ori_texta = self.prompt.format(item['label']) + item['content']
            mask_texta = self.prompt.format("[MASK]" * len(item['label'])) + item['content']
            # print('texta', texta)
            labels = self.label_classes[item['label']]

            ori_encode_dict = self.tokenizer.encode_plus(ori_texta,
                                            max_length=self.max_length,
                                            padding="longest",
                                            truncation=True
                                            )
            
            mask_encode_dict = self.tokenizer.encode_plus(mask_texta,
                                            max_length=self.max_length,
                                            padding="longest",
                                            truncation=True
                                            )

            ori_input_ids = torch.tensor(ori_encode_dict['input_ids']).long()
            token_type_ids = torch.tensor(ori_encode_dict['token_type_ids']).long()
            attention_mask = torch.tensor(ori_encode_dict['attention_mask']).float()

            mask_input_ids = torch.tensor(mask_encode_dict['input_ids']).long()
            mlmlabels = torch.where(mask_input_ids == self.tokenizer.mask_token_id, ori_input_ids, -100)

            labels = torch.tensor(labels).long()
            mlmlabels = torch.tensor(mlmlabels).long()

            encoded = {
                "sentence": item["content"],
                "input_ids": mask_input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "mlmlabels": mlmlabels,
            }

        else:

            texta = self.prompt.format("[MASK]" * self.args.fixed_lablen)  + item['content']

            encode_dict = self.tokenizer.encode_plus(texta,
                                                max_length=self.max_length,
                                                padding="longest",
                                                truncation=True
                                                )
            
            input_ids = encode_dict['input_ids']
            token_type_ids = encode_dict['token_type_ids']
            attention_mask = encode_dict['attention_mask']

            encoded = {
                "sentence": item["content"],
                "input_ids": torch.tensor(input_ids).long(),
                "token_type_ids": torch.tensor(token_type_ids).long(),
                "attention_mask": torch.tensor(attention_mask).float(),
            }
        return encoded



class TCBertDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--batchsize', default=16, type=int)
        parser.add_argument('--max_length', default=512, type=int)
        parser.add_argument('--fixed_lablen', default=2, type=int)
        return parent_args

    def __init__(self, train_data, val_data, tokenizer, args, prompt, prompt_label):
        super().__init__()
        self.batchsize = args.batchsize
        self.label_classes = self.get_label_classes(prompt_label)
        args.num_labels = len(self.label_classes)

        self.train_data = TCBertDataset(train_data, tokenizer, args, prompt, self.label_classes)
        self.valid_data = TCBertDataset(val_data, tokenizer, args, prompt, self.label_classes)

    def get_label_classes(self, prompt_label):
        label_classes = {}
        i = 0 
        for key in prompt_label.keys():
            label_classes[key] = i
            i+=1
        print("label_classes:",label_classes)
        return label_classes

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=self.collate_fn, batch_size=self.batchsize, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=self.collate_fn, batch_size=self.batchsize, pin_memory=False)

    def collate_fn(self, batch):
        '''
        Aggregate a batch data.
        batch = [ins1_dict, ins2_dict, ..., insN_dict]
        batch_data = {'sentence':[ins1_sentence, ins2_sentence...], 'input_ids':[ins1_input_ids, ins2_input_ids...], ...}
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        token_type_ids = batch_data["token_type_ids"]
        labels = None
        if 'labels' in batch_data:
            labels = torch.LongTensor(batch_data['labels'])

        mlmlabels = None
        if 'mlmlabels' in batch_data:
            mlmlabels = nn.utils.rnn.pad_sequence(batch_data['mlmlabels'],
                                                batch_first=True,
                                                padding_value=-100)
        
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                                batch_first=True,
                                                padding_value=0)
            
        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                                    batch_first=True,
                                                    padding_value=0)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                                    batch_first=True,
                                                    padding_value=0)

        batch_data = {
            "sentence":batch_data["sentence"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "mlmlabels":mlmlabels
        }

        return batch_data
        


class TCBertModel(nn.Module):
    def __init__(self, pre_train_dir, nlabels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pre_train_dir)
        print("pre_train_dir", pre_train_dir)
        # if self.config.model_type == 'megatron-bert':
        if "1.3B" in pre_train_dir:
            self.bert = MegatronBertForMaskedLM.from_pretrained(pre_train_dir)
        else:
            self.bert = BertForMaskedLM.from_pretrained(pre_train_dir)

        self.dropout = nn.Dropout(0.1)
        self.nlabels = nlabels
        self.linear_classifier = nn.Linear(self.config.hidden_size, self.nlabels)

    def forward(self, input_ids, attention_mask, token_type_ids, mlmlabels=None):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=mlmlabels,
                            output_hidden_states=True)  # (bsz, seq, dim)

        mlm_logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        cls_logits = hidden_states[:,0]
        cls_logits = self.dropout(cls_logits)

        logits = self.linear_classifier(cls_logits)

        return outputs.loss, logits, mlm_logits


class TCBertLitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        parser.add_argument('--num_labels', default=2, type=int)

        return parent_args

    def __init__(self, args, model_path, nlabels):
        super().__init__()
        self.args = args
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = TCBertModel(model_path, nlabels)

    def setup(self, stage) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data /
                                  (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)


    def train_inputs(self, batch):
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'mlmlabels': batch['mlmlabels']
        }
        return inputs 

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        batch = self.train_inputs(batch)
        mlm_loss, logits, _= self.model(**batch)
        if labels is not None:
            cls_loss = self.loss_fn(logits, labels.view(-1))

        loss = cls_loss + mlm_loss

        ntotal = logits.size(0)
        ncorrect = (logits.argmax(dim=-1) == labels).long().sum()
        acc = ncorrect / ntotal

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        batch = self.train_inputs(batch)
        mlm_loss, logits, _ = self.model(**batch)
        predict = logits.argmax(dim=-1).cpu().tolist()

        if labels is not None:
            cls_loss = self.loss_fn(logits, labels.view(-1))

        loss = cls_loss + mlm_loss
        ntotal = logits.size(0)
        
        ncorrect = int((logits.argmax(dim=-1) == labels).long().sum())
        acc = ncorrect / ntotal

        self.log('valid_loss', loss, on_step=True, prog_bar=True)
        self.log("valid_acc", acc, on_step=True, prog_bar=True)

        return int(ncorrect), int(ntotal)

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(paras, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.args.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]



class TCBertPredict:
    def __init__(self, model, tokenizer, args, prompt, prompt_label):
        self.tokenizer = tokenizer
        self.args = args
        self.data_model = TCBertDataModel(
            [], [], tokenizer, args, prompt, prompt_label)
        self.model = model
    
    def predict_inputs(self, batch):
        #  Filter reduntant information(for example: 'sentence') that will be passed to model.forward()
        inputs = {
            'input_ids': batch['input_ids'].cuda(),
            'attention_mask': batch['attention_mask'].cuda(),
            'token_type_ids': batch['token_type_ids'].cuda(),
        }
        return inputs 

    def predict(self, batch_data):
        batch = [self.data_model.train_data.encode(
            sample, labeled=False) for sample in batch_data]
        batch = self.data_model.collate_fn(batch)
        batch = self.predict_inputs(batch)
        _, logits, _ = self.model.model(**batch)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicts = torch.argmax(probs, dim=-1).detach().cpu().numpy()

        return predicts

