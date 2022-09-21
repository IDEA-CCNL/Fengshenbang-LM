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
from pathlib import Path

from dataclasses import dataclass
import copy
import logging
import numpy as np
import torch.nn.functional as F
import os
import json
import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from fengshen.models.tagging_models.bert_for_tagging import BertLinear,BertCrf,BertSpan,BertBiaffine
from fengshen.data.tag_dataloader.tag_collator import CollatorForLinear, CollatorForCrf, CollatorForSpan
from fengshen.data.tag_dataloader.tag_datamodule import TaskDataModel
from fengshen.data.tag_dataloader.tag_datasets import get_labels
from fengshen.metric.metric import SeqEntityScore
from fengshen.metric.utils_ner import get_entities
from fengshen.models.model_utils import configure_optimizers
from fengshen.utils.universal_checkpoint import UniversalCheckpoint

from fengshen.models.tagging_models.bert_for_tagging import (
    BertLinear,
)
from transformers import (
    BertTokenizer, BertConfig, AutoTokenizer
)
from fengshen.metric.metric import BiaffineEntityScore, SeqEntityScore, SpanEntityScore
from fengshen.metric.utils_ner import get_entities, bert_extract_item


_model_dict={
    'bert-linear': BertLinear,
    'bert-crf': BertCrf,
    'bert-span': BertSpan,
    'bert-biaffine': BertBiaffine
}

_collator_dict={
    'linear': CollatorForLinear,
    'crf': CollatorForCrf,
    'span': CollatorForSpan
}

logger = logging.getLogger(__name__)

# class DataProcessor(object):
#     def __init__(self, data_dir) -> None:
#         super().__init__()
#         self.data_dir = data_dir

#     def get_examples(self, mode):
#         return self._create_examples(self._read_text(os.path.join(self.data_dir, mode + ".all.bmes")), mode)

#     def get_labels(self):
#         with open(os.path.join(self.data_dir, "labels.txt")) as f:
#             label_list = ["[PAD]", "[START]", "[END]"]
#             for line in f.readlines():
#                 label_list.append(line.strip())

#         label2id = {label: i for i, label in enumerate(label_list)}
#         return label2id

#     def _create_examples(self, lines, set_type):
#         examples = []
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             text_a = line['words']
#             labels = []
#             for x in line['labels']:
#                 if 'M-' in x:
#                     labels.append(x.replace('M-', 'I-'))
#                 else:
#                     labels.append(x)
#             subject = get_entities(labels, id2label=None, markup='bioes')
#             examples.append({'guid':guid, 'text_a':text_a, 'labels':labels, 'subject':subject})
#         return examples

#     @classmethod
#     def _read_text(self, input_file):
#         lines = []
#         with open(input_file, 'r') as f:
#             words = []
#             labels = []
#             for line in f:
#                 if line.startswith("-DOCSTART-") or line == "" or line == "\n":
#                     if words:
#                         lines.append({"words": words, "labels": labels})
#                         words = []
#                         labels = []
#                 else:
#                     splits = line.split()
#                     words.append(splits[0])
#                     if len(splits) > 1:
#                         labels.append(splits[-1].replace("\n", ""))
#                     else:
#                         # Examples could have no label for mode = "test"
#                         labels.append("O")
#             if words:
#                 lines.append({"words": words, "labels": labels})
#         return lines


# class TaskDataset(Dataset):
#     def __init__(self, processor, mode='train'):
#         super().__init__()
#         self.data = self.load_data(processor, mode)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]

#     def load_data(self, processor, mode):
#         examples = processor.get_examples(mode)
#         return examples


# @dataclass
# class TaskCollator:
#     args = None
#     tokenizer = None
#     label2id = None

#     def __call__(self, samples):
#         cls_token = "[CLS]"
#         sep_token = "[SEP]"
#         pad_token = 0
#         special_tokens_count = 2
#         segment_id = 0

#         features=[]

#         for (ex_index, example) in enumerate(samples):
#             tokens = copy.deepcopy(example['text_a'])

#             label_ids = [self.label2id[x] for x in example['labels']]

#             if len(tokens) > self.args.max_seq_length - special_tokens_count:
#                 tokens = tokens[: (self.args.max_seq_length - special_tokens_count)]
#                 label_ids = label_ids[: (self.args.max_seq_length - special_tokens_count)]

#             tokens += [sep_token]
#             label_ids += [self.label2id["O"]]
#             segment_ids = [segment_id] * len(tokens)

#             tokens = [cls_token] + tokens
#             label_ids = [self.label2id["O"]] + label_ids
#             segment_ids = [segment_id] + segment_ids

#             input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#             input_mask = [1] * len(input_ids)
#             input_len = len(label_ids)
#             padding_length = self.args.max_seq_length - len(input_ids)

#             input_ids += [pad_token] * padding_length
#             input_mask += [0] * padding_length
#             segment_ids += [segment_id] * padding_length
#             label_ids += [pad_token] * padding_length

#             assert len(input_ids) == self.args.max_seq_length
#             assert len(input_mask) == self.args.max_seq_length
#             assert len(segment_ids) == self.args.max_seq_length
#             assert len(label_ids) == self.args.max_seq_length

#             features.append({
#                     'input_ids':torch.tensor(input_ids),
#                     'attention_mask':torch.tensor(input_mask),
#                     'input_len':torch.tensor(input_len),
#                     'token_type_ids':torch.tensor(segment_ids),
#                     'labels':torch.tensor(label_ids),
#             })

#         return default_collate(features)


# class TaskDataModel(pl.LightningDataModule):
#     @staticmethod
#     def add_data_specific_args(parent_args):
#         parser = parent_args.add_argument_group('DataModel')
#         parser.add_argument('--data_dir', default='./data', type=str)
#         parser.add_argument('--num_workers', default=0, type=int)
#         parser.add_argument('--train_batchsize', default=16, type=int)
#         parser.add_argument('--valid_batchsize', default=16, type=int)
#         parser.add_argument('--max_seq_length', default=512, type=int)

#         parser.add_argument(
#             "--pretrained_model_path",
#             default=None,
#             type=str,
#             help="Path to pre-trained model or shortcut name selected in the list: ",
#         )
#         parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

#         return parent_args

#     def __init__(self, args):
#         super().__init__()
#         self.train_batchsize = args.train_batchsize
#         self.valid_batchsize = args.valid_batchsize

#         tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=args.do_lower_case)
#         processor = DataProcessor(args.data_dir)
#         args.label2id=processor.get_labels()

#         self.collator = TaskCollator()
#         self.collator.args=args
#         self.collator.tokenizer=tokenizer
#         self.collator.label2id=args.label2id

#         self.train_data=TaskDataset(processor=processor,mode="train")
#         self.valid_data=TaskDataset(processor=processor,mode="test")
#         self.test_data=TaskDataset(processor=processor,mode="test")

#         self.save_hyperparameters(args)
    
#     def train_dataloader(self):
#         return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batchsize, pin_memory=False,
#                           collate_fn=self.collator)

#     def val_dataloader(self):
#         return DataLoader(self.valid_data, shuffle=False, batch_size=self.valid_batchsize, pin_memory=False,
#                           collate_fn=self.collator)

#     def predict_dataloader(self):
#         return DataLoader(self.test_data, shuffle=False, batch_size=self.valid_batchsize, pin_memory=False,
#                           collate_fn=self.collator)


class LitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--pretrained_model_path',type=str)
        parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'])
        return parent_args

    def __init__(self, args, label2id):
        super().__init__()

        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        self.config=BertConfig.from_pretrained(args.pretrained_model_path)
        self.model = BertLinear.from_pretrained(args.pretrained_model_path,config=self.config, num_labels=len(self.id2label), loss_type=args.loss_type)
        self.entity_score=SeqEntityScore(self.id2label)

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
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        preds = preds.detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
        y_true = []
        y_pred = []
        for i, label in enumerate(labels):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == (torch.sum(batch['attention_mask'][i]).item()-1):
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(self.id2label[labels[i][j]])
                    temp_2.append(self.id2label[preds[i][j]])

        self.entity_score.update(y_true, y_pred)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outputs):
        # compute metric for all process
        score_dict, _ = self.entity_score.result()
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            print('score_dict:\n', score_dict)
        # reset the metric after once validation
        self.entity_score.reset()
        for k, v in score_dict.items():
            self.log('val_{}'.format(k), v)

    # def predict_step(self, batch, batch_idx):
    #     batch['labels'] = None
    #     outputs = self.model(**batch)

    #     logits = outputs.logits.detach().cpu().numpy()
    #     preds = np.argmax(logits, axis=2).tolist()

    #     for i, pred in enumerate(preds):
    #         text = self.hparams.tokenizer.convert_ids_to_tokens(batch['input_ids'][i])[:batch['input_len'][i]][1:-1]
    #         pred = pred[:batch['input_len'][i]][1:-1]
    #         label_entities = get_entities(pred, self.id2label)
    #         for label_list in label_entities:
    #             label_list.append("".join(text[label_list[1]:label_list[2]+1]))

    def configure_optimizers(self):
        return configure_optimizers(self)

# class TaskModelCheckpoint:
#     @staticmethod
#     def add_argparse_args(parent_args):
#         parser = parent_args.add_argument_group('BaseModel')

#         parser.add_argument('--monitor', default='train_loss', type=str)
#         parser.add_argument('--mode', default='min', type=str)
#         parser.add_argument('--dirpath', default='./log/', type=str)
#         parser.add_argument(
#             '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)

#         parser.add_argument('--save_top_k', default=3, type=float)
#         parser.add_argument('--every_n_train_steps', default=100, type=float)
#         parser.add_argument('--save_weights_only', default=True, type=bool)

#         return parent_args

#     def __init__(self, args):
#         self.callbacks = ModelCheckpoint(monitor=args.monitor,
#                                          save_top_k=args.save_top_k,
#                                          mode=args.mode,
#                                          every_n_train_steps=args.every_n_train_steps,
#                                          save_weights_only=args.save_weights_only,
#                                          dirpath=args.dirpath,
#                                          filename=args.filename)

def main():
    total_parser = argparse.ArgumentParser("TASK NAME")

    # * Args for data preprocessing
    total_parser = TaskDataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = pl.Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)

    # * Args for base model
    from fengshen.models.model_utils import add_module_args
    total_parser = add_module_args(total_parser)
    total_parser = LitModel.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    checkpoint_callback = UniversalCheckpoint(args).callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[checkpoint_callback, lr_monitor]
                                            )

    label2id,id2label=get_labels(args.decode_type)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    collator = _collator_dict[args.decode_type]()
    collator.args=args
    collator.tokenizer=tokenizer
    collator.label2id=label2id                                        
    data_model = TaskDataModel(args,collator,tokenizer)

    model = LitModel(args,label2id)
    print(label2id)
    trainer.fit(model, data_model)
    # trainer.predict(model,dataloaders=data_model.predict_dataloader())

if __name__ == "__main__":
    main()