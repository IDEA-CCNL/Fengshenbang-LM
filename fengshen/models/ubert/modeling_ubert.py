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

from logging import basicConfig, setLogRecordFactory
import torch
from torch import nn
import json
from tqdm import tqdm
import os
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    file_utils
)
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import trainer, loggers
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertForPreTraining, BertForMaskedLM, BertModel
from transformers import BertConfig, BertForTokenClassification, BertPreTrainedModel
import transformers
import unicodedata
import re
import argparse


transformers.logging.set_verbosity_error()
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'


def search(pattern, sequence):
    n = len(pattern)
    res = []
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            res.append([i, i + n-1])
    return res


class UbertDataset(Dataset):
    def __init__(self, data, tokenizer, args, used_mask=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.num_labels = args.num_labels
        self.used_mask = used_mask
        self.data = data
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index], self.used_mask)

    def encode(self, item, used_mask=False):
        input_ids1 = []
        attention_mask1 = []
        token_type_ids1 = []
        span_labels1 = []
        span_labels_masks1 = []

        input_ids0 = []
        attention_mask0 = []
        token_type_ids0 = []
        span_labels0 = []
        span_labels_masks0 = []

        subtask_type = item['subtask_type']
        for choice in item['choices']:
            try:
                texta = item['task_type'] + '[SEP]'+subtask_type + '[SEP]' + choice['entity_type']
                textb = item['text']
                encode_dict = self.tokenizer.encode_plus(texta, textb,
                                                         max_length=self.max_length,
                                                         padding='max_length',
                                                         truncation='longest_first')

                encode_sent = encode_dict['input_ids']
                encode_token_type_ids = encode_dict['token_type_ids']
                encode_attention_mask = encode_dict['attention_mask']
                span_label = np.zeros((self.max_length, self.max_length))
                span_label_mask = np.zeros(
                    (self.max_length, self.max_length))-10000

                if item['task_type'] == '分类任务':
                    span_label_mask[0, 0] = 0
                    span_label[0, 0] = choice['label']

                else:
                    question_len = len(self.tokenizer.encode(texta))
                    span_label_mask[question_len:, question_len:] = np.zeros(
                        (self.max_length-question_len, self.max_length-question_len))
                    for entity in choice['entity_list']:
                        # if 'entity_name' in entity.keys() and entity['entity_name']=='':
                        #     continue
                        entity_idx_list = entity['entity_idx']
                        if entity_idx_list == []:
                            continue
                        for entity_idx in entity_idx_list:
                            if entity_idx == []:
                                continue
                            start_idx_text = item['text'][:entity_idx[0]]
                            start_idx_text_encode = self.tokenizer.encode(
                                start_idx_text, add_special_tokens=False)
                            start_idx = question_len + len(start_idx_text_encode)

                            end_idx_text = item['text'][:entity_idx[1]+1]
                            end_idx_text_encode = self.tokenizer.encode(
                                end_idx_text, add_special_tokens=False)
                            end_idx = question_len + len(end_idx_text_encode) - 1
                            if start_idx < self.max_length and end_idx < self.max_length:
                                span_label[start_idx, end_idx] = 1

                if np.sum(span_label) < 1:
                    input_ids0.append(encode_sent)
                    attention_mask0.append(encode_attention_mask)
                    token_type_ids0.append(encode_token_type_ids)
                    span_labels0.append(span_label)
                    span_labels_masks0.append(span_label_mask)
                else:
                    input_ids1.append(encode_sent)
                    attention_mask1.append(encode_attention_mask)
                    token_type_ids1.append(encode_token_type_ids)
                    span_labels1.append(span_label)
                    span_labels_masks1.append(span_label_mask)
            except:
                print(item)
                print(texta)
                print(textb)

        randomize = np.arange(len(input_ids0))
        np.random.shuffle(randomize)
        cur = 0
        count = len(input_ids1)
        while count < self.args.num_labels:
            if cur < len(randomize):
                input_ids1.append(input_ids0[randomize[cur]])
                attention_mask1.append(attention_mask0[randomize[cur]])
                token_type_ids1.append(token_type_ids0[randomize[cur]])
                span_labels1.append(span_labels0[randomize[cur]])
                span_labels_masks1.append(span_labels_masks0[randomize[cur]])
                cur += 1
            count += 1

        while len(input_ids1) < self.args.num_labels:
            input_ids1.append([0]*self.max_length)
            attention_mask1.append([0]*self.max_length)
            token_type_ids1.append([0]*self.max_length)
            span_labels1.append(np.zeros((self.max_length, self.max_length)))
            span_labels_masks1.append(
                np.zeros((self.max_length, self.max_length))-10000)

        input_ids = input_ids1[:self.args.num_labels]
        attention_mask = attention_mask1[:self.args.num_labels]
        token_type_ids = token_type_ids1[:self.args.num_labels]
        span_labels = span_labels1[:self.args.num_labels]
        span_labels_masks = span_labels_masks1[:self.args.num_labels]

        span_labels = np.array(span_labels)
        span_labels_masks = np.array(span_labels_masks)
        if np.sum(span_labels) < 1:
            span_labels[-1, -1, -1] = 1
            span_labels_masks[-1, -1, -1] = 10000

        sample = {
            "input_ids": torch.tensor(input_ids).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "attention_mask": torch.tensor(attention_mask).float(),
            "span_labels": torch.tensor(span_labels).float(),
            "span_labels_mask": torch.tensor(span_labels_masks).float()
        }

        return sample


class UbertDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--batchsize', default=8, type=int)
        parser.add_argument('--max_length', default=128, type=int)
        return parent_args

    def __init__(self, train_data, val_data, tokenizer, args):
        super().__init__()
        self.batchsize = args.batchsize

        self.train_data = UbertDataset(train_data, tokenizer, args, True)
        self.valid_data = UbertDataset(val_data, tokenizer, args, False)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batchsize, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, batch_size=self.batchsize, pin_memory=False)


class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.zeros(
            in_size + int(bias_x), out_size, in_size + int(bias_y)))
        torch.nn.init.normal_(self.U, mean=0, std=0.1)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


class multilabel_cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        y_pred = torch.mul((1.0 - torch.mul(y_true, 2.0)), y_pred)
        y_pred_neg = y_pred - torch.mul(y_true, 1e12)
        y_pred_pos = y_pred - torch.mul(1.0 - y_true, 1e12)
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
        pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
        loss = torch.mean(neg_loss + pos_loss)
        return loss


class UbertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.query_layer = torch.nn.Sequential(torch.nn.Linear(in_features=self.config.hidden_size,
                                                               out_features=self.config.biaffine_size),
                                               torch.nn.GELU())
        self.key_layer = torch.nn.Sequential(torch.nn.Linear(in_features=self.config.hidden_size, out_features=self.config.biaffine_size),
                                             torch.nn.GELU())
        self.biaffine_query_key_cls = biaffine(self.config.biaffine_size, 1)
        self.loss_softmax = multilabel_cross_entropy()
        self.loss_sigmoid = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                span_labels=None,
                span_labels_mask=None):

        batch_size, num_label, seq_len = input_ids.shape

        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)

        batch_size, seq_len = input_ids.shape
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)  # (bsz, seq, dim)

        hidden_states = outputs[0]
        batch_size, seq_len, hidden_size = hidden_states.shape

        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)

        span_logits = self.biaffine_query_key_cls(
            query, key).reshape(-1, num_label, seq_len, seq_len)

        span_logits = span_logits + span_labels_mask

        if span_labels == None:
            return 0, span_logits
        else:
            # soft_loss = self.loss_softmax(span_logits.permute(0,2,3,1), span_labels.permute(0,2,3,1))
            sig_loss = self.loss_sigmoid(span_logits, span_labels)
            all_loss = 100000*(sig_loss)
            return all_loss, span_logits


class UbertLitModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        parser.add_argument('--num_labels', default=10, type=int)

        return parent_args

    def __init__(self, args, num_data=1):
        super().__init__()
        self.args = args
        self.num_data = num_data
        self.model = UbertModel.from_pretrained(
            self.args.pretrained_model_path)
        self.count = 0

    def setup(self, stage) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data /
                                  (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def training_step(self, batch, batch_idx):
        loss, span_logits = self.model(**batch)
        span_acc, recall, precise = self.comput_metrix_span(
            span_logits, batch['span_labels'])
        self.log('train_loss', loss)
        self.log('train_span_acc', span_acc)
        self.log('train_span_recall', recall)
        self.log('train_span_precise', precise)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, span_logits = self.model(**batch)
        span_acc, recall, precise = self.comput_metrix_span(
            span_logits, batch['span_labels'])

        self.log('val_loss', loss)
        self.log('val_span_acc', span_acc)
        self.log('val_span_recall', recall)
        self.log('val_span_precise', precise)

    def predict_step(self, batch, batch_idx):
        loss, span_logits = self.model(**batch)
        span_acc = self.comput_metrix_span(span_logits, batch['span_labels'])
        return span_acc.item()

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

    def comput_metrix_span(self, logits, labels):
        ones = torch.ones_like(logits)
        zero = torch.zeros_like(logits)
        logits = torch.where(logits < 0, zero, ones)
        y_pred = logits.view(size=(-1,))
        y_true = labels.view(size=(-1,))
        corr = torch.eq(y_pred, y_true).float()
        corr = torch.multiply(y_true, corr)
        recall = torch.sum(corr.float())/(torch.sum(y_true.float())+1e-5)
        precise = torch.sum(corr.float())/(torch.sum(y_pred.float())+1e-5)
        f1 = 2*recall*precise/(recall+precise+1e-5)
        return f1, recall, precise


class TaskModelCheckpoint:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--checkpoint_path', default='./checkpoint/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)

        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_epochs', default=1, type=float)
        parser.add_argument('--every_n_train_steps', default=100, type=float)

        parser.add_argument('--save_weights_only', default=True, type=bool)
        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         save_last=True,
                                         every_n_train_steps=args.every_n_train_steps,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.checkpoint_path,
                                         filename=args.filename)


class OffsetMapping:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join(
                    [c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([offset])
                offset += 1
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


class extractModel:
    def get_actual_id(self, text, query_text, tokenizer, args):
        text_encode = tokenizer.encode(text)
        one_input_encode = tokenizer.encode(query_text)
        text_start_id = search(text_encode[1:-1], one_input_encode)[0][0]
        text_end_id = text_start_id+len(text_encode)-1
        if text_end_id > args.max_length:
            text_end_id = args.max_length

        text_token = tokenizer.tokenize(text)
        text_mapping = OffsetMapping().rematch(text, text_token)

        return text_start_id, text_end_id, text_mapping, one_input_encode

    def extract_index(self, span_logits, sample_length, split_value=0.5):
        result = []
        for i in range(sample_length):
            for j in range(i, sample_length):
                if span_logits[i, j] > split_value:
                    result.append((i, j, span_logits[i, j]))
        return result

    def extract_entity(self, text, entity_idx, text_start_id, text_mapping):
        start_split = text_mapping[entity_idx[0]-text_start_id] if entity_idx[0] - \
            text_start_id < len(text_mapping) and entity_idx[0]-text_start_id >= 0 else []
        end_split = text_mapping[entity_idx[1]-text_start_id] if entity_idx[1] - \
            text_start_id < len(text_mapping) and entity_idx[1]-text_start_id >= 0 else []
        entity = ''
        if start_split != [] and end_split != []:
            entity = text[start_split[0]:end_split[-1]+1]
        return entity


    def extract(self, batch_data, model, tokenizer, args):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        span_labels_masks = []

        for item in batch_data:
            input_ids0 = []
            attention_mask0 = []
            token_type_ids0 = []
            span_labels_masks0 = []
            for choice in item['choices']:
                texta = item['task_type'] + '[SEP]'+item['subtask_type'] + '[SEP]' + choice['entity_type']
                textb = item['text']
                encode_dict = tokenizer.encode_plus(texta, textb,
                                                    max_length=args.max_length,
                                                    padding='max_length',
                                                    truncation='longest_first')

                encode_sent = encode_dict['input_ids']
                encode_token_type_ids = encode_dict['token_type_ids']
                encode_attention_mask = encode_dict['attention_mask']
                span_label_mask = np.zeros(
                    (args.max_length, args.max_length))-10000

                if item['task_type'] == '分类任务':
                    span_label_mask[0, 0] = 0
                else:
                    question_len = len(tokenizer.encode(texta))
                    span_label_mask[question_len:, question_len:] = np.zeros(
                        (args.max_length-question_len, args.max_length-question_len))
                input_ids0.append(encode_sent)
                attention_mask0.append(encode_attention_mask)
                token_type_ids0.append(encode_token_type_ids)
                span_labels_masks0.append(span_label_mask)

            input_ids.append(input_ids0)
            attention_mask.append(attention_mask0)
            token_type_ids.append(token_type_ids0)
            span_labels_masks.append(span_labels_masks0)
        
        input_ids = torch.tensor(input_ids).to(model.device)
        attention_mask = torch.tensor(attention_mask).to(model.device)
        token_type_ids = torch.tensor(token_type_ids).to(model.device)
        span_labels_mask = torch.tensor(span_labels_masks).to(model.device)

        _, span_logits = model.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     span_labels=None,
                                     span_labels_mask=span_labels_mask)

        span_logits = torch.nn.functional.sigmoid(span_logits)
        span_logits = span_logits.cpu().detach().numpy()

        for i, item in enumerate(batch_data):
            if item['task_type'] == '分类任务':
                cls_idx = 0
                max_c = np.argmax(span_logits[i, :, cls_idx, cls_idx])
                batch_data[i]['choices'][max_c]['label'] = 1
                batch_data[i]['choices'][max_c]['score'] = span_logits[i, max_c, cls_idx, cls_idx]
            else:
                if item['subtask_type'] == '抽取式阅读理解':
                    for c in range(len(item['choices'])):
                        texta = item['subtask_type'] + \
                            '[SEP]' + choice['entity_type']
                        textb = item['text']
                        text_start_id, text_end_id, offset_mapping, input_ids = self.get_actual_id(
                            item['text'], texta+'[SEP]'+textb, tokenizer, args)
                        logits = span_logits[i, c, :, :]
                        max_index = np.unravel_index(
                            np.argmax(logits, axis=None), logits.shape)
                        entity_list = []
                        if logits[max_index] > args.threshold:

                            entity = self.extract_entity(
                                item['text'], (max_index[0], max_index[1]), text_start_id, offset_mapping)
                            entity = {
                                'entity_name': entity,
                                'score': logits[max_index]
                            }
                            if entity not in entity_list:
                                entity_list.append(entity)
                        batch_data[i]['choices'][c]['entity_list'] = entity_list
                else:
                    for c in range(len(item['choices'])):
                        texta = item['task_type'] + '[SEP]'+ item['subtask_type'] + \
                            '[SEP]' + item['choices'][c]['entity_type']

                        textb = item['text']
                        text_start_id, text_end_id, offset_mapping, input_ids = self.get_actual_id(
                            item['text'], texta+'[SEP]'+textb, tokenizer, args)
                        logits = span_logits[i, c, :, :]
                        sample_length = len(input_ids)
                        entity_idx_type_list = self.extract_index(
                            logits, sample_length, split_value=args.threshold)
                        entity_list = []

                        for entity_idx in entity_idx_type_list:
                            entity = self.extract_entity(
                                item['text'], (entity_idx[0], entity_idx[1]), text_start_id, offset_mapping)
                            entity = {
                                'entity_name': entity,
                                'score': entity_idx[2]
                            }
                            if entity not in entity_list:
                                entity_list.append(entity)
                        batch_data[i]['choices'][c]['entity_list'] = entity_list
        return batch_data


class UbertPiplines:
    @staticmethod
    def piplines_args(parent_args):
        total_parser = parent_args.add_argument_group("piplines args")
        total_parser.add_argument(
            '--pretrained_model_path', default='IDEA-CCNL/Erlangshen-Ubert-110M', type=str)
        total_parser.add_argument('--output_save_path',
                                  default='./predict.json', type=str)

        total_parser.add_argument('--load_checkpoints_path',
                                  default='', type=str)

        total_parser.add_argument('--max_extract_entity_number',
                                  default=1, type=float)

        total_parser.add_argument('--train', action='store_true')

        total_parser.add_argument('--threshold',
                                  default=0.5, type=float)

        total_parser = UbertDataModel.add_data_specific_args(total_parser)
        total_parser = TaskModelCheckpoint.add_argparse_args(total_parser)
        total_parser = UbertLitModel.add_model_specific_args(total_parser)
        total_parser = pl.Trainer.add_argparse_args(parent_args)
        
        return parent_args

    def __init__(self, args):
        
        if args.load_checkpoints_path != '':
            self.model = UbertLitModel.load_from_checkpoint(
                args.load_checkpoints_path, args=args)
        else:
            self.model = UbertLitModel(args)
        

        self.args = args
        self.checkpoint_callback = TaskModelCheckpoint(args).callbacks
        self.logger = loggers.TensorBoardLogger(save_dir=args.default_root_dir)
        self.trainer = pl.Trainer.from_argparse_args(args,
                                                     logger=self.logger,
                                                     callbacks=[self.checkpoint_callback])

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path,
                                                       additional_special_tokens=['[unused'+str(i+1)+']' for i in range(99)])

        self.em = extractModel()

    def fit(self, train_data, dev_data):
        data_model = UbertDataModel(
            train_data, dev_data, self.tokenizer, self.args)
        self.model.num_data = len(train_data)
        self.trainer.fit(self.model, data_model)

    def predict(self, test_data, cuda=True):
        result = []
        start = 0
        if cuda:
            self.model=self.model.cuda()
        self.model.eval()
        while start < len(test_data):
            batch_data = test_data[start:start+self.args.batchsize]
            start += self.args.batchsize

            batch_result = self.em.extract(
                batch_data, self.model, self.tokenizer, self.args)
            result.extend(batch_result)
        return result












