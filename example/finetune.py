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


from transformers import BertModel, BertForMaskedLM, MegatronBertConfig, MegatronBertModel, MegatronBertForMaskedLM, BertConfig
from model.roformer.modeling_roformer import RoFormerModel, RoFormerForMaskedLM
from transformers import BertTokenizer, AutoTokenizer
from example.utils.arguments_parse import args
from example.utils.logger import logger
from model.roformer.configuration_roformer import RoFormerConfig
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import json
from utils.arguments_parse import args
import numpy as np
import os
import sys
sys.path.append('./')


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model_type = {'bert': BertModel,
              'roformer': RoFormerModel,
              'megatron': MegatronBertModel}

model_mlm_type = {'bert': BertForMaskedLM,
                  'roformer': RoFormerForMaskedLM,
                  'megatron': MegatronBertForMaskedLM}

model_config = {'bert': BertConfig,
                'roformer': RoFormerConfig,
                'megatron': MegatronBertConfig}

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)


def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result = []
        for line in tqdm(lines):
            data = json.loads(line)
            # 测试集中没有label标签，默认为0
            labels = int(data['label']) if 'label' in data.keys() else 0
            # 在该份数据集中只有测试集才有id字段
            text_id = int(data['id']) if 'id' in data.keys() else 0
            result.append(
                {'texta': data['sentence1'], 'textb': data['sentence2'], 'labels': labels, 'id': text_id})
        return result


def encoder(texta, textb):
    encode_dict = tokenizer.encode_plus(texta+'[SEP]'+textb,
                                        max_length=args.max_length,
                                        pad_to_max_length=True)
    encode_sent = encode_dict['input_ids']
    token_type_ids = encode_dict['token_type_ids']
    attention_mask = encode_dict['attention_mask']
    return encode_sent, token_type_ids, attention_mask


class taskDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        input_ids, input_seg, input_mask = encoder(
            item['texta'], item['textb'])

        one_data = {
            "input_ids": torch.tensor(input_ids).long(),
            "input_seg": torch.tensor(input_seg).long(),
            "input_mask": torch.tensor(input_mask).float(),
            "labels": torch.tensor(item['labels']).long(),
            "id": item['id'],
        }
        return one_data


def yield_data(dataset):
    dataset = taskDataset(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    return data_loader


class metrics_func(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        ones = torch.ones_like(logits)
        zero = torch.zeros_like(logits)
        y_pred = torch.where(logits < 0.5, zero, ones)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.size()[0]
        return acc


class taskModel(nn.Module):
    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.config = model_config[args.model_type].from_pretrained(
            pre_train_dir)
        self.bert_encoder = model_type[args.model_type].from_pretrained(
            pre_train_dir)
        self.cls_layer = torch.nn.Linear(
            in_features=self.config.hidden_size, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_func = torch.nn.BCELoss()

    def forward(self, input_ids, input_mask, input_seg, labels=None, is_training=False):
        bert_output = self.bert_encoder(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)  # (bsz, seq, dim)
        logits = self.cls_layer(bert_output[1])
        logits = self.sigmoid(logits)
        if is_training:
            loss = self.loss_func(logits.view(-1,), labels.float().view(-1,))
            return loss, logits
        else:
            return 0, logits


def eval_dev(model, eval_func, dev_loader,):
    with torch.no_grad():
        acc = 0
        count = 0
        loss = 0
        for item in dev_loader:
            count += 1
            input_ids, input_mask, input_seg = item["input_ids"], item["input_mask"], item["input_seg"]
            labels = item["labels"]
            loss_tmp, logits = model(
                input_ids=input_ids.to(device),
                input_mask=input_mask.to(device),
                input_seg=input_seg.to(device),
                labels=labels.to(device),
                is_training=True
            )
            acc_tmp = eval_func(logits, labels.to(device))
            acc += acc_tmp
            loss += loss_tmp
        acc /= count
        loss /= count
    return loss, acc


def train():
    model = taskModel(pre_train_dir=args.pretrained_model_path,
                      dropout_rate=0).to(device)
    model.train()
    eval_func = metrics_func().to(device)

    train_loader = yield_data(load_data(args.train_data_path))
    dev_loader = yield_data(load_data(args.dev_data_path))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]

    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters, lr=args.learning_rate)

    step = 0
    best = 0
    for epoch in range(args.epoch):

        for item in train_loader:
            step += 1
            input_ids, input_mask, input_seg = item["input_ids"], item["input_mask"], item["input_seg"]
            labels = item["labels"]
            optimizer.zero_grad()
            loss, logits = model(
                input_ids=input_ids.to(device),
                input_mask=input_mask.to(device),
                input_seg=input_seg.to(device),
                labels=labels.to(device),
                is_training=True
            )
            loss = loss.float().mean().type_as(loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.clip_norm)
            optimizer.step()
            if step % 100 == 0:
                train_acc_step = eval_func(logits, labels.to(device))
                train_loss_step = loss
                dev_loss_step, dev_acc_step = eval_dev(
                    model, eval_func, dev_loader)
                logger.info('epoch %d, step %d, train_loss %.4f, train_acc %.4f, dev_loss %.4f, dev_acc %.4f' % (
                    epoch, step, train_loss_step, train_acc_step, dev_loss_step, dev_acc_step))

        if best < dev_acc_step:
            best = dev_acc_step
            torch.save(model.state_dict(), f=args.checkpoints)
            logger.info(f'-----save the best model to {args.checkpoints}----')


def model_predict(input_text_list, model):
    logits = []
    with torch.no_grad():
        batch_size = args.batch_size
        start = 0
        end = batch_size
        while start < len(input_text_list):
            sentence = input_text_list[start:end]
            start = end
            end += batch_size
            input_ids, input_seg, input_mask = [], [], []
            for sent in sentence:
                input_ids0, input_seg0, input_mask0 = encoder(
                    sent['texta'], sent['textb'])
                input_ids.append(input_ids0)
                input_seg.append(input_seg0)
                input_mask.append(input_mask0)
            input_ids = torch.Tensor(input_ids).long()
            input_seg = torch.Tensor(input_seg).long()
            input_mask = torch.Tensor(input_mask).float()
            _, logit = model(
                input_ids=input_ids.to(device),
                input_mask=input_mask.to(device),
                input_seg=input_seg.to(device),
                is_training=False
            )
            logit = logit.view(-1,).to(torch.device('cpu')
                                       ).detach().numpy().tolist()
            logits.extend(logit)
    return logits


def predict():
    sentences = load_data(args.test_data_path)
    task_model = taskModel(
        pre_train_dir=args.pretrained_model_path, dropout_rate=0).to(device)
    task_model.load_state_dict(torch.load(args.checkpoints))
    task_model.eval()
    logits = model_predict(sentences, task_model)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for i in range(len(sentences)):
            tmp_result = dict()
            tmp_result['id'] = sentences[i]['id']
            tmp_result['label'] = '1' if logits[i] > 0.5 else '0'
            json_data = json.dumps(tmp_result, ensure_ascii=False)
            f.write(json_data+'\n')
    logger.info('save the result to '+args.output_path)


if __name__ == "__main__":
    train()
    predict()
