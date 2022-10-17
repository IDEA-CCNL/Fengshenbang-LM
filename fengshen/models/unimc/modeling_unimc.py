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
from transformers import BertForMaskedLM, AlbertTokenizer
from transformers import AutoConfig
from transformers.pipelines.base import Pipeline
from transformers import MegatronBertForMaskedLM
from fengshen.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForMaskedLM
from fengshen.models.albert.modeling_albert import AlbertForMaskedLM
import argparse
import copy
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
import warnings
from transformers import TextClassificationPipeline as HuggingfacePipe


class UniMCDataset(Dataset):
    def __init__(self, data, yes_token, no_token, tokenizer, args, used_mask=True):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.num_labels = args.num_labels
        self.used_mask = used_mask
        self.data = data
        self.args = args
        self.yes_token = yes_token
        self.no_token = no_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index], self.used_mask)

    def get_token_type(self, sep_idx, max_length):
        token_type_ids = np.zeros(shape=(max_length,))
        for i in range(len(sep_idx)-1):
            if i % 2 == 0:
                ty = np.ones(shape=(sep_idx[i+1]-sep_idx[i],))
            else:
                ty = np.zeros(shape=(sep_idx[i+1]-sep_idx[i],))
            token_type_ids[sep_idx[i]:sep_idx[i+1]] = ty

        return token_type_ids

    def get_position_ids(self, label_idx, max_length, question_len):
        question_position_ids = np.arange(question_len)
        label_position_ids = np.arange(question_len, label_idx[-1])
        for i in range(len(label_idx)-1):
            label_position_ids[label_idx[i]-question_len:label_idx[i+1]-question_len] = np.arange(
                question_len, question_len+label_idx[i+1]-label_idx[i])
        max_len_label = max(label_position_ids)
        text_position_ids = np.arange(
            max_len_label+1, max_length+max_len_label+1-label_idx[-1])
        position_ids = list(question_position_ids) + \
            list(label_position_ids)+list(text_position_ids)
        if max_length <= 512:
            return position_ids[:max_length]
        else:
            for i in range(512, max_length):
                if position_ids[i] > 511:
                    position_ids[i] = 511
            return position_ids[:max_length]

    def get_att_mask(self, attention_mask, label_idx, question_len):
        max_length = len(attention_mask)
        attention_mask = np.array(attention_mask)
        attention_mask = np.tile(attention_mask[None, :], (max_length, 1))

        zeros = np.zeros(
            shape=(label_idx[-1]-question_len, label_idx[-1]-question_len))
        attention_mask[question_len:label_idx[-1],
                       question_len:label_idx[-1]] = zeros

        for i in range(len(label_idx)-1):
            label_token_length = label_idx[i+1]-label_idx[i]
            if label_token_length <= 0:
                print('label_idx', label_idx)
                print('question_len', question_len)
                continue
            ones = np.ones(shape=(label_token_length, label_token_length))
            attention_mask[label_idx[i]:label_idx[i+1],
                           label_idx[i]:label_idx[i+1]] = ones

        return attention_mask

    def random_masking(self, token_ids, maks_rate, mask_start_idx, max_length, mask_id, tokenizer):
        rands = np.random.random(len(token_ids))
        source, target = [], []
        for i, (r, t) in enumerate(zip(rands, token_ids)):
            if i < mask_start_idx:
                source.append(t)
                target.append(-100)
                continue
            if r < maks_rate * 0.8:
                source.append(mask_id)
                target.append(t)
            elif r < maks_rate * 0.9:
                source.append(t)
                target.append(t)
            elif r < maks_rate:
                source.append(np.random.choice(tokenizer.vocab_size - 1) + 1)
                target.append(t)
            else:
                source.append(t)
                target.append(-100)
        while len(source) < max_length:
            source.append(0)
            target.append(-100)
        return source[:max_length], target[:max_length]

    def encode(self, item, used_mask=False):

        while len(self.tokenizer.encode('[MASK]'.join(item['choice']))) > self.max_length-32:
            item['choice'] = [c[:int(len(c)/2)] for c in item['choice']]

        if 'textb' in item.keys() and item['textb'] != '':
            if 'question' in item.keys() and item['question'] != '':
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                    item['question'] + '[SEP]' + \
                        item['texta']+'[SEP]'+item['textb']
            else:
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                        item['texta']+'[SEP]'+item['textb']

        else:
            if 'question' in item.keys() and item['question'] != '':
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                    item['question'] + '[SEP]' + item['texta']
            else:
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + \
                    '[SEP]' + item['texta']

        encode_dict = self.tokenizer.encode_plus(texta,
                                                 max_length=self.max_length,
                                                 padding='max_length',
                                                 truncation='longest_first')

        encode_sent = encode_dict['input_ids']
        token_type_ids = encode_dict['token_type_ids']
        attention_mask = encode_dict['attention_mask']
        sample_max_length = sum(encode_dict['attention_mask'])

        if 'label' not in item.keys():
            item['label'] = 0
            item['answer'] = ''

        question_len = 1
        label_idx = [question_len]
        for choice in item['choice']:
            cur_mask_idx = label_idx[-1] + \
                len(self.tokenizer.encode(choice, add_special_tokens=False))+1
            label_idx.append(cur_mask_idx)

        token_type_ids = [0]*question_len+[1] * \
            (label_idx[-1]-label_idx[0]+1)+[0]*self.max_length
        token_type_ids = token_type_ids[:self.max_length]

        attention_mask = self.get_att_mask(
            attention_mask, label_idx, question_len)

        position_ids = self.get_position_ids(
            label_idx, self.max_length, question_len)

        clslabels_mask = np.zeros(shape=(len(encode_sent),))
        clslabels_mask[label_idx[:-1]] = 10000
        clslabels_mask = clslabels_mask-10000

        mlmlabels_mask = np.zeros(shape=(len(encode_sent),))
        mlmlabels_mask[label_idx[0]] = 1

        # used_mask=False
        if used_mask:
            mask_rate = 0.1*np.random.choice(4, p=[0.3, 0.3, 0.25, 0.15])
            source, target = self.random_masking(token_ids=encode_sent, maks_rate=mask_rate,
                                                 mask_start_idx=label_idx[-1], max_length=self.max_length,
                                                 mask_id=self.tokenizer.mask_token_id, tokenizer=self.tokenizer)
        else:
            source, target = encode_sent[:], encode_sent[:]

        source = np.array(source)
        target = np.array(target)
        source[label_idx[:-1]] = self.tokenizer.mask_token_id
        target[label_idx[:-1]] = self.no_token
        target[label_idx[item['label']]] = self.yes_token

        input_ids = source[:sample_max_length]
        token_type_ids = token_type_ids[:sample_max_length]
        attention_mask = attention_mask[:sample_max_length, :sample_max_length]
        position_ids = position_ids[:sample_max_length]
        mlmlabels = target[:sample_max_length]
        clslabels = label_idx[item['label']]
        clslabels_mask = clslabels_mask[:sample_max_length]
        mlmlabels_mask = mlmlabels_mask[:sample_max_length]

        return {
            "input_ids": torch.tensor(input_ids).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "attention_mask": torch.tensor(attention_mask).float(),
            "position_ids": torch.tensor(position_ids).long(),
            "mlmlabels": torch.tensor(mlmlabels).long(),
            "clslabels": torch.tensor(clslabels).long(),
            "clslabels_mask": torch.tensor(clslabels_mask).float(),
            "mlmlabels_mask": torch.tensor(mlmlabels_mask).float(),
        }


class UniMCDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--batchsize', default=16, type=int)
        parser.add_argument('--max_length', default=512, type=int)
        return parent_args

    def __init__(self, train_data, val_data, yes_token, no_token, tokenizer, args):
        super().__init__()
        self.batchsize = args.batchsize

        self.train_data = UniMCDataset(
            train_data, yes_token, no_token, tokenizer, args, True)
        self.valid_data = UniMCDataset(
            val_data, yes_token, no_token, tokenizer, args, False)

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

        batch_data['input_ids'] = nn.utils.rnn.pad_sequence(batch_data['input_ids'],
                                                            batch_first=True,
                                                            padding_value=0)
        batch_data['clslabels_mask'] = nn.utils.rnn.pad_sequence(batch_data['clslabels_mask'],
                                                                 batch_first=True,
                                                                 padding_value=-10000)

        batch_size, batch_max_length = batch_data['input_ids'].shape
        for k, v in batch_data.items():
            if k == 'input_ids' or k == 'clslabels_mask':
                continue
            if k == 'clslabels':
                batch_data[k] = torch.tensor(v).long()
                continue
            if k != 'attention_mask':
                batch_data[k] = nn.utils.rnn.pad_sequence(v,
                                                          batch_first=True,
                                                          padding_value=0)
            else:
                attention_mask = torch.zeros(
                    (batch_size, batch_max_length, batch_max_length))
                for i, att in enumerate(v):
                    sample_length, _ = att.shape
                    attention_mask[i, :sample_length, :sample_length] = att
                batch_data[k] = attention_mask
        return batch_data


class UniMCModel(nn.Module):
    def __init__(self, pre_train_dir, yes_token):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pre_train_dir)
        if self.config.model_type == 'megatron-bert':
            self.bert = MegatronBertForMaskedLM.from_pretrained(pre_train_dir)
        elif self.config.model_type == 'deberta-v2':
            self.bert = DebertaV2ForMaskedLM.from_pretrained(pre_train_dir)
        elif self.config.model_type == 'albert':
            self.bert = AlbertForMaskedLM.from_pretrained(pre_train_dir)
        else:
            self.bert = BertForMaskedLM.from_pretrained(pre_train_dir)

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.yes_token = yes_token

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids=None, mlmlabels=None, clslabels=None, clslabels_mask=None, mlmlabels_mask=None):

        batch_size, seq_len = input_ids.shape
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            labels=mlmlabels)  # (bsz, seq, dim)
        mask_loss = outputs.loss
        mlm_logits = outputs.logits
        cls_logits = mlm_logits[:, :,
                                self.yes_token].view(-1, seq_len)+clslabels_mask

        if mlmlabels == None:
            return 0, mlm_logits, cls_logits
        else:
            cls_loss = self.loss_func(cls_logits, clslabels)
            all_loss = mask_loss+cls_loss
            return all_loss, mlm_logits, cls_logits


class UniMCLitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        parser.add_argument('--num_labels', default=2, type=int)

        return parent_args

    def __init__(self, args, yes_token, model_path, num_data=100):
        super().__init__()
        self.args = args
        self.num_data = num_data
        self.model = UniMCModel(model_path, yes_token)

    def setup(self, stage) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data /
                                  (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def training_step(self, batch, batch_idx):
        loss, logits, cls_logits = self.model(**batch)
        cls_acc = self.comput_metrix(
            cls_logits, batch['clslabels'], batch['mlmlabels_mask'])
        self.log('train_loss', loss)
        self.log('train_acc', cls_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, cls_logits = self.model(**batch)
        cls_acc = self.comput_metrix(
            cls_logits, batch['clslabels'], batch['mlmlabels_mask'])
        self.log('val_loss', loss)
        self.log('val_acc', cls_acc)

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

    def comput_metrix(self, logits, labels, mlmlabels_mask):
        logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = torch.argmax(logits, dim=-1)
        y_pred = logits.view(size=(-1,))
        y_true = labels.view(size=(-1,))
        corr = torch.eq(y_pred, y_true).float()
        return torch.sum(corr.float())/labels.size(0)


class UniMCPredict:
    def __init__(self, yes_token, no_token, model, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.data_model = UniMCDataModel(
            [], [], yes_token, no_token, tokenizer, args)
        self.model = model

    def predict(self, batch_data):
        batch = [self.data_model.train_data.encode(
            sample) for sample in batch_data]
        batch = self.data_model.collate_fn(batch)
        batch = {k: v.cuda() for k, v in batch.items()}
        _, _, logits = self.model.model(**batch)
        soft_logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = torch.argmax(soft_logits, dim=-1).detach().cpu().numpy()

        soft_logits = soft_logits.detach().cpu().numpy()
        clslabels_mask = batch['clslabels_mask'].detach(
        ).cpu().numpy().tolist()
        clslabels = batch['clslabels'].detach().cpu().numpy().tolist()
        for i, v in enumerate(batch_data):
            label_idx = [idx for idx, v in enumerate(
                clslabels_mask[i]) if v == 0.]
            label = label_idx.index(logits[i])
            answer = batch_data[i]['choice'][label]
            score = {}
            for c in range(len(batch_data[i]['choice'])):
                score[batch_data[i]['choice'][c]] = float(
                    soft_logits[i][label_idx[c]])

            batch_data[i]['label_ori'] = copy.deepcopy(batch_data[i]['label'])
            batch_data[i]['label'] = label
            batch_data[i]['answer'] = answer
            batch_data[i]['score'] = score

        return batch_data


class UniMCPiplines(Pipeline):
    @staticmethod
    def piplines_args(parent_args):
        total_parser = parent_args.add_argument_group("piplines args")
        total_parser.add_argument(
            '--pretrained_model_path', default='', type=str)
        total_parser.add_argument('--load_checkpoints_path',
                                  default='', type=str)
        total_parser.add_argument('--train', action='store_true')
        total_parser.add_argument('--language',
                                  default='chinese', type=str)

        total_parser = UniMCDataModel.add_data_specific_args(total_parser)
        total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
        total_parser = UniMCLitModel.add_model_specific_args(total_parser)
        total_parser = pl.Trainer.add_argparse_args(parent_args)
        return parent_args

    def __init__(self, args, model_path):
        self.args = args
        self.checkpoint_callback = UniversalCheckpoint(args).callbacks
        self.logger = loggers.TensorBoardLogger(save_dir=args.default_root_dir)
        self.trainer = pl.Trainer.from_argparse_args(args,
                                                     logger=self.logger,
                                                     callbacks=[self.checkpoint_callback])
        self.config = AutoConfig.from_pretrained(model_path)
        if self.config.model_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(
                model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                model_path)

        if args.language == 'chinese':
            self.yes_token = self.tokenizer.encode('是')[1]
            self.no_token = self.tokenizer.encode('非')[1]
        else:
            self.yes_token = self.tokenizer.encode('yes')[1]
            self.no_token = self.tokenizer.encode('no')[1]

        if args.load_checkpoints_path != '':
            self.model = UniMCLitModel.load_from_checkpoint(
                args.load_checkpoints_path, args=args, yes_token=self.yes_token, model_path=model_path)
            print('load model from: ', args.load_checkpoints_path)
        else:
            self.model = UniMCLitModel(
                args, yes_token=self.yes_token, model_path=model_path)

    def train(self, train_data, dev_data, process=True):
        if process:
            train_data = self.preprocess(train_data)
            dev_data = self.preprocess(dev_data)
        data_model = UniMCDataModel(
            train_data, dev_data, self.yes_token, self.no_token, self.tokenizer, self.args)
        self.model.num_data = len(train_data)
        self.trainer.fit(self.model, data_model)

    def predict(self, test_data, cuda=True, process=True):
        if process:
            test_data = self.preprocess(test_data)

        result = []
        start = 0
        if cuda:
            self.model = self.model.cuda()
        self.model.model.eval()
        predict_model = UniMCPredict(
            self.yes_token, self.no_token, self.model, self.tokenizer, self.args)
        while start < len(test_data):
            batch_data = test_data[start:start+self.args.batchsize]
            start += self.args.batchsize
            batch_result = predict_model.predict(batch_data)
            result.extend(batch_result)
        if process:
            result = self.postprocess(result)
        return result

    def preprocess(self, data):

        for i, line in enumerate(data):
            if 'task_type' in line.keys() and line['task_type'] == '语义匹配':
                data[i]['choice'] = ['不能理解为：'+data[i]
                                     ['textb'], '可以理解为：'+data[i]['textb']]
                # data[i]['question']='怎么理解这段话？'
                data[i]['textb'] = ''

            if 'task_type' in line.keys() and line['task_type'] == '自然语言推理':
                data[i]['choice'] = ['不能推断出：'+data[i]['textb'],
                                     '很难推断出：'+data[i]['textb'], '可以推断出：'+data[i]['textb']]
                # data[i]['question']='根据这段话'
                data[i]['textb'] = ''

        return data

    def postprocess(self, data):
        for i, line in enumerate(data):
            if 'task_type' in line.keys() and line['task_type'] == '语义匹配':
                data[i]['textb'] = data[i]['choice'][0].replace('不能理解为：', '')
                data[i]['choice'] = ['不相似', '相似']
                ns = {}
                for k, v in data[i]['score'].items():
                    if '不能' in k:
                        k = '不相似'
                    if '可以' in k:
                        k = '相似'
                    ns[k] = v
                data[i]['score'] = ns
                data[i]['answer'] = data[i]['choice'][data[i]['label']]

            if 'task_type' in line.keys() and line['task_type'] == '自然语言推理':
                data[i]['textb'] = data[i]['choice'][0].replace('不能推断出：', '')
                data[i]['choice'] = ['矛盾', '自然', '蕴含']
                ns = {}
                for k, v in data[i]['score'].items():
                    if '不能' in k:
                        k = '矛盾'
                    if '很难' in k:
                        k = '自然'
                    if '可以' in k:
                        k = '蕴含'
                    ns[k] = v
                data[i]['score'] = ns
                data[i]['answer'] = data[i]['choice'][data[i]['label']]

        return data

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, top_k="", **tokenizer_kwargs):
        # Using "" as default argument because we're going to use `top_k=None` in user code to declare
        # "No top_k"
        preprocess_params = tokenizer_kwargs

        postprocess_params = {}
        if hasattr(self.model.config, "return_all_scores") and return_all_scores is None:
            return_all_scores = self.model.config.return_all_scores

        if isinstance(top_k, int) or top_k is None:
            postprocess_params["top_k"] = top_k
            postprocess_params["_legacy"] = False
        elif return_all_scores is not None:
            warnings.warn(
                "`return_all_scores` is now deprecated,  if want a similar funcionality use `top_k=None` instead of"
                " `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`.",
                UserWarning,
            )
            if return_all_scores:
                postprocess_params["top_k"] = None
            else:
                postprocess_params["top_k"] = 1

        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        return preprocess_params, {}, postprocess_params


def load_data(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        samples = [json.loads(line) for line in tqdm(lines)]
    return samples


def comp_acc(pred_data, test_data):
    corr = 0
    for i in range(len(pred_data)):
        if pred_data[i]['label'] == test_data[i]['label']:
            corr += 1
    return corr/len(pred_data)


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser.add_argument('--data_dir', default='./data', type=str)
    total_parser.add_argument('--train_data', default='train.json', type=str)
    total_parser.add_argument('--valid_data', default='dev.json', type=str)
    total_parser.add_argument('--test_data', default='test.json', type=str)
    total_parser.add_argument('--output_path', default='', type=str)
    total_parser = UniMCPiplines.piplines_args(total_parser)
    args = total_parser.parse_args()

    train_data = load_data(os.path.join(args.data_dir, args.train_data))
    dev_data = load_data(os.path.join(args.data_dir, args.valid_data))
    test_data = load_data(os.path.join(args.data_dir, args.test_data))

    dev_data_ori = copy.deepcopy(dev_data)

    model = UniMCPiplines(args)

    print(args.data_dir)

    if args.train:
        model.train(train_data, dev_data)
    result = model.predict(dev_data)
    for line in result[:20]:
        print(line)

    acc = comp_acc(result, dev_data_ori)
    print('acc:', acc)

    if args.output_path != '':
        test_result = model.predict(test_data)
        with open(args.output_path, 'w', encoding='utf8') as f:
            for line in test_result:
                json_data = json.dumps(line, ensure_ascii=False)
                f.write(json_data+'\n')


if __name__ == "__main__":
    main()
