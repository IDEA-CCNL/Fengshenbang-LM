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


class TCBertDataset(Dataset):
    def __init__(self, data, tokenizer, args, prompt_label, label_classes):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.num_labels = args.num_labels
        self.data = data
        self.args = args
        self.label_classes = label_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index])

    
    def encode(self, item, labeled=True):

        if labeled:
            texta = '这一句描述{}的内容如下：'.format(item['label']) + item['content']
            print('texta', texta)
            labels = self.label_classes[item['label']]

            encode_dict = self.tokenizer.encode_plus(texta,
                                            max_length=self.max_length,
                                            padding="longest",
                                            truncation=True
                                            )
        
            input_ids = encode_dict['input_ids']
            token_type_ids = encode_dict['token_type_ids']
            attention_mask = encode_dict['attention_mask']

            mlmlabels = copy.deepcopy(input_ids)
            mlmlabels[:] = [-100] * len(mlmlabels) 
            mlmlabels[6:8] = input_ids[6:8]
            input_ids[6:8] = [self.tokenizer.mask_token_id] * 2

            encoded = {
                "sentence": item["content"],
                "input_ids": torch.tensor(input_ids).long(),
                "token_type_ids": torch.tensor(token_type_ids).long(),
                "attention_mask": torch.tensor(attention_mask).float(),
                "labels": torch.tensor(labels).long(),
                "mlmlabels": torch.tensor(mlmlabels).long(),
            }

        else:

            texta = '这一句描述{}的内容如下：'.format('[MASK][MASK]')  + item['content']

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
        return parent_args

    def __init__(self, train_data, val_data, tokenizer, args, prompt_label):
        super().__init__()
        self.batchsize = args.batchsize
        self.label_classes = self.get_label_classes(prompt_label)
        args.num_labels = len(self.label_classes)

        self.train_data = TCBertDataset(train_data, tokenizer, args, prompt_label, self.label_classes)
        # print("self.train_data", next(iter(self.train_data)))
        self.valid_data = TCBertDataset(val_data, tokenizer, args, prompt_label, self.label_classes)
        # print("self.valid_data", next(iter(self.valid_data)))
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
            # mlmlabels = torch.LongTensor(batch_data['mlmlabels'])
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
        if self.config.model_type == 'megatron-bert':
            self.bert = MegatronBertForMaskedLM.from_pretrained(pre_train_dir)
        elif self.config.model_type == 'deberta-v2':
            self.bert = DebertaV2ForMaskedLM.from_pretrained(pre_train_dir)
        elif self.config.model_type == 'albert':
            self.bert = AlbertForMaskedLM.from_pretrained(pre_train_dir)
        else:
            self.bert = BertForMaskedLM.from_pretrained(pre_train_dir)

        self.dropout = nn.Dropout(0.1)
        self.nlabels = nlabels
        self.linear_classifier = nn.Linear(self.config.hidden_size, self.nlabels)

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids=None, mlmlabels=None, clslabels=None, clslabels_mask=None, mlmlabels_mask=None, sentence=None, labels=None):

        batch_size, seq_len = input_ids.shape
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
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model = TCBertModel(model_path, nlabels)

    def setup(self, stage) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data /
                                  (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def training_step(self, batch, batch_idx):
        print("training_step batch", batch)
        labels = batch['labels']
        mlm_loss, logits, cls_logits = self.model(**batch)
        if labels is not None:
            cls_loss = self.loss_fn(logits, labels.view(-1))

        loss = cls_loss + 0.5 * mlm_loss

        ntotal = logits.size(0)
        ncorrect = (logits.argmax(dim=-1) == labels).long().sum()
        acc = ncorrect / ntotal

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        print("validation_step batch", batch)
        mlm_loss, logits, cls_logits = self.model(**batch)
        labels = batch['labels']
        
        predict = logits.argmax(dim=-1).cpu().tolist()
        if labels is not None:
            cls_loss = self.loss_fn(logits, labels.view(-1))

        loss = cls_loss + 0.5 * mlm_loss
        ntotal = logits.size(0)
        
        ncorrect = int((logits.argmax(dim=-1) == batch['labels']).long().sum())
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
    def __init__(self, model, tokenizer, args, prompt_label):
        self.tokenizer = tokenizer
        self.args = args
        self.data_model = TCBertDataModel(
            [], [], tokenizer, args, prompt_label)
        self.model = model

    def predict(self, batch_data):
        batch = [self.data_model.train_data.encode(
            sample, labeled=False) for sample in batch_data]
        batch = self.data_model.collate_fn(batch)
        batch = {k: v.cuda() for k, v in batch.items()}
        _, logits, _ = self.model.model(**batch)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicts = torch.argmax(probs, dim=-1).detach().cpu().numpy()


        return predicts


class TCBertPipelines(Pipeline):
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

        total_parser = TCBertDataModel.add_data_specific_args(total_parser)
        total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
        total_parser = TCBertLitModel.add_model_specific_args(total_parser)
        total_parser = pl.Trainer.add_argparse_args(parent_args)
        return parent_args

    def __init__(self, args, model_path, nlables):
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


        if args.load_checkpoints_path != '':
            self.model = TCBertLitModel.load_from_checkpoint(
                args.load_checkpoints_path, args=args, model_path=model_path, nlables=nlables)
            print('load model from: ', args.load_checkpoints_path)
        else:
            self.model = TCBertLitModel(
                args, model_path=model_path, nlables=nlables)

    def train(self, train_data, dev_data, process=True):
        if process:
            train_data = self.preprocess(train_data)
            dev_data = self.preprocess(dev_data)
        data_model = TCBertDataModel(
            train_data, dev_data, self.tokenizer, self.args)
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
        predict_model = TCBertPredict(
            self.yes_token, self.no_token, self.model, self.tokenizer, self.args)
        while start < len(test_data):
            batch_data = test_data[start:start+self.args.batchsize]
            start += self.args.batchsize
            batch_result = predict_model.predict(batch_data)
            result.extend(batch_result)
        if process:
            result = self.postprocess(result)
        return result

    
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
    total_parser = TCBertPipelines.piplines_args(total_parser)
    args = total_parser.parse_args()

    train_data = load_data(os.path.join(args.data_dir, args.train_data))
    dev_data = load_data(os.path.join(args.data_dir, args.valid_data))
    test_data = load_data(os.path.join(args.data_dir, args.test_data))

    dev_data_ori = copy.deepcopy(dev_data)

    model = TCBertPipelines(args)

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











