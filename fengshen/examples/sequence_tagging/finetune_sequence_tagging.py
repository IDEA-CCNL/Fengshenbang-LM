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


from dataclasses import dataclass
import copy
import logging
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
from fengshen.data.sequence_tagging_dataloader.sequence_tagging_collator import CollatorForLinear, CollatorForCrf, CollatorForSpan, CollatorForBiaffine
from fengshen.data.sequence_tagging_dataloader.sequence_tagging_datasets import DataProcessor, get_datasets
from fengshen.metric.metric import EntityScore
from fengshen.models.model_utils import configure_optimizers, get_total_steps
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.data.universal_datamodule import UniversalDataModule

from transformers import (
    BertTokenizer, BertConfig, AutoTokenizer
)
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

_validation_dict={
    'linear': 'validation_linear',
    'crf': 'validation_crf',
    'span': 'validation_span',
    'biaffine': 'validation_biaffine',
}

_prediction_dict={
    'linear': 'predict_linear',
    'crf': 'predict_crf',
    'span': 'predict_span',
    'biaffine': 'predict_biaffine',
}

logger = logging.getLogger(__name__)


class LitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument("--max_seq_length", default=512, type=int)
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--model_type', default='bert', type=str)
        parser.add_argument("--decode_type", default="linear", choices=["linear", "crf", "biaffine", "span"], type=str)
        parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'])
        return parent_args

    def __init__(self, args, id2label, tokenizer):
        super().__init__()

        self.model_name=args.model_type+"-"+args.decode_type
        self.id2label = id2label
        
        self.config=BertConfig.from_pretrained(args.model_path)
        self.tokenizer = tokenizer
        self.model = _model_dict[self.model_name].from_pretrained(args.model_path, config=self.config, num_labels=len(self.id2label), loss_type=args.loss_type)
        self.entity_score=EntityScore()

        self.validate_fn=getattr(self,_validation_dict[args.decode_type])
        self.predict_fn=getattr(self,_prediction_dict[args.decode_type])

        self.predict_result=[]

        self.save_hyperparameters(args)
        
    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.validate_fn(batch,batch_idx)

    def validation_linear(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        preds = preds.detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()

        for i, label in enumerate(labels):
            y_true = []
            y_pred = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == (torch.sum(batch['attention_mask'][i]).item()-1):
                    true_subject=get_entities(y_true,self.id2label)
                    pred_subject=get_entities(y_pred,self.id2label)
                    self.entity_score.update(true_subject=true_subject, pred_subject=pred_subject)
                    break
                else:
                    y_true.append(self.id2label[labels[i][j]])
                    y_pred.append(self.id2label[preds[i][j]])
        
        self.log('val_loss', loss)

    def validation_crf(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = self.model.crf.decode(logits, batch['attention_mask'])
        preds = preds.detach().squeeze(0).cpu().numpy().tolist()
        labels = batch['labels'].detach().cpu().numpy()

        for i, label in enumerate(labels):
            y_true = []
            y_pred = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == (torch.sum(batch['attention_mask'][i]).item()-1):
                    true_subject=get_entities(y_true,self.id2label)
                    pred_subject=get_entities(y_pred,self.id2label)
                    self.entity_score.update(true_subject=true_subject, pred_subject=pred_subject)
                    break
                else:
                    y_true.append(self.id2label[labels[i][j]])
                    y_pred.append(self.id2label[preds[i][j]])

        self.log('val_loss', loss)

    def validation_span(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        labels=batch['subjects']
        for i, T in enumerate(labels):
            active_start_logits=start_logits[i][:batch['input_len'][i]]
            active_end_logits=end_logits[i][:batch['input_len'][i]]
            R = bert_extract_item(active_start_logits, active_end_logits)

            T=T[~torch.all(T==-1,dim=-1)].cpu().numpy()
            T=list(map(lambda x:(self.id2label[x[0]],x[1],x[2]),T))
            R=list(map(lambda x:(self.id2label[x[0]],x[1],x[2]),R))

            self.entity_score.update(true_subject=T, pred_subject=R)
        self.log('val_loss', loss)

    def validation_biaffine(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.span_logits

        preds = torch.argmax(logits.cpu().numpy(), axis=-1)
        labels = batch['span_labels'].cpu().numpy()

        for i, label in enumerate(labels):
            input_len=(batch['input_len'][i])-2
            active_label=labels[i,1:input_len+1,1:input_len+1]
            active_pred=preds[i,1:input_len+1,1:input_len+1]

            temp_1 = []
            temp_2 = []

            for j in range(input_len):
                for k in range(input_len):
                    if self.id2label[active_label[j,k]]!="O":
                        temp_1.append([self.id2label[active_label[j,k]],j,k])
                    if self.id2label[active_pred[j,k]]!="O":
                        temp_2.append([self.id2label[active_pred[j,k]],j,k])

            self.entity_score.update(pred_subject=temp_2, true_subject=temp_1)

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

    def predict_step(self, batch, batch_idx):
        batch['labels'] = None
        outputs = self.model(**batch)

        self.predict_fn(batch,batch_idx)

    def predict_linear(self, batch, outputs):
        logits = torch.argmax(F.log_softmax(outputs.logits, dim=2), dim=2)
        preds = logits.detach().cpu().numpy()

        for i, pred in enumerate(preds):
            text = self.tokenizer.convert_ids_to_tokens(batch['input_ids'][i])[:batch['input_len'][i]][1:-1]
            pred = pred[:batch['input_len'][i]][1:-1]
            label_entities = get_entities(pred, self.id2label)
            for label_list in label_entities:
                label_list.append("".join(text[label_list[1]:label_list[2]+1]))

            self.predict_result.extend(label_entities)

    def predict_crf(self, batch, batch_idx):
        logits = self.model(**batch).logits
        preds = self.model.crf.decode(logits, batch['attention_mask']).squeeze(0).cpu().numpy().tolist()

        for i, pred in enumerate(preds):
            text = self.tokenizer.convert_ids_to_tokens(batch['input_ids'][i])[:batch['input_len'][i]][1:-1]
            pred = pred[:batch['input_len'][i]][1:-1]
            label_entities = get_entities(pred, self.id2label)
            for label_list in label_entities:
                label_list.append("".join(text[label_list[1]:label_list[2]+1]))
        
            self.predict_result.extend(label_entities)

    def predict_span(self, batch, batch_idx):
        batch['start_positions'] = None
        batch['end_positions'] = None
        outputs = self.model(**batch)

        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        for i, _ in enumerate(start_logits):
            text = self.tokenizer.convert_ids_to_tokens(batch['input_ids'][i])[:batch['input_len'][i]][1:-1]
            R = bert_extract_item(start_logits[i][:batch['input_len'][i]], end_logits[i][:batch['input_len'][i]])
            if R:
                label_entities = [[self.id2label[x[0]],x[1],x[2],"".join(text[x[1]:x[2]+1])] for x in R]
            else:
                label_entities = []
            
            self.predict_result.extend(label_entities)

    

    def configure_optimizers(self):
        return configure_optimizers(self)

def main():
    total_parser = argparse.ArgumentParser("TASK NAME")

    # * Args for data preprocessing
    total_parser = UniversalDataModule.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = pl.Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)

    # * Args for base model
    from fengshen.models.model_utils import add_module_args
    total_parser = add_module_args(total_parser)
    total_parser = LitModel.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    datasets=get_datasets(args)

    checkpoint_callback = UniversalCheckpoint(args).callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[checkpoint_callback, lr_monitor]
                                            )

    label2id,id2label=DataProcessor.get_labels(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    collator = _collator_dict[args.decode_type]()
    collator.args=args
    collator.tokenizer=tokenizer
    collator.label2id=label2id                                        
    data_model = UniversalDataModule(tokenizer,collator,args,datasets)

    model = LitModel(args,id2label,tokenizer)
    print(label2id)
    trainer.fit(model, data_model)
    # trainer.predict(model,dataloaders=data_model.predict_dataloader())

if __name__ == "__main__":
    main()