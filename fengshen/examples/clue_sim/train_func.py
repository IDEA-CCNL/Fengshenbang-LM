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


import json
import os
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
from collections import defaultdict
from transformers import AutoConfig, AutoModel, MegatronBertConfig, MegatronBertModel, get_cosine_schedule_with_warmup


class FocalLoss(torch.nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss

class DiceLoss(torch.nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        prob = torch.softmax(input, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()

        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)


class CustomDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, mode='no_test'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        self.ex_list = []
        with open('./dataset/' + file, "r", encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                query = sample["query"]
                title = sample["title"]
                id = int(sample["id"])
                if self.mode == 'no_test':
                    relevant = int(sample["label"])
                    self.ex_list.append((query, title, relevant, id))
                else:
                    self.ex_list.append((query, title, id))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, index):
        if self.mode == 'no_test':
            query, title, relevant, id = self.ex_list[index]
        else:
            query, title, id = self.ex_list[index]

        inputs = self.tokenizer.encode_plus(
            query, title,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        if self.mode == 'no_test':
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(relevant, dtype=torch.float),
                'id': torch.tensor(id, dtype=torch.long)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'id': torch.tensor(id, dtype=torch.long)
            }

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.max_len = self.args.max_seq_length
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage):
        data_path = "/cognitive_comp/wangjunjie/projects/clue_sim/QBQTC/dataset"
        assert os.path.exists(os.path.join(data_path, 'train.json'))
        assert os.path.exists(os.path.join(data_path, 'dev.json'))
        assert os.path.exists(os.path.join(data_path, 'test_public.json'))
        if stage == 'fit':
            self.train_dataset = CustomDataset('train.json', self.tokenizer, self.max_len)
            self.val_dataset = CustomDataset('dev.json', self.tokenizer, self.max_len)
            self.test_dataset = CustomDataset('test_public.json', self.tokenizer, self.max_len)
        elif stage == 'test':
            self.test_dataset = CustomDataset('test_public.json', self.tokenizer, self.max_len)
    
    def train_dataloader(self):
        full_dataset = ConcatDataset([self.train_dataset, self.val_dataset])
        train_dataloader = DataLoader(
            full_dataset, 
            batch_size=self.args.batch_size, 
            num_workers=4,
            shuffle=True,  
            pin_memory=True, 
            drop_last=True)
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True, 
            drop_last=False)
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True, 
            drop_last=False)
        return test_dataloader

class CustomModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = self.args.model_name
        self.cache_dir = self.args.model_path
        self.scheduler = self.args.scheduler
        self.step_scheduler_after = "batch"
        self.optimizer = self.args.optimizer
        self.pooler = self.args.use_original_pooler
        self.category = self.args.cate_performance
        self.loss_func = self.args.loss_function

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(self.model, cache_dir=self.cache_dir)

        config.update(
            {
                "output_hidden_states": False,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
            }
        )
        self.transformer = AutoModel.from_pretrained(self.model, config=config, cache_dir=self.cache_dir)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = torch.nn.Linear(config.hidden_size, self.args.num_labels, bias=True)  # 分三类

    def configure_optimizers(self):
        """Prepare optimizer and schedule"""
        model = self.transformer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer_index = ['Adam', 'AdamW'].index(self.optimizer)
        optimizer = [
            torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.learning_rate),
            torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)][optimizer_index]
        
        scheduler_index = ['StepLR', 'CosineWarmup', 'CosineAnnealingLR'].index(self.scheduler)
        scheduler = [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.warmup_step, gamma=self.args.warmup_proportion),
            get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.args.warmup_proportion * self.total_steps),
                num_training_steps=self.total_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=2e-06)][scheduler_index]

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage != "fit":
            return
        # calculate total steps
        train_dataloader = self.trainer.datamodule.train_dataloader()
        gpus = 0 if self.trainer.gpus is None else self.trainer.gpus
        tb_size = self.args.batch_size * max(1, gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_dataloader.dataset) // tb_size) // ab_size

    def loss(self, outputs, targets):
        lossf_index = ['CE', 'Focal', 'Dice', 'LSCE'].index(self.loss_func)
        loss_fct = [nn.CrossEntropyLoss(), FocalLoss(), DiceLoss(), LabelSmoothingCrossEntropy(), LabelSmoothingCorrectionCrossEntropy()][lossf_index]
        loss = loss_fct(outputs, targets)
        return loss
    
    def category_performance_measure(self, labels_right, labels_pred, num_label=3):
        text_labels = [i for i in range(num_label)]

        TP = dict.fromkeys(text_labels, 0)  # 预测正确的各个类的数目
        TP_FP = dict.fromkeys(text_labels, 0)  # 测试数据集中各个类的数目
        TP_FN = dict.fromkeys(text_labels, 0)  # 预测结果中各个类的数目

        label_dict = defaultdict(list)
        for num in range(num_label):
            label_dict[num].append(str(num))

        # 计算TP等数量
        for i in range(0, len(labels_right)):
            TP_FP[labels_right[i]] += 1
            TP_FN[labels_pred[i]] += 1
            if labels_right[i] == labels_pred[i]:
                TP[labels_right[i]] += 1
                
        # 计算准确率P，召回率R，F1值
        results = []
        for key in TP_FP:
            P = float(TP[key]) / float(TP_FP[key] + 1e-9)
            R = float(TP[key]) / float(TP_FN[key] + 1e-9)
            F1 = P * R * 2 / (P + R) if (P + R) != 0 else 0
            #results.append("%s:\t P:%f\t R:%f\t F1:%f" % (key, P, R, F1))
            results.append(F1)
        return results

    def monitor_metrics(self, outputs, targets):
        pred = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        targets = targets.int().cpu().numpy().tolist()
        if self.category:
            category_results = self.category_performance_measure(
                labels_right=targets, 
                labels_pred=pred, 
                num_label=self.args.num_labels
            )
            return {"f1": category_results}
        else:
            f1_score = metrics.f1_score(targets, pred, average="macro")
            return {"f1": f1_score}

    def forward(self, ids, mask, token_type_ids, labels):
        transformer_out = self.transformer(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)

        if self.pooler:
            pooler_output = transformer_out.pooler_output
        else:
            sequence_output = transformer_out.last_hidden_state
            pooler_output = torch.mean(sequence_output, dim=1)
        logits = self.linear(self.dropout(pooler_output))

        labels_hat = torch.argmax(logits, dim=1)
        correct_count = torch.sum(labels == labels_hat)
        return logits, correct_count
    
    def predict(self, ids, mask, token_type_ids):
        transformer_out = self.transformer(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooler_output = transformer_out.pooler_output
        logits = self.linear(self.dropout(pooler_output))
        logits = torch.argmax(logits, dim=1)
        return logits
    
    def training_step(self, batch, batch_idx):
        ids, mask, token_type_ids, labels = batch['ids'], batch['mask'], batch['token_type_ids'], batch['targets']
        logits, correct_count = self.forward(ids, mask, token_type_ids, labels)
        loss = self.loss(logits, labels.long())
        f1 = self.monitor_metrics(logits, labels)["f1"]
        self.log("train_loss", loss, logger=True, prog_bar=True)
        self.log('train_acc', correct_count.float() / len(labels), logger=True, prog_bar=True)
        if self.category:
            self.log("train_f1_key0", f1[0], logger=True, prog_bar=True)
            self.log("train_f1_key1", f1[1], logger=True, prog_bar=True)
            self.log("train_f1_key2", f1[2], logger=True, prog_bar=True)
        else:
            self.log("train_f1", f1, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask, token_type_ids, labels = batch['ids'], batch['mask'], batch['token_type_ids'], batch['targets']
        logits, correct_count = self.forward(ids, mask, token_type_ids, labels)
        loss = self.loss(logits, labels.long())
        f1 = self.monitor_metrics(logits, labels)["f1"]
        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_acc", correct_count.float() / len(labels), logger=True, prog_bar=True)
        if self.category:
            self.log("val_f1_key0", f1[0], logger=True, prog_bar=True)
            self.log("val_f1_key1", f1[1], logger=True, prog_bar=True)
            self.log("val_f1_key2", f1[2], logger=True, prog_bar=True)
        else:
            self.log("val_f1", f1, logger=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        ids, mask, token_type_ids, labels = batch['ids'], batch['mask'], batch['token_type_ids'], batch['targets']
        logits, correct_count = self.forward(ids, mask, token_type_ids, labels)
        loss = self.loss(logits, labels.long())
        f1 = self.monitor_metrics(logits, labels)["f1"]
        self.log("test_loss", loss, logger=True, prog_bar=True)
        self.log("test_acc", correct_count.float() / len(labels), logger=True, prog_bar=True)
        if self.category:
            self.log("test_f1_key0", f1[0], logger=True, prog_bar=True)
            self.log("test_f1_key1", f1[1], logger=True, prog_bar=True)
            self.log("test_f1_key2", f1[2], logger=True, prog_bar=True)
        else:
            self.log("test_f1", f1, logger=True, prog_bar=True)
        return {"test_loss": loss, "logits":logits, "labels": labels}
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        ids, mask, token_type_ids, id = batch['ids'], batch['mask'], batch['token_type_ids'], batch['id']
        logits = self.predict(ids, mask, token_type_ids)
        return {'id':id.cpu().numpy().tolist(), 'logits': logits.cpu().numpy().tolist()}
