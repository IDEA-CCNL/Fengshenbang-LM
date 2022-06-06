from data.bert_dataloader.load import BertDataModule
from transformers import (
    BertTokenizer,
    BertConfig,
    BertForPreTraining,
    BertModel,
    BertForMaskedLM
)
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    loggers,
    Trainer,
)
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from typing import Optional
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
import argparse
import sys
import torch
import os
import re
import jieba
import numpy as np

sys.path.insert(0, '/data0/wuziwei/codes/Fengshenbang-LM/fengshen')

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


class DataCollate(object):

    def __init__(self, tokenizer, max_length, mask_rate=0.15, max_ngram=3, if_padding=True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.word_cuter = jieba.cut
        self.vocab_length = len(tokenizer)
        self.mask_rate = mask_rate
        self.ignore_labels = -100
        self.ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
        self.pvals = pvals
        self.padding = if_padding

    def token_process(self, token_id):
        rand = np.random.random()
        if rand <= 0.8:
            return self.tokenizer.mask_token_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(1, self.vocab_length)

    def __call__(self, samples):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        batch_labels = []
        # print('^-^ batch size :',len(samples))
        for sample in samples:
            word_list = list(self.word_cuter(sample['text']))
            mask_ids, labels = [], []

            record = []
            for i in range(len(word_list)):
                rands = np.random.random()
                if i in record:
                    continue
                word = word_list[i]
                if rands > self.mask_rate and len(word) < 4:
                    word = word_list[i]
                    word_encode = tokenizer.encode(word, add_special_tokens=False)
                    for token in word_encode:
                        mask_ids.append(token)
                        labels.append(self.ignore_labels)
                    record.append(i)
                else:
                    n = np.random.choice(self.ngrams, p=self.pvals)
                    for index in range(n):
                        ind = index + i
                        if ind in record or ind >= len(word_list):
                            continue
                        record.append(ind)
                        word = word_list[ind]
                        word_encode = tokenizer.encode(word, add_special_tokens=False)
                        for token in word_encode:
                            mask_ids.append(self.token_process(token))
                            labels.append(token)
            if self.padding:
                if len(mask_ids) > self.max_length:
                    input_ids.append(mask_ids[:self.max_length])
                    batch_labels.append(labels[:self.max_length])
                else:
                    lenght = len(mask_ids)
                    mask_ids.extend([0]*(self.max_length-lenght))
                    labels.extend([-100]*(self.max_length-lenght))
                    input_ids.append(mask_ids)
                    batch_labels.append(labels)
            attention_mask.append([1]*self.max_length)
            token_type_ids.append([0]*self.max_length)

        #     print('sentence:',sample['text'])
        #     print('input_ids:',mask_ids)
        #     print('decode inputids:',self.tokenizer.decode(mask_ids))
        #     print('labels',labels)
        #     print('decode labels:',self.tokenizer.decode(labels))
        #     print('*'*20)
        # print('!!!!!!',torch.tensor(input_ids).shape)
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(batch_labels),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids)
        }


class MegatronBert(LightningModule):
    @staticmethod
    def add_module_specific_args(args_parser):
        parser = args_parser.add_argument_group('MegatronBert')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        return args_parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.bertconfig = BertConfig.from_pretrained(args.model_path)
        # self.model = BertForPreTraining(self.bertconfig)
        self.model = BertForMaskedLM(self.bertconfig)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batchsize * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(paras, lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_steps * self.hparams.warmup),
            self.total_steps)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        # print(output)
        self.log('train_loss', output.loss)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        # print(output)
        acc = self.comput_metrix(output.logits, batch['labels'])
        print('val_loss ', output.loss)
        self.log('val_loss', output.loss)
        self.log('val_acc', acc)
        # pass

    def predict_step(self, batch, batch_idx):
        output = self.model(**batch)
        return output.prediction_logits


class CustomCKPT:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('ckpt call back')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./ckpt/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)
        parser.add_argument('--save_last', action='store_true', default=True)
        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=100, type=float)
        parser.add_argument('--save_weights_only', action='store_true', default=False)

        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         every_n_train_steps=args.every_n_train_steps,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.dirpath,
                                         filename=args.filename,
                                         save_last=args.save_last)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = BertDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = MegatronBert.add_module_specific_args(args_parser)
    args_parser = CustomCKPT.add_argparse_args(args_parser)
    args_parser.add_argument('--deepspeed')
    args_parser.add_argument('--seq_max_length')

    args = args_parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    collate_fn = DataCollate(tokenizer, 512)
    data_module = BertDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)

    print('data load complete')

    model = MegatronBert(args)
    print('model load complete')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'),
        name=os.path.basename(os.path.dirname(args.model_path)))
    checkpoint_callback = CustomCKPT(args).callbacks

    if args.resume_from_checkpoint is not None and \
            not os.path.exists(args.resume_from_checkpoint):
        print('--------warning no checkpoint found--------, remove args')
        del args.resume_from_checkpoint

    # autotuning
    if args.deepspeed is not None:
        os.environ['PL_DEEPSPEED_CONFIG_PATH'] = args.deepspeed

    trainer = Trainer.from_argparse_args(args, logger=logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, data_module)
