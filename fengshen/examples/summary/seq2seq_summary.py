from fengshen.utils.utils import chinese_char_tokenize
from torchmetrics.text.rouge import ROUGEScore
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from builtins import list, print
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning import Trainer, loggers
import pytorch_lightning as pl
import json
import argparse
import torch
import os
import sys
from pytorch_lightning.callbacks import LearningRateMonitor
sys.path.append('../../../')

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'


class FinetuneSummary(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        parser.add_argument('--rouge_keys', default='rougeL,rouge1,rouge2', type=str)
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.pretrained_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_model_path, use_fast=False)
        self.rouge_keys = tuple(args.rouge_keys.split(','))
        self.rouge_metric = ROUGEScore(rouge_keys=self.rouge_keys, normalizer=lambda x: x)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batchsize * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * \
                float(self.trainer.max_epochs)
            self.total_steps = (
                len(train_loader.dataset) // tb_size) // ab_size

    def training_step(self, batch, batch_idx):
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], labels=batch['labels'])
        self.log('train_loss', output.loss, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], labels=batch['labels'])
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.max_dec_length
        )

        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        labels = self.tokenizer.batch_decode(
            batch['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # save preds for every rank
        prefix, ext = os.path.splitext(self.hparams.output_save_path)
        file_path_rank = '{}_{}{}'.format(
            prefix, self.trainer._accelerator_connector.cluster_environment.global_rank(), ext)
        self.save_prediction_to_file(preds=preds, texts=batch['text'],
                                     summarys=batch['summary'], file_path=file_path_rank)
        # you need to split chinese char with space for rouge metric
        new_preds = [chinese_char_tokenize(p) for p in preds]
        new_labels = [chinese_char_tokenize(label) for label in labels]
        # update metric
        self.rouge_metric.update(preds=new_preds, target=new_labels)
        self.log('val_loss', output.loss, sync_dist=True)

    def validation_epoch_end(self, outputs):
        # compute metric for all process
        rouge_dict = self.rouge_metric.compute()
        for k, v in rouge_dict.items():
            self.log('val_{}'.format(k), v, sync_dist=True)
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            print('rouge:\n', rouge_dict)

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

    def save_prediction_to_file(self, preds, texts, summarys, file_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            for idx, pred in enumerate(preds):
                text = texts[idx]
                summary = summarys[idx]
                tmp_result = dict()
                tmp_result['pred'] = pred
                tmp_result['label'] = summary
                tmp_result['text'] = text
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data + '\n')

    def predict_step(self, batch, batch_idx):
        # print(batch)
        texts = batch['text']
        # output summary and metrics
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.max_dec_length
        )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        labels = self.tokenizer.batch_decode(
            batch['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(batch_idx, len(preds), len(labels))
        self.save_prediction_to_file(preds, texts, labels)

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


def main():
    total_parser = argparse.ArgumentParser("Summary Task")
    total_parser.add_argument(
        '--do_eval_only', action='store_true', default=False)
    total_parser.add_argument(
        '--pretrained_model_path', default='google/mt5-small', type=str)
    total_parser.add_argument('--output_save_path',
                              default='./predict.json', type=str)
    # * Args for data preprocessing
    from fengshen.data.task_dataloader.task_datasets import LCSTSDataModel
    total_parser = LCSTSDataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
    total_parser = FinetuneSummary.add_model_specific_args(total_parser)
    # * Args for base model
    args = total_parser.parse_args()

    data_model = LCSTSDataModel(args)
    if not args.do_eval_only:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        model = FinetuneSummary(args)
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(
            args.default_root_dir, 'log/'))
        checkpoint_callback = UniversalCheckpoint(args).callbacks
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             callbacks=[lr_monitor,
                                                        checkpoint_callback]
                                             )
        trainer.fit(model, data_model)
    else:
        trainer = Trainer.from_argparse_args(args)
        model = FinetuneSummary(args)
        # trainer.predict(model, data_model)
        trainer.validate(model, data_model)


if __name__ == '__main__':
    main()
