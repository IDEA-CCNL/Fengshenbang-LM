
import torch
import os
import argparse
import json
import pytorch_lightning as pl
from fengshen.models.model_utils import add_module_args
from fengshen.data.task_dataloader.task_datasets import AbstractCollator
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.utils.utils import chinese_char_tokenize
from torchmetrics.text.rouge import ROUGEScore
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
sys.path.append('../../../')


# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'


class FinetuneSummary(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--rouge_keys', default='rougeL,rouge1,rouge2', type=str)
        return parent_args

    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.pretrained_model_path)
        self.tokenizer = tokenizer
        assert self.tokenizer, "tokenizer is None!"
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
            print('total_steps is :', self.total_steps)

    def training_step(self, batch, batch_idx):
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], labels=batch['labels'])
        self.log('train_loss', output.loss, sync_dist=True)
        return output.loss

    def on_validation_start(self) -> None:
        # rm file at validation start
        prefix, ext = os.path.splitext(self.hparams.output_save_path)
        file_path_rank = '{}_{}{}'.format(
            prefix, self.trainer._accelerator_connector.cluster_environment.global_rank(), ext)
        if os.path.exists(file_path_rank):
            print('rm {}'.format(file_path_rank))
            os.remove(file_path_rank)

    def validation_step(self, batch, batch_idx):
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], labels=batch['labels'])
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.max_dec_length
        )

        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        labels = torch.where(batch['labels'] != -100, batch['labels'],
                             self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
        # reset the metric after once validation
        self.rouge_metric.reset()
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
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)


def main():
    total_parser = argparse.ArgumentParser("Summary Task")
    total_parser.add_argument('--do_eval_only',
                              action='store_true',
                              default=False)
    total_parser.add_argument('--pretrained_model_path',
                              default='google/mt5-small',
                              type=str)
    total_parser.add_argument('--output_save_path',
                              default='./predict.json',
                              type=str)
    total_parser.add_argument('--self_tokenizer',
                              action='store_true',
                              default=False)
    total_parser.add_argument('--max_enc_length', default=1024, type=int)
    total_parser.add_argument('--max_dec_length', default=256, type=int)
    total_parser.add_argument('--prompt', default='summarize:', type=str)
    # * Args for data preprocessing
    # from fengshen.data.task_dataloader.task_datasets import LCSTSDataModel
    total_parser = UniversalDataModule.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = add_module_args(total_parser)
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
    total_parser = FinetuneSummary.add_model_specific_args(total_parser)
    # * Args for base model
    args = total_parser.parse_args()

    if args.self_tokenizer:
        from fengshen.examples.pegasus.tokenizers_pegasus import PegasusTokenizer
        tokenizer = PegasusTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)
    collator = AbstractCollator(tokenizer, args.max_enc_length,
                                args.max_dec_length, args.prompt)
    data_model = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collator)
    model = FinetuneSummary(args, tokenizer)
    if not args.do_eval_only:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(
            args.default_root_dir, 'log/'))
        checkpoint_callback = UniversalCheckpoint(args)
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             callbacks=[lr_monitor,
                                                        checkpoint_callback]
                                             )
        trainer.fit(model, data_model)
    else:
        trainer = Trainer.from_argparse_args(args)
        # trainer.predict(model, data_model)
        trainer.validate(model, data_model)


if __name__ == '__main__':
    main()
