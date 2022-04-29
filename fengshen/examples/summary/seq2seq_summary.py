from fengshen.utils.universal_checkpoint import UniversalCheckpoint
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

# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'


class FinetuneSummary(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.pretrained_model_path)

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
        self.log('val_loss', output.loss, sync_dist=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

    def predict_step(self, batch, batch_idx):
        text = batch['text']
        summary = batch['summary']
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.args.max_dec_length
        )
        return {"pred": generated_ids, "text": text, "summary": summary}

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


def save_test(data, args, data_model):
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path, use_fast=False)
    with open(os.path.join(args.output_save_path), 'w', encoding='utf-8') as f:
        for _, batch in enumerate(data):
            texts = batch['text']
            summarys = batch['summary']
            preds = batch['pred']
            for idx, pred_ids in enumerate(preds):
                text = texts[idx]
                summary = summarys[idx]
                tmp_result = dict()
                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for g in pred_ids]
                tmp_result['summary'] = ''.join(preds)
                tmp_result['label'] = summary
                tmp_result['origin_text'] = text
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data + '\n')
    print('save the result to ' + args.output_save_path)


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
        model = FinetuneSummary.load_from_checkpoint(
            args.resume_from_checkpoint, args=args)
        result = trainer.predict(model, data_model)
        if torch.distributed.get_rank() == 0:
            save_test(result, args, data_model)


if __name__ == '__main__':
    main()
