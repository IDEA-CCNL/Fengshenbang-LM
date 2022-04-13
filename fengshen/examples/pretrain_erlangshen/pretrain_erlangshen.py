from transformers import (
    BertTokenizer,
    MegatronBertForPreTraining,
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

sys.path.append('../../')


class MegatronDataModule(LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('MegatronDataModule')
        parser.add_argument('--data_prefix', type=str, default='')
        parser.add_argument('--samples', type=int, nargs='+')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_batchsize', default=32, type=int)
        parser.add_argument('--eval_batchsize', default=32, type=int)
        parser.add_argument('--test_batchsize', default=32, type=int)
        return parent_parser

    def __init__(self, tokenizer, args, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.save_hyperparameters(args)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != 'fit' and stage != 'validate':
            raise NotImplementedError
        from fengshen.data.megatron_dataloader.dataset_utils import build_train_valid_test_datasets
        self.train_dataset, self.val_dataset, self.test_dataset = build_train_valid_test_datasets(
            data_prefix=[self.hparams.data_prefix],
            data_impl='mmap',
            splits_string='949,50,1',
            train_valid_test_num_samples=self.hparams.samples,
            max_seq_length=512,
            masked_lm_prob=0.15,
            short_seq_prob=0.1,
            seed=1234,
            tokenizer=self.tokenizer,
            dataset_type='bert_cn_wwm',
            binary_head=True,
            skip_warmup=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batchsize,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.eval_batchsize,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.test_batchsize,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class MegatronBertLGTN(LightningModule):
    @staticmethod
    def add_module_specific_args(args_parser):
        parser = args_parser.add_argument_group('MegatronBertLGTN')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        return args_parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = MegatronBertForPreTraining.from_pretrained(args.model_path)

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
        self.log('train_loss', output.loss)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        acc = self.comput_metrix(output.prediction_logits, batch['labels'])
        print('val_loss ', output.loss)
        self.log('val_loss', output.loss)
        self.log('val_acc', acc)

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
    args_parser = MegatronDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = MegatronBertLGTN.add_module_specific_args(args_parser)
    args_parser = CustomCKPT.add_argparse_args(args_parser)
    args_parser.add_argument('--deepspeed')
    args = args_parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    data_module = MegatronDataModule(tokenizer, args)
    print('data load complete')

    model = MegatronBertLGTN(args)
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
