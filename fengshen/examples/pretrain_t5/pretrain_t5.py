import time
from builtins import print
import sys
import os
import torch
import argparse
import json
import pytorch_lightning as pl
from transformers import MT5Config, MT5Tokenizer
from pytorch_lightning import Trainer, loggers
from transformers import MT5ForConditionalGeneration
from pytorch_lightning.callbacks import LearningRateMonitor
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'


class MT5PretrainModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--keep_tokens_path', default=None, type=str)
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        if args.tokenizer_type == 't5_tokenizer':
            if args.new_vocab_path is not None:
                # 用于从mt5继续训练，此时只保留中英文词表，spm采用新模型
                assert args.keep_tokens_path is not None
                keep_tokens = json.load(open(args.keep_tokens_path))
                self.model = MT5ForConditionalGeneration.from_pretrained(
                    args.pretrained_model_path)
                new_config = self.model.config
                new_config.vocab_size = len(keep_tokens)
                print('vocab_size:', new_config.vocab_size)

                new_state_dict = self.model.state_dict()
                select_index = torch.tensor(keep_tokens)
                new_state_dict['encoder.embed_tokens.weight'] = torch.index_select(
                    new_state_dict['encoder.embed_tokens.weight'], dim=0, index=select_index)
                new_state_dict['shared.weight'] = torch.index_select(
                    new_state_dict['shared.weight'], dim=0, index=select_index)
                new_state_dict['decoder.embed_tokens.weight'] = torch.index_select(
                    new_state_dict['decoder.embed_tokens.weight'], dim=0, index=select_index)
                new_state_dict['lm_head.weight'] = torch.index_select(
                    new_state_dict['lm_head.weight'], dim=0, index=select_index)
                self.model = MT5ForConditionalGeneration.from_pretrained(
                    args.pretrained_model_path, config=new_config, state_dict=new_state_dict)
                # self.model = MT5ForConditionalGeneration(config=new_config)
            else:
                # 用于继续训练
                self.model = MT5ForConditionalGeneration.from_pretrained(
                    args.pretrained_model_path
                )
        else:
            self.model = MT5ForConditionalGeneration(
                MT5Config.from_pretrained(args.pretrained_model_path)
            )

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batchsize * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * \
                float(self.trainer.max_epochs)
            self.total_steps = (
                len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'], labels=batch['labels'])
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss, sync_dist=True)
        self.log('train_acc', acc, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        # print('is out of index: ', batch['input_ids'][batch['input_ids'] >= 32598])
        output = self.model(
            input_ids=batch['input_ids'], labels=batch['labels'])
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/y_true.shape[0]
        return acc

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.global_rank == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

    def predict_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'], labels=batch['labels'])
        print('predict_loss', output.loss)
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
        )
        return {'predict_ids': generated_ids, 'labels': batch['labels'], 'input_ids': batch['input_ids']}


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def main():
    total_parser = argparse.ArgumentParser("Pretrain Unsupervise.")
    total_parser.add_argument(
        '--do_eval_only', action='store_true', default=False)
    total_parser.add_argument(
        '--pretrained_model_path', default=None, type=str)
    total_parser.add_argument(
        '--new_vocab_path', default=None, type=str)
    total_parser.add_argument('--max_seq_length', default=1024, type=int)
    total_parser.add_argument('--ckpt_path', default=None, type=str)
    sys.path.append('../../../')
    from fengshen.data.t5_dataloader.t5_datasets import UnsuperviseT5DataModel
    from fengshen.utils.universal_checkpoint import UniversalCheckpoint
    # * Args for data preprocessing
    total_parser = UnsuperviseT5DataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
    total_parser = MT5PretrainModel.add_model_specific_args(total_parser)
    # * Args for base model
    args = total_parser.parse_args()
    print('Argument parse success.')
    print('UnsuperviseT5DataModel load start {}'.format(get_time_str()))
    data_model = UnsuperviseT5DataModel(args)
    print('UnsuperviseT5DataModel load end {}'.format(get_time_str()))
    if not args.do_eval_only:
        model = MT5PretrainModel(args)
        checkpoint_callback = UniversalCheckpoint(args).callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(
            args.default_root_dir, 'logs/'))
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             callbacks=[checkpoint_callback, lr_monitor]
                                             )
        trainer.fit(model, data_model, ckpt_path=args.ckpt_path)
    else:
        tokenizer = MT5Tokenizer.from_pretrained(args.new_vocab_path, extra_ids=0)
        model = MT5PretrainModel(args=args, num_data=len(data_model.predict_dataloader()))
        trainer = Trainer.from_argparse_args(args)

        result = trainer.predict(model, data_model)
        result = result[0]
        for i in range(4):
            print(tokenizer.batch_decode(result['input_ids'][i]))
            print(tokenizer.batch_decode(result['predict_ids'][i]))
            print(tokenizer.batch_decode(result['labels'][i]))


if __name__ == '__main__':
    main()
