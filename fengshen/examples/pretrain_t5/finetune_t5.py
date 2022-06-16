import time
from builtins import print
import sys
import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from transformers import MT5ForConditionalGeneration
from pytorch_lightning.callbacks import LearningRateMonitor
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'


class MT5FinetuneModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--keep_tokens_path', default=None, type=str)
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = MT5ForConditionalGeneration.from_pretrained(
            args.pretrained_model_path
        )

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
                self.total_steps = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss, sync_dist=True)
        self.log('train_acc', acc, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        # print('is out of index: ', batch['input_ids'][batch['input_ids'] >= 32598])
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        acc = self.comput_metrix(output.logits, batch['labels'])
        cond_output = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            force_words_ids=batch['force_words_ids'],
            num_beams=2,
        )
        cond_acc = self.comput_metrix(cond_output, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)
        self.log('cond_acc', cond_acc, sync_dist=True)

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/y_true.shape[0]
        return acc

    def on_save_checkpoint(self, checkpoint) -> None:
        # Save the current loop info in the mid of epoch
        # if you lightning <= 1.6.0  uncomment the line below
        # checkpoint['loops'] = self.trainer.checkpoint_connector._get_loops_state_dict()
        if self.trainer.global_rank == 0 and self.trainer.global_step % self.hparams.every_n_train_steps == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(self.trainer.current_epoch, self.trainer.global_step)))

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


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
    from fengshen.data.t5_dataloader.t5_datasets import TaskT5DataModel
    from fengshen.utils.universal_checkpoint import UniversalCheckpoint
    # * Args for data preprocessing
    total_parser = TaskT5DataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
    total_parser = MT5FinetuneModel.add_model_specific_args(total_parser)
    # * Args for base model
    args = total_parser.parse_args()
    print('Argument parse success.')
    print('TaskT5DataModel load start {}'.format(get_time_str()))
    data_model = TaskT5DataModel(args)
    print('TaskT5DataModel load end {}'.format(get_time_str()))
    if not args.do_eval_only:
        model = MT5FinetuneModel(args)
        checkpoint_callback = UniversalCheckpoint(args).callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(
            args.default_root_dir, 'logs/'))
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             callbacks=[checkpoint_callback, lr_monitor]
                                             )
        trainer.fit(model, data_model, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
