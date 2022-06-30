from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration
from pytorch_lightning import (
    LightningModule,
    LightningDataModule,
    Trainer,
    loggers,
)
from pytorch_lightning.callbacks import LearningRateMonitor
import os
import argparse
import torch
from torch.utils.data import DataLoader
from typing import Optional
import jieba_fast as jieba
# import jieba as jieba
jieba.dt.tmp_dir = os.path.expanduser('~/.cache')
jieba.disable_parallel()


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
        self.zh_tokenizer = jieba.lcut
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
            dataset_type='bart',
            skip_warmup=True,
            zh_tokenizer=self.zh_tokenizer
        )

    def train_dataloader(self):
        import sys
        sys.path.append('../../../')
        from fengshen.data.universal_datamodule.universal_sampler import PretrainingSampler
        from fengshen.data.universal_datamodule.universal_datamodule import get_consume_samples
        consumed_samples = get_consume_samples(self)
        batch_sampler = PretrainingSampler(
            total_samples=len(self.train_dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.hparams.train_batchsize,
            data_parallel_rank=self.trainer.global_rank,
            data_parallel_size=self.trainer.world_size,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        # if only do a batch of validate, need shuffle the dataset
        shuffle = self.hparams.limit_val_batches is not None
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_dataset, shuffle=shuffle)
        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            batch_size=self.hparams.eval_batchsize,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.test_dataset, shuffle=False)
        return DataLoader(
            self.test_dataset,
            sampler=sampler,
            batch_size=self.hparams.test_batchsize,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


class BartLightning(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parent_parser.add_argument_group('Bart Lightning')
        return parent_parser

    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        config = BartConfig.from_pretrained(args.model_path)
        self.model = BartForConditionalGeneration(config=config)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches
                self.total_steps = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log('train_loss', output.loss, sync_dist=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / y_true.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        # Save the current loop info in the mid of epoch
        # if you lightning <= 1.6.0  uncomment the line below
        # checkpoint['loops'] = self.trainer.checkpoint_connector._get_loops_state_dict()
        if self.trainer.global_rank == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(self.trainer.current_epoch, self.trainer.global_step)))

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    import sys
    sys.path.append('../../../')

    from fengshen.utils import UniversalCheckpoint
    from fengshen.models.model_utils import add_module_args
    args_parser = add_module_args(args_parser)
    args_parser = MegatronDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = BartLightning.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser.add_argument('--deepspeed')
    args_parser.add_argument('--pretrain_sp_tokenizer', type=str, )
    args_parser.add_argument('--ckpt_path', type=str, )
    args = args_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    data_module = MegatronDataModule(tokenizer=tokenizer, args=args)

    module = BartLightning(args)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'),
        name=os.path.basename(os.path.dirname(args.model_path)))
    checkpoint_callback = UniversalCheckpoint(args).callbacks

    if args.ckpt_path is not None and \
            not os.path.exists(args.ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        del args.ckpt_path

    # autotuning
    if args.deepspeed is not None:
        os.environ['PL_DEEPSPEED_CONFIG_PATH'] = args.deepspeed

    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(module, data_module, ckpt_path=args.ckpt_path)
