import fengshen.data.hubert.hubert_dataset as datasets
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.models.hubert.modeling_hubert import HubertModelForPretrain, HubertConfig
# from transformers.models.hubert.modeling_hubert import _compute_mask_indices
import argparse
from fairseq.data import Dictionary
from pytorch_lightning import (
    LightningModule,
    Trainer,
    loggers,
)
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import os
import math
import numpy as np

torch.set_printoptions(precision=2, profile="full")


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    def __call__(self, label: str):
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )


class HubertPretrainDataLoader():
    def __init__(self, args):
        self.cfg = args
        self.dictionaries = self.load_dictionaries()
        self.load_datasets = {}

    # TODO 改成HuggingFace Tokenizer
    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries

    def get_label_dir(self):
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    @property
    def datasets(self):
        return self.load_datasets

    def load_dataset(self, split: str, **kwargs):
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [f"{self.get_label_dir()}/{split}.{lb}" for lb in self.cfg.labels]

        # hubert v1: pad_audio=True, random_crop=False;
        self.load_datasets[split] = datasets.HubertDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
        )


def perpare_data(args):
    loader = HubertPretrainDataLoader(args)
    loader.load_dataset('train')
    loader.load_dataset('valid')
    return loader


class HubertLightning(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HuBert Lightning')
        parser.add_argument('--pred_masked_weight', type=float, default=1.0)
        parser.add_argument('--logit_temp', type=float, default=1.0)
        parser.add_argument('--loss_weights', type=float, nargs='+')
        parser.add_argument('--final_dim', type=int, default=0)
        parser.add_argument('--pred_nomask_weight', type=float, default=0)
        parser.add_argument('--skip_masked', type=bool, default=False)
        parser.add_argument('--skip_nomask', type=bool, default=False)
        parser.add_argument('--target_glu', type=bool, default=False)
        return parent_parser

    def __init__(self, args, loader, ** kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        config = HubertConfig.from_pretrained(args.model_path)
        config.pred_masked_weight = args.pred_masked_weight
        config.loss_weights = args.loss_weights
        config.logit_temp = args.logit_temp
        config.final_dim = args.final_dim
        config.pred_nomask_weight = args.pred_nomask_weight
        config.dictionaries = loader.dictionaries
        feature_ds_rate = np.prod(config.conv_stride)
        config.feat2tar_ratio = args.label_rate * feature_ds_rate / args.sample_rate
        config.skip_masked = args.skip_masked
        config.skip_nomask = args.skip_nomask
        config.target_glu = args.target_glu
        self.model = HubertModelForPretrain(config=config)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches
                self.total_steps = (len(train_loader) *
                                    self.trainer.max_epochs) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)

    def forward(self, **batch):

        target_list = batch['target_list']
        padding_mask = batch['net_input']['padding_mask']
        input_values = batch['net_input']['source']
        output = self.model(input_values=input_values,
                            attention_mask=padding_mask,
                            target_list=target_list,
                            mask_time_indices=None)
        return output

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        self.log('train_loss', output.loss / output.sample_size / math.log(2), sync_dist=True)
        for k, v in output.loss_m_dict.items():
            self.log(f'train_{k}', v / output.sample_size / math.log(2))
        for k, v in output.loss_u_dict.items():
            self.log(f'train_{k}', v / output.sample_size / math.log(2))
        for k, v in output.loss_extra_dict.items():
            self.log(f'train_{k}', v / output.sample_size / math.log(2))
        self.log('train_batch_size', float(batch['net_input']['source'].shape[0]))
        print(
            f'{self.trainer.fit_loop.epoch_loop._batches_that_stepped} batch: {batch["net_input"]["source"].shape}')
        # output.loss /= (output.sample_size * math.log(2))
        return output

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / y_true.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.log('val_loss', output.loss / output.sample_size / math.log(2), sync_dist=True)
        for k, v in output.loss_m_dict.items():
            self.log(f'val_{k}', v / output.sample_size / math.log(2), sync_dist=True)
        for k, v in output.loss_u_dict.items():
            self.log(f'val_{k}', v / output.sample_size / math.log(2), sync_dist=True)
        for k, v in output.loss_extra_dict.items():
            self.log(f'val_{k}', v / output.sample_size / math.log(2), sync_dist=True)
        # acc = self.comput_metrix(output.logits, batch['labels'])
        # self.log('val_acc', acc, sync_dist=True)
        return output

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
    from fengshen.utils import UniversalCheckpoint
    from fengshen.models.model_utils import add_module_args
    args_parser = add_module_args(args_parser)
    args_parser = datasets.add_data_specific_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = HubertLightning.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser.add_argument('--ckpt_path', type=str, )
    args = args_parser.parse_args()

    data_loader = perpare_data(args)
    data_module = UniversalDataModule(args=args, tokenizer=None,
                                      collate_fn=None, datasets=data_loader.datasets)
    module = HubertLightning(args, loader=data_loader)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'),
        name=os.path.basename(os.path.dirname(args.model_path)))
    checkpoint_callback = UniversalCheckpoint(args).callbacks

    if args.ckpt_path is not None and \
            not os.path.exists(args.ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.ckpt_path = None

    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(module, data_module, ckpt_path=args.ckpt_path)
