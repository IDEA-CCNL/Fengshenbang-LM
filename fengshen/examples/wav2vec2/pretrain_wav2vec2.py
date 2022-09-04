from transformers import Wav2Vec2FeatureExtractor
from fengshen.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from fengshen.models.wav2vec2.modeling_wav2vec import Wav2Vec2ForPreTraining

import fengshen.data.wav2vec2.wav2vec2_dataset as wv2_datasets
from fengshen.data.universal_datamodule import UniversalDataModule
# from transformers.models.hubert.modeling_hubert import _compute_mask_indices
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
    loggers,
)
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import os
import math


class Wav2vec2PretrainDataLoader():
    def __init__(self, args, model):
        self.args = args
        self.load_datasets = {}
        self.wav2vec2_model = model.model
        self.feature_extractor = model.feature_extractor

    @property
    def datasets(self):
        return self.load_datasets

    def load_dataset(self, split: str, **kwargs):
        data_path = self.args.data
        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        self.datasets[split] = wv2_datasets.Wav2vec2Dataset(
            manifest_path=manifest_path,
            sample_rate=self.args.sample_rate,
            model=self.wav2vec2_model,
            feature_extractor=self.feature_extractor,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.min_sample_size,
            pad=self.args.labels is not None or self.args.enable_padding,
            normalize=self.args.normalize,
            max_tokens=args.max_tokens
        )


def prepare_data(args, model):
    loader = Wav2vec2PretrainDataLoader(args, model)
    loader.load_dataset('train')
    loader.load_dataset('valid')
    return loader


class Wav2vec2Lightning(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Wav2vec2 Lightning')
        parser.add_argument(
            "--audio_column_name",
            type=str,
            default="audio",
            help="Column in the dataset that contains speech file path. Defaults to 'audio'",
        )
        parser.add_argument(
            "--max_gumbel_temperature",
            type=float,
            default=2.0,
            help="Maximum temperature for gumbel softmax.",
        )
        parser.add_argument(
            "--min_gumbel_temperature",
            type=float,
            default=0.5,
            help="Minimum temperature for gumbel softmax.",
        )
        parser.add_argument(
            "--gumbel_temperature_decay", type=float, default=0.999995,
            help="Decay of gumbel temperature during training."
        )
        parser.add_argument(
            "--pad_to_multiple_of",
            type=int,
            default=None,
            help=(
                "If set will pad the sequence to a multiple of the provided value. "
                "This is especially useful to enable the"
                " use of Tensor Cores on NVIDIA hardware with compute capability"
                " >= 7.5 (Volta)."
            ),
        )
        return parent_parser

    def __init__(self, args, ** kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.model_path)
        config = Wav2Vec2Config.from_pretrained(args.model_path)
        self.config = config
        self.model = Wav2Vec2ForPreTraining(config=config)
        self.completed_steps = 0
        # used to update gumbel temperature
        self.sub_steps = 0
        # used for accumulate gradient

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
                self.total_steps = self.trainer.max_steps

            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)

    def forward(self, **batch):
        batch.pop("sub_attention_mask", None)
        output = self.model(
            **batch
        )
        return output

    def multiply_grads(self, params, c):
        """Multiplies grads by a constant *c*."""
        for p in params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def compute_metrix(self, logits, mask_time_indices):
        mask_time_indices = mask_time_indices.transpose(0, 1).flatten()
        max = (logits.argmax(-1) == 0) & mask_time_indices
        min = logits.argmin(-1) == 0 & mask_time_indices
        count = float(mask_time_indices.sum())
        both = max & min
        corr = max.long().sum().item() - both.long().sum().item()
        return corr/count

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        num_losses = batch["mask_time_indices"].sum()
        if self.trainer.accumulate_grad_batches and (self.sub_steps % self.trainer.accumulate_grad_batches == 0):
            gumbel_temperature = max(
                self.args.max_gumbel_temperature *
                self.args.gumbel_temperature_decay**self.completed_steps,
                self.args.min_gumbel_temperature,
            )
            if hasattr(self.model, "module"):
                self.model.module.set_gumbel_temperature(gumbel_temperature)
            else:
                self.model.set_gumbel_temperature(gumbel_temperature)
            self.completed_steps += 1

        self.sub_steps += 1
        logs = {
            "loss": output.loss/num_losses/math.log(2),
            'contrast_loss': output.contrastive_loss/num_losses/math.log(2),
            'div_loss': self.config.diversity_loss_weight * output.diversity_loss/num_losses/math.log(2)
        }
        for k in logs:
            self.log(f'train_{k}', logs[k])

        return {
            "loss": output.loss,
        }

    def validation_step(self, batch, batch_idx):
        # print([item for item in batch])
        num_losses = batch["mask_time_indices"].sum()
        output = self(return_logits=True, **batch)
        acc = self.compute_metrix(output.logits, batch["mask_time_indices"])
        logs = {
            "loss": output.loss/num_losses/math.log(2),
            'contrast_loss': output.contrastive_loss/num_losses/math.log(2),
            'div_loss': self.config.diversity_loss_weight * output.diversity_loss/num_losses/math.log(2)
        }
        for k in logs:
            self.log(f'val_{k}', logs[k], sync_dist=True)
        self.log("acc", acc, sync_dist=True)
        return {
            "loss": output.loss/num_losses,
        }

    def on_save_checkpoint(self, checkpoint) -> None:
        # Save the current loop info in the mid of epoch
        # if you lightning <= 1.6.0  uncomment the line below
        # checkpoint['loops'] = self.trainer.checkpoint_connector._get_loops_state_dict()
        checkpoint["sub_steps"] = self.sub_steps
        checkpoint["completed_steps"] = self.completed_steps
        if self.trainer.global_rank == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(self.trainer.current_epoch, self.trainer.global_step)))

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
        self.completed_steps = checkpoint["completed_steps"]
        self.sub_steps = checkpoint["sub_steps"]


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    from fengshen.utils import UniversalCheckpoint
    from fengshen.models.model_utils import add_module_args
    args_parser = add_module_args(args_parser)
    args_parser = wv2_datasets.add_data_specific_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = Wav2vec2Lightning.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser.add_argument('--ckpt_path', type=str, )
    args = args_parser.parse_args()

    module = Wav2vec2Lightning(args)
    args.sample_way = module.config.sample_way
    data_loader = prepare_data(args, module)
    data_module = UniversalDataModule(
        args=args, datasets=data_loader.datasets, tokenizer=None, collate_fn=None)

    data_module.datasets = data_loader.datasets

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'),
        name=os.path.basename(os.path.dirname(args.model_path)))
    checkpoint_callback = UniversalCheckpoint(args).callbacks

    if args.ckpt_path is not None and \
            not os.path.exists(args.ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.ckpt_path = None
    # from pytorch_lightning.profiler import PyTorchProfiler
    # schedule = torch.profiler.schedule(
    #     wait=1,
    #     warmup=2,
    #     active=4,
    #     repeat=1)
    # os.makedirs("./run/{}".format(os.environ["SLURM_JOB_ID"]), exist_ok=True)
    # profiler = PyTorchProfiler(
    #     dirpath="./run/{}".format(os.environ["SLURM_JOB_ID"]),
    #     filename="profile", schedule=schedule
    # )
    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         #  profiler=profiler,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(module, data_module, ckpt_path=args.ckpt_path)
