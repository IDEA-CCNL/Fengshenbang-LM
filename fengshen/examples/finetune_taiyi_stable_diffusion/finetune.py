import os
import torch
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from transformers import BertTokenizer, BertModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from torch.nn import functional as F
from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data


class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Taiyi Stable Diffusion Module')
        parser.add_argument('--train_whole_model', action='store_true', default=False)
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            args.model_path, subfolder="tokenizer")
        self.text_encoder = BertModel.from_pretrained(
            args.model_path, subfolder="text_encoder")  # load from taiyi_finetune-v0
        self.vae = AutoencoderKL.from_pretrained(
            args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            args.model_path, subfolder="unet")

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        self.save_hyperparameters(args)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        model_params = [{'params': self.text_encoder.parameters()}]
        if self.hparams.train_whole_model:
            model_params.append({'params': self.unet.parameters()})
        return configure_optimizers(self, model_params=model_params)

    def training_step(self, batch, batch_idx):
        self.text_encoder.train()

        with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        noise = noise.to(dtype=self.unet.dtype)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)

        # Get the text embedding for conditioning
        # with torch.no_grad():
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)

        if self.trainer.global_rank == 0:
            if (self.global_step+1) % 5000 == 0:
                print('saving model...')
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.model_path, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                )
                self.trainer.current_epoch
                pipeline.save_pretrained(os.path.join(
                    args.default_root_dir, f'hf_out_{self.trainer.current_epoch}'))

        return {"loss": loss}

    def on_train_epoch_end(self):
        if self.trainer.global_rank == 0:
            print('saving model...')
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model_path, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
            )
            self.trainer.current_epoch
            pipeline.save_pretrained(os.path.join(
                args.default_root_dir, f'hf_out_{self.trainer.current_epoch}'))

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = add_data_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = StableDiffusion.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    model = StableDiffusion(args)
    tokenizer = model.tokenizer
    datasets = load_data(args, tokenizer=tokenizer)

    def collate_fn(examples):
        # print(examples)
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True,
                                  return_tensors="pt").input_ids
        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        return batch

    datamoule = UniversalDataModule(
        tokenizer=tokenizer, collate_fn=collate_fn, args=args, datasets=datasets)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)
