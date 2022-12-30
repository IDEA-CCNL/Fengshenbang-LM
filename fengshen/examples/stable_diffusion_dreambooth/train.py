# -*- encoding: utf-8 -*-
'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   train.py
@Time    :   2022/11/09 22:27
@Author  :   Gan Ruyi
@Version :   1.0
@Contact :   ganruyi@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''
import hashlib
import itertools
import os
from pathlib import Path
from tqdm.auto import tqdm
import torch
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from transformers import BertTokenizer, BertModel, CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from torch.nn import functional as F
from fengshen.data.dreambooth_datasets.dreambooth_datasets import PromptDataset, DreamBoothDataset
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.data.dreambooth_datasets.dreambooth_datasets import add_data_args


class StableDiffusionDreamBooth(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Taiyi Stable Diffusion Module')
        parser.add_argument('--train_text_encoder', action='store_true', default=False)
        # dreambooth train unet only default
        parser.add_argument('--train_unet', action='store_true', default=True)
        return parent_parser

    def __init__(self, args):
        super().__init__()
        if 'Taiyi-Stable-Diffusion-1B-Chinese-v0.1' in args.model_path:
            self.tokenizer = BertTokenizer.from_pretrained(
                args.model_path, subfolder="tokenizer")
            self.text_encoder = BertModel.from_pretrained(
                args.model_path, subfolder="text_encoder")  # load from taiyi_finetune-v0
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                args.model_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(
                args.model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(
            args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            args.model_path, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_config(
            args.model_path, subfolder="scheduler")

        # set model
        self.vae.requires_grad_(False)
        if not args.train_text_encoder:
            self.requires_grad_(False)
        if not args.train_unet:
            self.requires_grad_(False)

        self.save_hyperparameters(args)

    def generate_extra_data(self):
        global_rank = self.global_rank
        device = self.trainer.device_ids[global_rank]
        print('generate on device {} of global_rank {}'.format(device, global_rank))
        class_images_dir = Path(self.hparams.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < self.hparams.num_class_images:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.hparams.model_path,
                safety_checker=None,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = self.hparams.num_class_images - cur_class_images
            print(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(self.hparams.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.hparams.sample_batch_size)

            pipeline.to(device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=global_rank != 0
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

    def setup(self, stage) -> None:
        if self.hparams.with_prior_preservation:
            self.generate_extra_data()
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        model_params = []
        if self.hparams.train_unet and self.hparams.train_text_encoder:
            model_params = itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
        elif self.hparams.train_unet:
            model_params = self.unet.parameters()
        elif self.hparams.train_text_encoder:
            model_params = self.text_encoder.parameters()
        return configure_optimizers(self, model_params=model_params)

    def training_step(self, batch, batch_idx):
        if self.hparams.train_text_encoder:
            self.text_encoder.train()
        if self.hparams.train_unet:
            self.unet.train()

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

        if self.hparams.with_prior_preservation:
            # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
            # Compute instance loss
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            # Compute prior loss
            prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="mean")
            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)

        if self.trainer.global_rank == 0:
            if (self.global_step+1) % 5000 == 0:
                print('saving model...')
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.model_path, unet=self.unet, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                )
                pipeline.save_pretrained(os.path.join(
                    args.default_root_dir, f'hf_out_{self.trainer.current_epoch}'))

        return {"loss": loss}

    def on_train_end(self) -> None:
        if self.trainer.global_rank == 0:
            print('saving model...')
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model_path, unet=self.unet, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
            )
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
    args_parser = StableDiffusionDreamBooth.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    model = StableDiffusionDreamBooth(args)

    tokenizer = model.tokenizer
    datasets = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        class_data_dir=args.class_data_dir,
        class_prompt=args.class_prompt,
        size=512,
        center_crop=args.center_crop,
    )
    # construct the datasets to a dict for universal_datamodule
    datasets = {'train': datasets}

    def collate_fn(examples):
        # print(examples)
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        return batch

    datamodule = UniversalDataModule(
        tokenizer=tokenizer, collate_fn=collate_fn, args=args, datasets=datasets)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, datamodule, ckpt_path=args.load_ckpt_path)
