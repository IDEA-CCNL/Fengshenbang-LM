from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import (
    WandbLogger
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from fengshen.models.lyrics.modeling_lyrics import LyricsQFromerForPretrain
import fengshen.models.lyrics.groundingdino.transforms as T
from fengshen.models.lyrics.configuration_lyrics import LyricsConfig
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
import argparse
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
import numpy as np
from torchvision.transforms import Normalize, Compose, Resize, RandomResizedCrop, InterpolationMode, ToTensor, RandomHorizontalFlip
from PIL import Image
from transformers import BertTokenizer, Blip2Processor
from torch.utils.data._utils.collate import default_collate
import os
import torch

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

class TensorObject(object):
    def __init__(self, tensor: torch.Tensor):
        self.data = tensor

class Collator():
    def __init__(self, args):
        self.transforms = Blip2Processor.from_pretrained(args.model_path)
        self.grounding_transforms = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                # T.RandomResize([800]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.ram_transforms = Compose([
                    Resize((384, 384)),
                    ToTensor(), 
                    Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
                ])

    def __call__(self, inputs):
        # samples = []
        image = []
        grounding_image = []
        ram_image = []
        input_captions = []
        input_languages = []

        ran = None
        for (cnt, i) in enumerate(inputs):
            if 'npy_path' in i:
                instance_image = Image.fromarray(np.load(i['npy_path']))
            elif 'img_path' in i:
                try:
                    instance_image = Image.open(i['img_path'])
                    if not instance_image.mode == "RGB":
                        instance_image = instance_image.convert("RGB")
                except:
                    continue
            elif "image" in i and i["image"] is not None:
                instance_image = i["image"]
                if not instance_image.mode == "RGB":
                    instance_image = instance_image.convert("RGB")
            elif "img" in i and i["img"] is not None:
                instance_image = i["img"]
                if not instance_image.mode == "RGB":
                    instance_image = instance_image.convert("RGB")
            else:
                raise ValueError('no img path in samples')

            if 'blip_caption' in i:
                try:
                    loc = torch.multinomial(torch.tensor(i['blip_scores']), 1)
                    caption = i['blip_caption'][loc]
                    language = 'zh'
                except Exception:
                    caption = ''
                    print(i)
            elif 'caption' in i:
                caption = i['caption']
                language = 'en'
            elif 'caption_zh' in i:
                caption = i['caption_zh']
                language = 'zh'
            image.append(self.transforms(instance_image, return_tensors="pt")['pixel_values'][0])
            grounding_image.append(self.grounding_transforms(instance_image, None)[0])
            ram_image.append(self.ram_transforms(instance_image))
            input_captions.append(caption)
            input_languages.append(language)
        model_inputs = {
            "image": torch.stack(image),
            "grounding_image": grounding_image,
            "ram_image": torch.stack(ram_image),
            "caption": input_captions,
            "language": input_languages,
            }
        return model_inputs


class LyricsQFromer(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LyricsQFromer')
        parser.add_argument('--freeze_image_tower', default=False, action='store_true')
        return parent_parser

    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)

        # self.model = LyricsQFromerForPretrain.from_pretrained(args.model_path, ignore_mismatched_sizes=True)
        self.model = LyricsQFromerForPretrain.from_pretrained(args.model_path)
        tokenizer = BertTokenizer.from_pretrained(os.path.join(args.model_path, 'tokenizer'))
        self.model.tokenizer = tokenizer
        self.model.box_threshold = 0.25
        self.model.text_threshold = 0.2
        self.model.iou_threshold = 0.6

        if args.freeze_image_tower:
            self.model.vision_model.eval()
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            self.model.ram.eval()
            for param in self.model.ram.parameters():
                param.requires_grad = False
            self.model.grounding_dino.eval()
            for param in self.model.grounding_dino.parameters():
                param.requires_grad = False

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            self.steps_per_epoch = self.total_steps // self.trainer.max_epochs
            print('Total steps: {}' .format(self.total_steps))
        elif stage == 'validate':
            self.total_steps = 100

    def configure_optimizers(self):
        return configure_optimizers(self)

    def training_step(self, batch):
        output = self.model(**batch)
        self.log('train/loss_itc', output.loss_itc)
        self.log('train/loss_itm', output.loss_itm)
        self.log('train/loss_lm', output.loss_lm)
        self.log('train/loss_mlm', output.loss_mlm)
        self.log('train/loss', output.loss)
        if self.trainer.global_rank == 0:
            if self.trainer.global_step % 1000 == 0:
                print('loss_itc:', output.loss_itc)
                print('loss_itm:', output.loss_itm)
                print('loss_lm:', output.loss_lm)
                print('loss_mlm:', output.loss_mlm)
        return output.loss

    def validation_step(self, batch, batch_idx):
        raise Exception("not impl")

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset

    def on_save_checkpoint(self, checkpoint) -> None:
        # 保存的时候把权重按huggingface的形式保存出来
        if self.global_rank == 0:
            dir_path = os.path.join(
                self.hparams.default_root_dir, f'hf_out_{self.trainer.current_epoch}_{self.trainer.global_step}')
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            self.model.save_pretrained(dir_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = add_data_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = LyricsQFromer.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    # wandb_logger = WandbLogger(project="ditto_pretrain")  # 初始化个WandbLogger对象
    trainer = Trainer.from_argparse_args(args,
                                        #  logger=wandb_logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    model = LyricsQFromer(args)
    collate_fn = Collator(args)
    datasets = load_data(args, global_rank=trainer.global_rank)
    datamoule = UniversalDataModule(
        tokenizer=None, collate_fn=collate_fn, args=args, datasets=datasets)
    trainer.fit(model, datamoule)
    # trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)