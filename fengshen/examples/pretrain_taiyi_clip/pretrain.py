from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from transformers import (
    CLIPModel,
    AutoModel,
    AutoTokenizer,
)
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
import argparse
import math
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
import os
from PIL import Image
from torch.utils.data._utils.collate import default_collate

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def DataFrameFilter(dataframe):
    '''
    类似这些过滤条件，可以写在这个函数里面
    # dataframe = dataframe[dataframe['success'] == 1]
    '''
    dataframe = dataframe[dataframe['used'] == 1]
    thres = 0.2  # NOTE 只读thres够大的数据
    if thres:
        # CLIP相似度分数。
        if 'score' in dataframe.columns:
            dataframe = dataframe[dataframe['score'] > thres]
    return dataframe


class ResizeMaxSize(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w -
                                      pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


class SingleDataProcessor:
    def __init__(self, args, tokenizer, is_train):
        self.image_transforms = self.image_transform(image_size=args.resolution,
                                                     is_train=is_train,
                                                     mean=None,
                                                     std=None)
        self.tokenizer = tokenizer

    def image_transform(
        self,
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
    ):
        mean = mean or OPENAI_DATASET_MEAN
        if not isinstance(mean, (list, tuple)):
            mean = (mean,) * 3

        std = std or OPENAI_DATASET_STD
        if not isinstance(std, (list, tuple)):
            std = (std,) * 3

        if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
            # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
            image_size = image_size[0]

        normalize = Normalize(mean=mean, std=std)
        if is_train:
            return Compose([
                RandomResizedCrop(image_size, scale=(0.9, 1.0),
                                  interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                normalize,
            ])
        else:
            if resize_longest_max:
                transforms = [
                    ResizeMaxSize(image_size, fill=fill_color)
                ]
            else:
                transforms = [
                    Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_size),
                ]
            transforms.extend([
                ToTensor(),
                normalize,
            ])
            return Compose(transforms)

    def __call__(self, image_path, text):
        instance_image = Image.open(image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        image = self.image_transforms(instance_image)
        text = self.tokenizer(text,
                              max_length=77,
                              padding='max_length',
                              truncation=True,
                              return_tensors='pt')['input_ids'][0]
        return image, text


class TaiyiCLIP(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Taiyi CLIP')
        parser.add_argument('--loss_type', choices=['local', 'global'], default='local')
        parser.add_argument('--gather_with_grad', default=False, action='store_true')
        parser.add_argument('--freeze_image_tower', default=False, action='store_true')
        return parent_parser

    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)

        # 这里本来打算直接用CLIPVisionModel，可惜CLIPVisionModel没有带一层project_layer转换维度，feature_dim不一样
        # self.vision_model = CLIPVisionModel.from_pretrained(args.model_path, subfolder='vision_encoder')
        vision_model = CLIPModel.from_pretrained(
            args.model_path, subfolder='vision_encoder')
        # 这里没有用到CLIPModel的TextModel，删掉这部分
        del vision_model.text_model
        text_model = AutoModel.from_pretrained(args.model_path, subfolder='text_encoder')
        # 需要手动更改clip里面的proj层
        vision_model.text_projection = nn.Linear(
            text_model.config.hidden_size, vision_model.config.projection_dim, bias=False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, subfolder='text_encoder')
        self.text_model = text_model
        self.vision_model = vision_model

        # from flagai.model.mm.AltCLIP import CLIPHF
        # self.vision_model = CLIPHF.from_pretrained(
        #     '/cognitive_comp/gaoxinyu/huggingface_model/AltCLIP')
        # self.text_model = self.vision_model.text_model
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     '/cognitive_comp/gaoxinyu/huggingface_model/AltCLIP')

        self.local_loss = args.loss_type == 'local'

        if args.freeze_image_tower:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.visual_projection.requires_grad = False

        # cache
        self.cache_labels = True
        self.prev_num_logits = 0
        self.labels = {}

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))
        elif stage == 'validate':
            self.total_steps = 100

    def configure_optimizers(self):
        return configure_optimizers(self)

    def forward(self, image, text):
        assert image is not None
        assert text is not None
        image_features = self.vision_model.get_image_features(image)
        # text_features = self.vision_model.get_text_features(text)
        text_outputs = self.text_model(input_ids=text)
        pooled_output = text_outputs[1]
        text_features = self.vision_model.text_projection(pooled_output)
        # text_features = self.text_model(input_ids=text).logits

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return image_features, text_features, self.vision_model.logit_scale.exp()

    def gather_features(self, image_features, text_features):
        all_image_features = self.all_gather(
            image_features, sync_grads=self.hparams.gather_with_grad)
        all_text_features = self.all_gather(
            text_features, sync_grads=self.hparams.gather_with_grad)
        if not self.local_loss and not self.gather_with_grad:
            # 如果是全局loss，并且不需要梯度，需要把梯度更新回tensor
            all_image_features[self.global_rank] = image_features
            all_text_features[self.global_rank] = text_features
        all_image_features = all_image_features.view(-1, all_image_features.shape[-1])
        all_text_features = all_text_features.view(-1, all_text_features.shape[-1])
        return all_image_features, all_text_features

    def clip_loss(self, image_features, text_features, logit_scale):
        if self.trainer.world_size > 1:
            all_image_features, all_text_features = self.gather_features(
                image_features, text_features)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or self.device not in self.labels:
            labels = torch.arange(num_logits, device=self.device, dtype=torch.long)
            if self.trainer.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.global_rank
            if self.cache_labels:
                self.labels[self.device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[self.device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss

    def training_step(self, batch):
        image, text = batch
        image_features, text_features, logit_scale = self(image, text)
        total_loss = self.clip_loss(image_features, text_features, logit_scale)
        self.log('train_loss', total_loss, sync_dist=True)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        with torch.no_grad():
            self.vision_model.logit_scale.clamp_(0, math.log(100))

    def get_metrics(self, image_features, text_features, labels, logit_scale):
        # 计算相似度，支持多个样本的情况（比如一个图片有多个caption）
        # img2txt计算的时候要用到，因为一张图片可能对应多个文本。
        # txt2img计算的时候不需要（一般一个text只有一个对应图片）
        metrics = {}
        logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
        logits_per_text = logits_per_image.t().detach().cpu()

        logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}

        label2idx = {}  # 计算label到idx的映射。
        repeat_id = []
        for i, label in enumerate(labels):
            if label not in label2idx:
                label2idx[label] = [i]
            else:
                # 表示该index的标签出现过，记录这个index，后续算txt2img分数的时候，这些index的权值要降低。
                label2idx[label].append(i)
                repeat_id.append(i)

        ground_truth = [label2idx[label] for label in labels]

        for name, logit in logits.items():
            if name == 'text_to_image':
                logit[:, repeat_id] -= 1e8   # 这部分的分数要降低。（重复出现的图片，直接忽略）
            r_stat = {1: [], 5: [], 10: []}
            # r1_stat, r5_stat, r10_stat = [], [], []
            # index of the largest element to the smallest
            ranking = torch.argsort(logit, descending=True)
            for i, each_query in enumerate(ranking[:, :10]):
                for j, q in enumerate(each_query):
                    found = False
                    if q in ground_truth[i]:
                        for k, v in r_stat.items():
                            if j < k:
                                found = True
                                v.append(1)
                    if found:
                        break
            for k, v in r_stat.items():
                metrics[f'{name}_R@{k}'] = sum(v)/len(logit)
        return metrics

    def validation_step(self, batch, batch_idx):
        image, text, label = batch
        image_features, text_features, logit_scale = self(image, text)
        return image_features, text_features, logit_scale, image.shape[0], label

    def validation_epoch_end(self, val_outputs):
        all_image_features = []
        all_text_features = []
        all_labels = []
        sample_size = 0
        for o in val_outputs:
            all_image_features.append(o[0])
            all_text_features.append(o[1])
            sample_size += o[3]
            all_labels += o[4]
        all_image_features = torch.cat(all_image_features)
        all_text_features = torch.cat(all_text_features)
        logit_scale = val_outputs[0][2].mean()
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(sample_size, device=self.device).long()
        total_loss = (F.cross_entropy(logits_per_image, labels)
                      + F.cross_entropy(logits_per_text, labels)) / 2

        val_metrics = self.get_metrics(
            image_features=all_image_features,
            text_features=all_text_features,
            logit_scale=logit_scale,
            labels=all_labels)
        loss = total_loss / sample_size
        self.log('val_loss', loss, sync_dist=True)
        for k, v in val_metrics.items():
            self.log(f'val_{k}', v, sync_dist=True)

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
            self.vision_model.save_pretrained(os.path.join(dir_path, 'vision_encoder'))
            self.text_model.save_pretrained(os.path.join(dir_path, 'text_encoder'))
            self.tokenizer.save_pretrained(os.path.join(dir_path, 'text_encoder'))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = add_data_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = TaiyiCLIP.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    model = TaiyiCLIP(args)
    tokenizer = model.tokenizer
    data_process = SingleDataProcessor(args, tokenizer, is_train=True)
    datasets = load_data(args, data_filter_fn=DataFrameFilter,
                         data_process_fn=data_process, global_rank=trainer.global_rank)

    # 加载单个验证集：！！！验证代码有效性临时这样干的，验证完有效性会删除
    from fengshen.examples.pretrain_taiyi_clip.flickr_datasets import flickr30k_CNA
    img_root = '/shared_space/ccnl/mm_data/Flickr30k-CNA/flickr30k/images'
    text_annot_path = '/shared_space/ccnl/mm_data/Flickr30k-CNA/test/flickr30k_cn_test.txt'
    val_data_process = SingleDataProcessor(args, tokenizer, is_train=False)
    datasets[args.val_datasets_field] = flickr30k_CNA(img_root, text_annot_path, val_data_process)

    def train_collator(inputs):
        samples = []
        for i in inputs:
            image, captions = data_process(i['img_path'], i['caption'])
            samples.append((image, captions))
        return default_collate(samples)

    datamoule = UniversalDataModule(
        tokenizer=tokenizer, collate_fn=train_collator, args=args, datasets=datasets)

    trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)
