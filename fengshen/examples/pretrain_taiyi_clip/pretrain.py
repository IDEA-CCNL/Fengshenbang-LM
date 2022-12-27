from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from fengshen.models.clip import (
    TaiyiCLIPModel,
    TaiyiCLIPProcessor,
)
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
import torch
import torch.nn.functional as F
import argparse
import math
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
import os
import numpy as np
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class Collator():
    def __init__(self, args, processor):
        self.processor = processor
        self.seq_length = args.seq_length
        self.transforms = Compose([
            ToTensor(),
            RandomResizedCrop(args.resolution, scale=(0.9, 1.0),
                              interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ])

    def __call__(self, inputs):
        max_length = min(self.seq_length, max([len(i['caption']) for i in inputs]))
        images = []
        texts = []
        labels = []
        for i in inputs:
            # instance_image = Image.open(i['img_path'])
            # instance_image = jpeg4py.JPEG(i['img_path']).decode()
            instance_image = np.load(i['npy_path'])
            images.append(self.transforms(instance_image))
            texts.append(i['caption'])
            labels.append(i['labels'] if 'labels' in i else -100)
        # images_input = self.processor(images=images, return_tensors="pt")
        texts_input = self.processor(text=texts,
                                     max_length=max_length,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors='pt')
        # return images_input, texts_input, labels
        return {'pixel_values': torch.stack(images)}, texts_input, labels


class TaiyiCLIP(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Taiyi CLIP')
        parser.add_argument('--loss_type', choices=['local', 'global'], default='local')
        parser.add_argument('--seq_length', default=77)
        parser.add_argument('--gather_with_grad', default=False, action='store_true')
        parser.add_argument('--freeze_image_tower', default=False, action='store_true')
        return parent_parser

    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)

        self.model = TaiyiCLIPModel.from_pretrained(args.model_path)
        self.processor = TaiyiCLIPProcessor.from_pretrained(args.model_path)

        self.local_loss = args.loss_type == 'local'

        if args.freeze_image_tower:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            self.model.visual_projection.requires_grad = False

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
        image_features = self.model.get_image_features(**image)
        text_features = self.model.get_text_features(**text)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return image_features, text_features, self.model.logit_scale.exp()

    def gather_features(self, features):
        if self.trainer.world_size == 1:
            return features
        all_features = self.all_gather(
            features, sync_grads=self.hparams.gather_with_grad)
        if not self.local_loss and not self.gather_with_grad:
            # 如果是全局loss，并且不需要梯度，需要把梯度更新回tensor
            all_features[self.global_rank] = features
        all_features = all_features.view(-1, all_features.shape[-1])
        return all_features

    def clip_loss(self, image_features, text_features, logit_scale):

        logits_per_image = None

        # 如果我冻住VIT并且是local_loss，那么我只需要自己的这部分text feature就行
        # 因为根本不需要image2text的feature训练VIT
        if self.hparams.freeze_image_tower and self.local_loss:
            all_text_features = None
        else:
            all_text_features = self.gather_features(
                text_features)
        all_image_features = self.gather_features(
            image_features)

        if self.local_loss:
            if all_text_features is not None:
                logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
        else:
            # 如果是global_loss，那all_text_features肯定不是空的
            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T

        num_logits = logits_per_text.shape[0]
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
        ) / 2 if logits_per_image is not None else F.cross_entropy(logits_per_text, labels)
        return total_loss

    def training_step(self, batch):
        image, text, _ = batch
        image_features, text_features, logit_scale = self(image, text)
        total_loss = self.clip_loss(image_features, text_features, logit_scale)
        self.log('train_loss', total_loss, sync_dist=False)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        with torch.no_grad():
            self.model.logit_scale.clamp_(0, math.log(100))

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
        return image_features, text_features, logit_scale, text['input_ids'].shape[0], label

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
        if len(all_image_features) == 0 or len(all_text_features) == 0:
            return
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
        self.log('val_loss', loss, sync_dist=False)
        for k, v in val_metrics.items():
            self.log(f'val_{k}', v, sync_dist=False)

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
            self.processor.save_pretrained(dir_path)


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
    processor = model.processor
    collate_fn = Collator(args, processor)
    datasets = load_data(args, global_rank=trainer.global_rank)

    # 加载单个验证集：！！！验证代码有效性临时这样干的，验证完有效性会删除
    from fengshen.examples.pretrain_taiyi_clip.flickr_datasets import flickr30k_CNA
    img_root = '/shared_space/ccnl/mm_data/Flickr30k-CNA/flickr30k/images'
    text_annot_path = '/shared_space/ccnl/mm_data/Flickr30k-CNA/test/flickr30k_cn_test.txt'

    datasets[args.val_datasets_field] = flickr30k_CNA(img_root, text_annot_path, collate_fn)

    datamoule = UniversalDataModule(
        tokenizer=None, collate_fn=collate_fn, args=args, datasets=datasets)

    trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)
