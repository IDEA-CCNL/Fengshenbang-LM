from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
from transformers import BertTokenizer
import pytorch_lightning as pl
from PIL import Image
import os


class flickr30k_CNA(Dataset):
    def __init__(self, img_root_path,
                 annot_path,
                 transform=None):
        self.images = []
        self.captions = []
        self.labels = []
        self.root = img_root_path
        with open(annot_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                key, caption = line[0].split('#')[0], line[1]
                img_path = key + '.jpg'
                self.images.append(img_path)
                self.captions.append(caption)
                self.labels.append(key)
        self.transforms = transform
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # NOTE large 模型
        self.context_length = 77

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        image = self.transforms(Image.open(os.path.join(self.root, img_path)))
        text = self.tokenizer(str(self.captions[idx]), max_length=self.context_length,
                              padding='max_length', truncation=True, return_tensors='pt')['input_ids'][0]
        label = self.labels[idx]
        return image, text, label


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
):
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])


class FlickrDataModule(pl.LightningDataModule):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.train_filename = args.train_filename  # NOTE 标注的文件夹
        self.train_root = args.train_root  # NOTE 图片地址
        self.val_filename = args.val_filename
        self.val_root = args.val_root
        self.test_filename = args.test_filename
        self.test_root = args.test_root

        self.pretrain_model = args.pretrain_model
        self.image_size = 224
        self.prepare_data_per_node = True
        self._log_hyperparams = False
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        # dataset
        train_transform = image_transform(224, True)
        val_transform = image_transform(224, False)
        test_transform = image_transform(224, False)

        self.train_dataset = flickr30k_CNA(self.train_root, self.train_filename, transform=train_transform)
        self.val_dataset = flickr30k_CNA(self.val_root, self.val_filename, transform=val_transform)
        self.test_dataset = flickr30k_CNA(self.test_root, self.test_filename, transform=test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
