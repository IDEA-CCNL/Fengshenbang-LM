# 这里这个dataset只是临时测试用的，所以暂时用最简陋的方式放在这里，后续会优化
from torch.utils.data import Dataset
from PIL import Image


class flickr30k_CNA(Dataset):
    def __init__(self, img_root_path=None,
                 text_annot_path=None,
                 data_process_fn=None):
        self.images = []
        self.captions = []
        self.labels = []
        self.root = img_root_path
        with open(text_annot_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                key, caption = line[0].split('#')[0], line[1]
                img_path = key + '.jpg'
                self.images.append(img_path)
                self.captions.append(caption)
                self.labels.append(key)
        self.data_process_fn = data_process_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.root + "/" + self.images[idx])
        instance_image = Image.open(img_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        captions = self.captions[idx]
        label = self.labels[idx]
        image, text = self.data_process_fn(instance_image, captions)
        return image, text, label
