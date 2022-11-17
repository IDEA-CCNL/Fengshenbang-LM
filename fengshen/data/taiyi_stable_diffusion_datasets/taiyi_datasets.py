from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


def add_data_args(parent_args):
    parser = parent_args.add_argument_group('taiyi stable diffusion data args')
    # 支持传入多个路径，分别加载
    parser.add_argument(
        "--datasets_path", type=str, default=None, required=True, nargs='+',
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--datasets_type", type=str, default=None, required=True, choices=['txt', 'csv'], nargs='+',
        help="dataset type, txt or csv, same len as datasets_path",
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", default=False,
        help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--thres", type=float, default=0.2,)
    return parent_args


class TXTDataset(Dataset):
    # 添加Txt数据集读取，主要是针对Zero23m数据集。
    def __init__(self, foloder_name, tokenizer, thres=0.2, size=512, center_crop=False):
        print(f'Loading folder data from {foloder_name}.')
        self.image_paths = []
        self.tokenizer = tokenizer
        '''
        暂时没有开源这部分文件
        score_data = pd.read_csv(os.path.join(foloder_name, 'score.csv'))
        img_path2score = {score_data['image_path'][i]: score_data['score'][i]
                          for i in range(len(score_data))}
        '''
        # print(img_path2score)
        # 这里都存的是地址，避免初始化时间过多。
        for each_file in os.listdir(foloder_name):
            if each_file.endswith('.jpg'):
                self.image_paths.append(os.path.join(foloder_name, each_file))
                # self.image_paths.append(os.path.join(foloder_name, each_file))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        print('Done loading data. Len of images:', len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        example = {}
        instance_image = Image.open(img_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        # 通过裁剪去水印！裁掉1/10的图片。
        instance_image = instance_image.crop(
            (0, 0, instance_image.size[0], instance_image.size[1] - instance_image.size[1] // 10))
        example["instance_images"] = self.image_transforms(instance_image)

        caption_path = img_path.replace('.jpg', '.txt')  # 图片名称和文本名称一致。
        with open(caption_path, 'r') as f:
            caption = f.read()
            example["instance_prompt_ids"] = self.tokenizer(
                caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        return example


# NOTE 加速读取数据，直接用原版的，在外部使用并行读取策略。30min->3min
class CSVDataset(Dataset):
    def __init__(self, input_filename, image_root, tokenizer, img_key, caption_key, thres=0.2, size=512, center_crop=False, sep="\t"):
        # logging.debug(f'Loading csv data from {input_filename}.')
        print(f'Loading csv data from {input_filename}.')
        self.images = []
        self.captions = []
        self.tokenizer = tokenizer

        if input_filename.endswith('.csv'):
            # print(f"Load Data from{input_filename}")
            df = pd.read_csv(input_filename, index_col=0)
            '''
            这个地方对数据的过滤跟数据集结构强相关，需要修改这部分的代码
            df = df[df['used'] == 1]
            df = df[df['score'] > thres]
            df = df[df['success'] == 1]
            '''
            print(f'file {input_filename} datalen {len(df)}')
            # 这个图片的路径也需要根据数据集的结构稍微做点修改
            self.images.extend(k + '/' + v for k, v in zip(df['class'], df[img_key]))
            self.captions.extend(df[caption_key].tolist())

        # NOTE 中文的tokenizer
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_root = image_root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, str(self.images[idx]))
        example = {}
        instance_image = Image.open(img_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            str(self.captions[idx]),
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example


def process_pool_read_txt_dataset(args, input_root=None, tokenizer=None, thres=0.2):
    root_path = input_root
    p = ProcessPoolExecutor(max_workers=24)
    # 此处输入为文件夹。
    all_folders = os.listdir(root_path)
    all_datasets = []
    res = []
    for filename in all_folders:
        each_folder_path = os.path.join(root_path, filename)
        res.append(p.submit(TXTDataset, each_folder_path, tokenizer,
                            thres, args.resolution, args.center_crop))
    p.shutdown()
    for future in res:
        all_datasets.append(future.result())
    dataset = ConcatDataset(all_datasets)
    return dataset


def process_pool_read_csv_dataset(args, input_root, tokenizer, thres=0.20):
    # here input_filename is a directory containing a CSV file
    all_csvs = os.listdir(os.path.join(input_root, 'release'))
    image_root = os.path.join(input_root, 'images')
    # csv_with_score = [each for each in all_csvs if 'score' in each]
    all_datasets = []
    res = []
    p = ProcessPoolExecutor(max_workers=24)
    for path in all_csvs:
        each_csv_path = os.path.join(input_root, 'release', path)
        res.append(p.submit(CSVDataset, each_csv_path, image_root, tokenizer, img_key="name",
                            caption_key="caption", thres=thres, size=args.resolution, center_crop=args.center_crop))
    p.shutdown()
    for future in res:
        all_datasets.append(future.result())
    dataset = ConcatDataset(all_datasets)
    return dataset


def load_data(args, tokenizer):
    assert len(args.datasets_path) == len(args.datasets_type), \
        "datasets_path num not equal to datasets_type"
    all_datasets = []
    for path, type in zip(args.datasets_path, args.datasets_type):
        if type == 'txt':
            all_datasets.append(process_pool_read_txt_dataset(
                args, input_root=path, tokenizer=tokenizer, thres=args.thres))
        elif type == 'csv':
            all_datasets.append(process_pool_read_csv_dataset(
                args, input_root=path, tokenizer=tokenizer, thres=args.thres))
        else:
            raise ValueError('unsupport dataset type: %s' % type)
    return {'train': ConcatDataset(all_datasets)}
