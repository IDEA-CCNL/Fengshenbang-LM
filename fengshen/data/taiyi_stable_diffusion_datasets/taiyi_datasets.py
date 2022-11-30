from torch.utils.data import Dataset, ConcatDataset
import os
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
        "--datasets_type", type=str, default=None, required=True, choices=['txt', 'csv', 'fs_datasets'], nargs='+',
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
    parser.add_argument("--thres", type=float, default=0.2)
    return parent_args


class TextImageBaseDataset(Dataset):
    def __init__(self, data_filter_fn, data_process_fn):
        super().__init__()
        self.data_filter_fn = data_filter_fn
        self.data_process_fn = data_process_fn
        '''
        data_filter_fn: 用来过滤数据的函数 def data_filter_fn(dataframe) -> dataframe
        data_process_fn： 在__getitem__中用于数据处理的函数 def data_process_fn(image, text) -> sample
                          兼容性可能没那么好，以后再改
        '''
        # data_process_fn 一定得自己实现一个
        # assert data_process_fn is not None


class TXTDataset(TextImageBaseDataset):
    # 添加Txt数据集读取，主要是针对Zero23m数据集。
    def __init__(self,
                 foloder_name,
                 thres=0.2,
                 data_filter_fn=None,
                 data_process_fn=None):
        super().__init__(data_filter_fn=data_filter_fn, data_process_fn=data_process_fn)
        # print(f'Loading folder data from {foloder_name}.')
        self.image_paths = []
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
                # 需要把读到的列表转换成dataframe，以便支持统一的过滤函数
                # self.image_paths = self.data_filter_fn(self.image_paths)

        # print('Done loading data. Len of images:', len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        caption_path = img_path.replace('.jpg', '.txt')  # 图片名称和文本名称一致。
        with open(caption_path, 'r') as f:
            caption = f.read()
        return self.data_process_fn(img_path, caption)


# NOTE 加速读取数据，直接用原版的，在外部使用并行读取策略。30min->3min
class CSVDataset(TextImageBaseDataset):
    def __init__(self,
                 input_filename,
                 image_root,
                 img_key,
                 caption_key,
                 thres=0.2,
                 data_filter_fn=None,
                 data_process_fn=None):
        super().__init__(data_filter_fn=data_filter_fn, data_process_fn=data_process_fn)
        # logging.debug(f'Loading csv data from {input_filename}.')
        print(f'Loading csv data from {input_filename}.')
        self.images = []
        self.captions = []

        if input_filename.endswith('.csv'):
            # print(f"Load Data from{input_filename}")
            df = pd.read_csv(input_filename, index_col=0, on_bad_lines='skip')
            if self.data_filter_fn is not None:
                df = self.data_filter_fn(df)
            print(f'file {input_filename} datalen {len(df)}')
            # 这个图片的路径也需要根据数据集的结构稍微做点修改
            self.images.extend(df[img_key].tolist())
            self.captions.extend(df[caption_key].tolist())
        self.image_root = image_root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, str(self.images[idx]))
        return self.data_process_fn(img_path, self.captions[idx])


def if_final_dir(path: str) -> bool:
    # 如果当前目录有一个文件，那就算是终极目录
    for f in os.scandir(path):
        if f.is_file():
            return True
    return False


def process_pool_read_txt_dataset(args,
                                  input_root=None,
                                  thres=0.2,
                                  data_filter_fn=None,
                                  data_process_fn=None):
    p = ProcessPoolExecutor(max_workers=20)
    all_datasets = []
    res = []

    # 遍历该目录下所有的子目录
    def traversal_files(path: str):
        list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
        for dir_path in list_subfolders_with_paths:
            if if_final_dir(dir_path):
                res.append(p.submit(TXTDataset,
                                    dir_path,
                                    thres,
                                    data_filter_fn=data_filter_fn,
                                    data_process_fn=data_process_fn))
            else:
                traversal_files(dir_path)
    traversal_files(input_root)
    p.shutdown()
    for future in res:
        all_datasets.append(future.result())
    dataset = ConcatDataset(all_datasets)
    return dataset


def process_pool_read_csv_dataset(args,
                                  input_root,
                                  thres=0.20,
                                  data_filter_fn=None,
                                  data_process_fn=None):
    # here input_filename is a directory containing a CSV file
    all_csvs = os.listdir(os.path.join(input_root, 'release'))
    image_root = os.path.join(input_root, 'images')
    # csv_with_score = [each for each in all_csvs if 'score' in each]
    all_datasets = []
    res = []
    p = ProcessPoolExecutor(max_workers=150)
    for path in all_csvs:
        each_csv_path = os.path.join(input_root, 'release', path)
        res.append(p.submit(CSVDataset,
                            each_csv_path,
                            image_root,
                            img_key="name",
                            caption_key="caption",
                            thres=thres,
                            data_filter_fn=data_filter_fn,
                            data_process_fn=data_process_fn))
    p.shutdown()
    for future in res:
        all_datasets.append(future.result())
    dataset = ConcatDataset(all_datasets)
    return dataset


def load_data(args, data_filter_fn=None, data_process_fn=None, global_rank=0):
    assert len(args.datasets_path) == len(args.datasets_type), \
        "datasets_path num not equal to datasets_type"
    all_datasets = []
    for path, type in zip(args.datasets_path, args.datasets_type):
        if type == 'txt':
            all_datasets.append(process_pool_read_txt_dataset(
                args, input_root=path, thres=args.thres,
                data_filter_fn=data_filter_fn,
                data_process_fn=data_process_fn))
        elif type == 'csv':
            all_datasets.append(process_pool_read_csv_dataset(
                args, input_root=path, thres=args.thres,
                data_filter_fn=data_filter_fn,
                data_process_fn=data_process_fn))
        elif type == 'fs_datasets':
            from fengshen.data.fs_datasets import load_dataset
            all_datasets.append(load_dataset(path, num_proc=args.num_workers,
                                             thres=args.thres, global_rank=global_rank)['train'])
        else:
            raise ValueError('unsupport dataset type: %s' % type)
        print(f'load datasset {type} {path} len {len(all_datasets[-1])}')
    return {'train': ConcatDataset(all_datasets)}
