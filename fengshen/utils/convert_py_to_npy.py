import argparse
import torch
import glob
import os
import numpy as np


class MMapIndexDataset():
    def __init__(self, datapath):
        self.idxfp = np.load(datapath + '.npy', mmap_mode='r')
        self.binfp = np.memmap(datapath + '.bin', dtype='long', mode='r')

    def __len__(self):
        return self.idxfp.shape[0]

    def __getitem__(self, idx):
        return self.binfp[self.idxfp[idx, 0]:self.idxfp[idx, 1]]


def convert_py_to_npy(input_tensor, bin_out, idx_out):
    idx = torch.empty(len(input_tensor), 2, dtype=torch.long)
    start = 0
    for i, input in enumerate(input_tensor):
        idx[i] = torch.tensor([start, start + len(input)])
        start += len(input)
    np.save(idx_out, idx)
    binfp = np.memmap(bin_out, dtype='long', mode='w+', shape=(start))
    start = 0
    for i, input in enumerate(input_tensor):
        for j, idx in enumerate(input):
            binfp[start + j] = idx
        start += len(input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text infilling.")
    parser.add_argument('--data_path', type=str,
                        default='/cognitive_comp/gaoxinyu/data/wudao')
    args = parser.parse_args()
    process_key = [
        'incorrect_input_ids_list',
        'label_ids_list',
        'target_ids_list',
    ]
    if os.path.exists(args.data_path):
        print(f'''Loading data from {args.data_path}''')
        data_dict = torch.load(args.data_path)
        for k in process_key:
            bin_out = ('_' + k + '.bin').join(args.data_path.rsplit('.pt', 1))
            idx_out = ('_' + k).join(args.data_path.rsplit('.pt', 1))
            convert_py_to_npy(data_dict[k], bin_out, idx_out)
    else:
        print(
            f'Please create the synthetic datafile {args.data_path} with create_synthetic_data.py.')
