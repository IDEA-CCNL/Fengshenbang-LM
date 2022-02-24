import numpy as np
import torch
from typing import List
from torch.utils.data import Dataset


class MMapIndexDataset(Dataset):
    # datapaths 是所有的内存映射文件的路径
    # input_tensor_name 是输入的tensor的名字 例如 ['input_ids'] 会存储在对应的文件里面
    def __init__(self, datapaths: List[str], input_tensor_name: List[str]):
        dict_idx_fp = {}
        dict_bin_fp = {}
        idx_len = []
        for tensor_name in input_tensor_name:
            idx_fp = []
            bin_fp = []
            len = 0
            for data_path in datapaths:
                idx_fp += [np.load(
                    data_path + '_' + tensor_name + '.npy', mmap_mode='r')]
                bin_fp += [np.memmap(
                    data_path + '_' + tensor_name + '.bin',
                    dtype='long',
                    mode='r')]
                len += idx_fp[-1].shape[0]
                idx_len += [idx_fp[-1].shape[0]]
            dict_idx_fp[tensor_name] = idx_fp
            dict_bin_fp[tensor_name] = bin_fp
            #  通常情况下不同的tensor的长度是一样的
            self._len = len

        self._input_tensor_name = input_tensor_name
        self._dict_idx_fp = dict_idx_fp
        self._dict_bin_fp = dict_bin_fp
        self._idx_len = idx_len

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        sample = {}
        for i in range(len(self._idx_len)):
            if idx >= self._idx_len[i]:
                idx -= self._idx_len[i]
            else:
                break
        for tensor_name in self._input_tensor_name:
            sample[tensor_name] = torch.tensor(self._dict_bin_fp[tensor_name][i][
                self._dict_idx_fp[tensor_name][i][idx, 0]:
                    self._dict_idx_fp[tensor_name][i][idx, 1]
            ], dtype=torch.long)
        # print(sample)
        return sample
