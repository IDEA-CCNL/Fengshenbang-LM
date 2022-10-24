from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import json
import random


class QGDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""

    def __init__(self, test_file, sample_num=0):
        super(QGDataset, self).__init__()
        self.test_file = test_file
        with open(test_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            datas = [json.loads(line) for line in lines]
        if sample_num == 0:
            self.samples = datas
        else:
            self.samples = datas[0:sample_num]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def test_dataloader(collate_fn, args):
    """avoid to inference should modify fsdataset and loaddataset"""
    dataset = QGDataset(args.test_file, args.sample_num)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.test_batchsize,
        shuffle=False,
        num_workers=args.dataloader_workers,
        collate_fn=collate_fn,
        # sampler=DistributedSampler(
        #    self.datasets[self.hparams.test_datasets_field], shuffle=False),
        pin_memory=True,
    )


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = -100

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        targets_ignore = torch.where(target != self.ignore_index, target, 0)
        nll_loss = -logprobs.gather(dim=-1, index=targets_ignore.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def truncate_sequence(document: str, max_num_tokens: int, reverse=False):
    total_length = len(document)
    if total_length <= max_num_tokens:
        return document
    else:
        if reverse:
            return document[-1*max_num_tokens:]
        else:
            return document[:max_num_tokens]


def padding_to_maxlength(ids, max_length, pad_id):
    cur_len = len(ids)
    len_diff = max_length - len(ids)
    return ids + [pad_id] * len_diff, [1] * cur_len + [0] * len_diff


def white_space_fix(text):
    return "".join(text.split(" "))


def remove_prompt(text, prompt):
    if prompt in text:
        return text.split(":")[1]
    return text
