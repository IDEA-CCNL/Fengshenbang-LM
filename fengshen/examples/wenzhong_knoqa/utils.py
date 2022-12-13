# -*- encoding: utf-8 -*-
'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   utils.py
@Time    :   2022/10/28 18:27
@Author  :   Qi Yang
@Version :   1.0
@Contact :   yangqi@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from fengshen.data.universal_datamodule import UniversalDataModule
from torch.utils.data import DataLoader, DistributedSampler


class GPTDataModule(UniversalDataModule):
    def __init__(self, tokenizer, collate_fn, collate_fn_eval, args, datasets=None, **kwargs):
        super().__init__(tokenizer, collate_fn, args, datasets, **kwargs)
        self.collate_fn_eval = collate_fn_eval

    def val_dataloader(self):
        ds = self.datasets[self.hparams.val_datasets_field]
        collate_fn = self.collate_fn_eval
        if collate_fn is None and hasattr(ds, 'collater'):
            collate_fn = ds.collater

        return DataLoader(
            ds,
            batch_size=self.hparams.val_batchsize,
            shuffle=False,
            num_workers=self.hparams.dataloader_workers,
            collate_fn=collate_fn,
            sampler=DistributedSampler(
                ds, shuffle=False),
            pin_memory=True,
        )

        # return DataLoader(
        #     ds, shuffle=False, batch_size=self.hparams.val_batchsize, pin_memory=False, collate_fn=collate_fn,
        # )

    def test_dataloader(self):
        ds = self.datasets[self.hparams.test_datasets_field]

        collate_fn = self.collate_fn_eval
        if collate_fn is None and hasattr(ds, 'collater'):
            collate_fn = ds.collater

        return DataLoader(
            ds,
            batch_size=self.hparams.test_batchsize,
            shuffle=False,
            num_workers=self.hparams.dataloader_workers,
            collate_fn=collate_fn,
            sampler=DistributedSampler(
                ds, shuffle=False),
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


def remove_prompt(text):
    if ":" in text:
        return text.split(":")[1]
    return text
