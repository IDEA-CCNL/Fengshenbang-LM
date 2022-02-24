# coding=utf-8
import torch
from torch import nn


class metrics_mlm_acc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, masked_lm_metric):

        # if len(list(logits.shape))==3:
        mask_label_size = 0
        for i in masked_lm_metric:
            for j in i:
                if j > 0:
                    mask_label_size += 1

        y_pred = torch.argmax(logits, dim=-1)

        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,))
        masked_lm_metric = masked_lm_metric.view(size=(-1,))

        corr = torch.eq(y_pred, y_true)
        corr = torch.multiply(masked_lm_metric, corr)

        acc = torch.sum(corr.float())/mask_label_size
        return acc
