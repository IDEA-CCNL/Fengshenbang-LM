# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch.nn import functional as F


class FocalLoss(torch.nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss

# 交叉熵平滑滤波 防止过拟合


class LabelSmoothingCorrectionCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCorrectionCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        # task specific
        labels_hat = torch.argmax(output, dim=1)
        lt_sum = labels_hat + target
        abs_lt_sub = abs(labels_hat - target)
        correction_loss = 0
        for i in range(c):
            if lt_sum[i] == 0:
                pass
            elif lt_sum[i] == 1:
                if abs_lt_sub[i] == 1:
                    pass
                else:
                    correction_loss -= self.eps*(0.5945275813408382)
            else:
                correction_loss += self.eps*(1/0.32447699714575207)
        correction_loss /= c
        # print(correction_loss)
        return loss*self.eps/c + (1-self.eps) * \
            F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index) + correction_loss
