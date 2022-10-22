# coding=utf-8
# Copyright 2022 IDEA-CCNL The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch TransfoXLDenoise model. """

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli


def enforce_repetition_penalty(lprobs, prev_output_tokens, repetition_penalty=1.5):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(len(prev_output_tokens)):
        for previous_token in set(prev_output_tokens[i]):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1# batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for i in range(sorted_indices.size()[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # logits[indices_to_remove] = filter_value
    return logits


def word_drop(x, p, unk_token):
    x_ = x.detach().clone()
    mask = Bernoulli(1. - p).sample(x.shape)
    x_[mask == 0] = unk_token
    return x_


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def connect(mean, logvar, nsamples=1, sample=True, clip=False, min_clip_val=-1., beta_logvar=1.):
    """
    Returns: Tensor1, Tensor2
        Tensor1: the tensor latent z with shape [batch, nsamples, nz]
    """
    # (batch, nsamples, nz)
    if sample:
        if clip:
            # NOTE: clip the logvar here to see if we can force z to be more distant
            logvar = torch.clip(logvar, min=min_clip_val)
        z = reparameterize(mean, logvar, nsamples, beta_logvar)
    else:
        batch_size, nz = mean.size()
        z = mean.unsqueeze(1).expand(batch_size, nsamples, nz)
    if nsamples == 1:
        z = z.squeeze(dim=1)
    return z


def reparameterize(mu, logvar, nsamples=1, beta_logvar=1.):
    """sample from posterior Gaussian family
    Args:
        mu: Tensor
            Mean of gaussian distribution with shape (batch, nz)
        logvar: Tensor
            logvar of gaussian distibution with shape (batch, nz)
    Returns: Tensor
        Sampled z with shape (batch, nsamples, nz)
    """
    batch_size, nz = mu.size()
    std = logvar.mul(0.5).exp().mul(beta_logvar)

    mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
    std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

    eps = torch.zeros_like(std_expd).normal_()

    return mu_expd + torch.mul(eps, std_expd)


def compute_kl_loss(mean1, logvar1, mean2, logvar2):
    '''adapted from adaVAE implementation https://github.com/ImKeTT/adavae/blob/main/src/adapters/vae.py#L1627'''
    exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
    result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
    return result
