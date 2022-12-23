'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   relative_pos.py
@Time    :   2022/12/17 18:23
@Author  :   Qi Yang
@Version :   1.0
@Contact :   yangqi@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''
import math
import torch
from torch import nn
from inspect import isfunction
from einops import rearrange, repeat


def exists(val):
    return val is not None


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, learned_freq=False):
        super().__init__()
        d = head_dim
        theta = []
        for i in range(1, d//2 + 1):
            theta.append(10000**-2*(i-1)/d)
        rotary_matrix = torch.tensor(theta)

        if learned_freq:
            self.freqs = nn.Parameter(rotary_matrix)
        else:
            self.register_buffer('freqs', rotary_matrix)

        self.cache = dict()

    def rotate_queries_or_keys(self, x, seq_dim=-2):
        seq_len = x.shape[seq_dim]
        x_ = torch.arange(seq_len).to(x.device)
        freqs = self.forward(x_, cache_key=seq_len)
        return self._apply_rotary_emb(freqs, x)

    @staticmethod
    def _rotate_half(x):
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')

    def _apply_rotary_emb(self, freqs, t, start_index=0):
        freqs = freqs.to(t)
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * freqs.cos()) + (self._rotate_half(t) * freqs.sin())
        return torch.cat((t_left, t, t_right), dim=-1)

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs


class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal=False,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal=True,
        num_buckets=32,
        max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale
