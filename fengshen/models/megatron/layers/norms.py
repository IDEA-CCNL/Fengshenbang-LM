# Copyright (c) 2021, EleutherAI
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

import torch
from torch.nn import LayerNorm as LayerNorm
from torch import nn


def get_norm(config):
    if config.norm == "rmsnorm":
        norm = RMSNorm
        eps = config.rms_norm_epsilon
    elif config.norm == "layernorm":
        eps = config.layer_norm_eps
        norm = LayerNorm
    elif config.norm == "scalenorm":
        eps = config.scalenorm_epsilon
        norm = ScaleNorm
    else:
        raise ValueError(f"norm {config.norm} not recognized")
    return norm, eps


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.scale.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.scale.dtype)

        return self.scale * hidden_states


class ScaleNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g
