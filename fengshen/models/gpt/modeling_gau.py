'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   modeling_gau.py
@Time    :   2022/12/17 18:23
@Author  :   Qi Yang
@Version :   1.0
@Contact :   yangqi@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''
from fengshen.models.gpt.relative_pos import T5RelativePositionBias
import torch
from torch import nn
from typing import Optional, Tuple, Union
import sys
sys.path.append("../../../")


class ScaleOffset(nn.Module):
    """
    This is the scale offset layer (x = x * gamma + beta).
    Args:
        dim: int
            The last dimention of input x.
    Example:
        out = tensor([[[[10.1000, 20.1000]],
                    [[30.1000, 40.1000]]]])
        x = torch.randn(1, 4, 2) # heads, seq_length, qk_dim
        scaleoff = ScaleOffset(dim=2,heads=1) # qk_dim, heads
        scaleoff(x)
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = x * self.gamma + self.beta
        return out


class GatedAttentionUnit(nn.Module):
    """
    This is a class of Gate Attention Unit (Flash-quad)
    paper: https://readpaper.com/paper/4594890495981789185
    Args:
        config (GPT2Config): special config of gpt2 models with use_gau=True
        dropout (float): Dropout prob
        x (tensor,dtype=float): (heads, seq_length, qk_dim)
    Example:
        x = torch.randint(0,100,(1, 4, 2), dtype=torch.float) #heads, seq_length, qk_dim
        x = gau.forward(x)
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.max_positions = config.n_positions
        self.register_buffer(
            "causal_mask",
            torch.ones((self.max_positions, self.max_positions), dtype=torch.bool).triu(1)  # seq_len=4
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.n_embd
        # self.head_dim = config.n_embd // config.n_head # ? qk_dim = 128 default
        self.head_dim = 128
        self.inner_dim = config.n_inner  # e = int(d ∗ expansion_factor)
        self.num_heads = config.n_head

        if self.inner_dim <= self.embed_dim:
            raise ValueError(
                f"`n_inner` must be int(n_embd ∗ expansion_factor) (got `n_embd`: {self.n_embd} and `n_inner`:"
                f" {self.inner_dim})."
            )
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        # hidden dim usually 2 x dim
        # init module
        self.q_scaleoff = ScaleOffset(dim=self.head_dim)
        self.k_scaleoff = ScaleOffset(dim=self.head_dim)  # qk_dim/head_dim

        # input_layer = nn.Linear(2, 4) # dim, hidden_dim * 2   the input x to v & gate
        # in flash code they use only 1 layer to generate v and gate and chunk then
        # we use v_layer and gate layer seperately
        self.v_layer = nn.Sequential(nn.Linear(self.embed_dim, self.inner_dim), nn.SiLU())  # dim , dim * expansion factor
        self.gate_layer = nn.Sequential(nn.Linear(self.embed_dim, self.inner_dim), nn.SiLU())
        self.input_layer = nn.Sequential(nn.Linear(self.embed_dim, self.head_dim), nn.SiLU())
        # to_hidden_layer = nn.Linear(2,4) split

        # similarly, flash code calculate q k together(heads=2) and unbind
        # we use two layer seperately and remove unbind
        self.qk_layer = nn.Sequential(nn.Linear(self.embed_dim, self.head_dim), nn.SiLU())  # dim, qk_dim
        self.output_layer = nn.Linear(self.inner_dim, self.embed_dim)  # hidden_dim , dim     the concat result to output
        self.attn_fn = nn.functional.relu  # activation funcation
        self.rel_pos_bias = T5RelativePositionBias(self.head_dim**0.5, causal=True)

        # self.norm = nn.LayerNorm(dim) # dim # move to Block
        self.dropout = nn.Dropout(config.embd_pdrop)  # dropout weight

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, causal=True):
        attn = torch.einsum('...ns,...ms ->...nm', q, k)
        # print("qk",qk, qk.shape)

        # scale
        if self.scale_attn_weights:
            attn = attn / torch.full(
                [], v.size(-1) ** 0.5, dtype=attn.dtype, device=attn.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn = attn / float(self.layer_idx + 1)

        # option
        attn = attn + self.rel_pos_bias(attn)  # qk

        # causal attention mask
        if causal:
            attn = attn.masked_fill(self.causal_mask, 0.)
            print("causal_mask", self.causal_mask.shape)
            print("attn", attn.shape)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = self.attn_fn(attn) ** 2
        attn = self.dropout(attn)
        print("attn", attn, attn.shape)

        if head_mask is not None:
            attn = attn * head_mask

        o = torch.einsum('...nm,...me -> ...ne', attn, v)  # o
        return o, attn

    def forward(self,
                x: Optional[Tuple[torch.FloatTensor]],
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = True,
                causal: Optional[bool] = True,
                add_residual: Optional[bool] = False,
                ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        v = self.v_layer(x)
        gate = self.gate_layer(x)
        # same as v, gate = to_hidden_layer(x).chunk(2,dim=-1)
        x = self.input_layer(x)

        q, k = self.q_scaleoff(x), self.k_scaleoff(x)

        o, attn = self._attn(q, k, v, attention_mask, head_mask, causal)

        o = o * gate  # o
        o = self.output_layer(o)  # o
        if add_residual:
            o = o + x

        print("o", o.shape)
        if use_cache:
            present = (k, v)
        else:
            present = None

        o = (o, present)
        if output_attentions:
            o += (attn,)

        return o  # output, (k,v), attention


class MixedChunkAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.max_positions = config.n_positions
        self.register_buffer(
            "causal_mask",
            torch.ones((self.max_positions, self.max_positions), dtype=torch.bool).triu(1)  # seq_len=4
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.n_embd
        # self.head_dim = config.n_embd // config.n_head # ? qk_dim = 128 default
        self.head_dim = 128
        self.inner_dim = config.n_inner  # e = int(d ∗ expansion_factor)
        self.num_heads = config.n_head

        if self.inner_dim <= self.embed_dim:
            raise ValueError(
                f"`n_inner` must be int(n_embd ∗ expansion_factor) (got `n_embd`: {self.n_embd} and `n_inner`:"
                f" {self.inner_dim})."
            )
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        # hidden dim usually 2 x dim
        # init module
        self.q_scaleoff = ScaleOffset(dim=self.head_dim)
        self.k_scaleoff = ScaleOffset(dim=self.head_dim)  # qk_dim/head_dim

        # input_layer = nn.Linear(2, 4) # dim, hidden_dim * 2   the input x to v & gate
        # in flash code they use only 1 layer to generate v and gate and chunk then
        # we use v_layer and gate layer seperately
        self.v_layer = nn.Sequential(nn.Linear(self.embed_dim, self.inner_dim), nn.SiLU())  # dim , dim * expansion factor
        self.gate_layer = nn.Sequential(nn.Linear(self.embed_dim, self.inner_dim), nn.SiLU())
        self.input_layer = nn.Sequential(nn.Linear(self.embed_dim, self.head_dim), nn.SiLU())
        # to_hidden_layer = nn.Linear(2,4) split

        # similarly, flash code calculate q k together(heads=2) and unbind
        # we use two layer seperately and remove unbind
        self.qk_layer = nn.Sequential(nn.Linear(self.embed_dim, self.head_dim), nn.SiLU())  # dim, qk_dim
        self.output_layer = nn.Linear(self.inner_dim, self.embed_dim)  # hidden_dim , dim     the concat result to output
        self.attn_fn = nn.functional.relu  # activation funcation
        self.rel_pos_bias = T5RelativePositionBias(self.head_dim**0.5, causal=True)

        # self.norm = nn.LayerNorm(dim) # dim # move to Block
        self.dropout = nn.Dropout(config.embd_pdrop)  # dropout weight

    def _local_quadratic_attn(self, q, k, v, attention_mask, head_mask, causal=True):
        """the normal attn in o(n^2) complexity"""
        attn = torch.einsum('...ns,...ms ->...nm', q, k)

        # scale
        if self.scale_attn_weights:
            attn = attn / torch.full(
                [], v.size(-1) ** 0.5, dtype=attn.dtype, device=attn.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn = attn / float(self.layer_idx + 1)

        # relative position bias
        attn = attn + self.rel_pos_bias(attn)  # qk

        # causal attention mask
        if causal:
            attn = attn.masked_fill(self.causal_mask, 0.)
            print("causal_mask", self.causal_mask.shape)
            print("attn", attn.shape)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = self.attn_fn(attn) ** 2
        attn = self.dropout(attn)
        print("attn", attn, attn.shape)

        if head_mask is not None:
            attn = attn * head_mask

        o = torch.einsum('...nm,...me -> ...ne', attn, v)  # o
        return o, attn

    def _global_linear_attn(self, q, k, v, causal=True):
        if causal:
            attn = torch.einsum('...cs,...ce -> ...se', k, v)
            attn = torch.cumsum(attn, dim=1) - attn  # tf.cumsum with exclusive=True
            # todo: segment id
            o = torch.einsum('...cs,...se -> ...ce', q, attn)
            return o, attn
        else:
            attn = torch.einsum('...cs,...ce -> ...bse', k, v)
            o = torch.einsum('...gcs,...se -> ...gce', q, attn)
            return o, attn

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, causal=True):
        v_quad, attn_quad = self._local_quadratic_attn(q, k, v, attention_mask, head_mask, causal)
        v_lin, attn_lin = self._global_linear_attn(q, k, v, causal)
        return v_quad+v_lin, (attn_quad, attn_lin)

    def forward(self,
                x: Optional[Tuple[torch.FloatTensor]],
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = True,
                causal: Optional[bool] = True,
                add_residual: Optional[bool] = False,
                ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        v = self.v_layer(x)
        gate = self.gate_layer(x)
        # same as v, gate = to_hidden_layer(x).chunk(2,dim=-1)
        x = self.input_layer(x)

        q, k = self.q_scaleoff(x), self.k_scaleoff(x)

        o, attn = self._attn(q, k, v, attention_mask, head_mask, causal)

        o = o * gate  # o

        o = self.output_layer(o)  # o
        if add_residual:
            o = o + x

        print("o", o.shape)
        if use_cache:
            present = (k, v)
        else:
            present = None

        o = (o, present)
        if output_attentions:
            o += attn

        return o  # output, (k,v), attention


class GAUBlock(nn.Module):
    """The is a class of GAU(FLASH-QUAD) + LayerNorm Layer
    paper: https://readpaper.com/paper/4594890495981789185
    Args:
        config (GPT2Config): config with use_gau=True
        layer_idx (int): the index of layer (2 x Attn)
    Examples:
        gau = GAUBlock(config)
        x = torch.randint(low,high,(config.n_head, config.n_positions, config.n_embd),dtype=torch.float) # heads, seq_length, qk_dim
        x = gau.forward(x)
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gau_attn = GatedAttentionUnit(
            config=config,
            layer_idx=layer_idx,
        )
        # config.n_layer should x 2 to aligned with Attn+FFN in one Block

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        hidden_states = self.ln(hidden_states)
        attn_outputs = self.gau_attn(
            x=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            causal=True,
            add_residual=True,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        if use_cache:
            outputs = (attn_output,) + outputs
        else:
            outputs = (attn_output,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
