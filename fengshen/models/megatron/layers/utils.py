# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for models."""

import torch
from .norms import LayerNorm, RMSNorm, ScaleNorm
from .fused_softmax import SoftmaxFusionTypes
from types import GeneratorType


def get_attn_mask(seq_length, device):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )

    # convert to binary
    return mask < 0.5

def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    eod_mask_loss=False,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    attention_mask = get_attn_mask(
        seq_length=seq_length,
        device=data.device,
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids

def exists(x):
    return x is not None


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class SequentialWrapper(torch.nn.Module):
    """
    Used to convert a deepspeed PipelineModule to an nn.Sequential like model whilst retaining
    activation checkpointing.
    """

    def __init__(
        self,
        layers,
        activation_checkpoint_interval,
        activation_checkpoint_func,
        parent_class_name=None,
    ):
        super().__init__()
        self.sequential = torch.nn.Sequential(*layers)
        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.parent_class_name = parent_class_name
        self.activation_checkpoint_func = activation_checkpoint_func

    def _is_checkpointable(self, funcs):
        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)

    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.

        :param cache: (bool) True if you want to use caching during inference, False otherwise
        """
        _set_use_cache(self.sequential, use_cache)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching.
        """
        _set_use_cache(self.sequential, False)

    def forward(self, forward_input):
        def exec_range_func(start, end):
            """Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            """

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.sequential[start:end]):
                    inputs = layer(inputs)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.sequential))
            x = func(forward_input)
        else:
            num_layers = len(self.sequential)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(
                    start_idx + self.activation_checkpoint_interval, num_layers
                )

                funcs = self.sequential[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x,)

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx, end_idx), *x
                    )
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x


def recursive_setattr(m, attr, value, assert_type=None, type_filter=None):
    """
    Recursively set attributes on a pytorch module or an iterable of modules.
    If an assert_type is provided, it will assert that the type of the value is the same as the assert_type.
    If a type_filter is provided, it will only set attributes on modules that match that type.
    """
    if assert_type is not None:
        assert isinstance(value, assert_type), "Value is not the correct type."

    # if m is a list or a generator, iterate over the elements
    if isinstance(m, (list, GeneratorType)):
        for i in m:
            recursive_setattr(i, attr, value, assert_type, type_filter)
    elif isinstance(m, torch.nn.Module):
        if hasattr(m, attr):
            if type_filter is None or isinstance(m, type_filter):
                setattr(m, attr, value)
        if hasattr(m, "children"):
            recursive_setattr(m.children(), attr, value, assert_type, type_filter)


def _set_use_cache(modules, value: bool):
    """
    Recursively sets an use_cache to `value` on a list of pytorch modules, if they have a use_cache attribute.
    use_cache is used to decide whether we cache past key value activations or not in inference.
    """
    recursive_setattr(modules, "use_cache", value, assert_type=bool)


def configure_sparse_attention(config, attention_type, num_attention_heads, mpu):
    from deepspeed.ops.sparse_attention import (
        SparseSelfAttention,
        VariableSparsityConfig,
        FixedSparsityConfig,
        BigBirdSparsityConfig,
        BSLongformerSparsityConfig,
    )
    from deepspeed.ops.sparse_attention.sparsity_config import (
        LocalSlidingWindowSparsityConfig,
    )

    if attention_type == "sparse_fixed":
        # you can think of local window size as `block_size` * `num_local_blocks`.
        # so if you wanted to set a local window size of 256, set block size to 16 and `num_local_blocks` to 16
        sparsity_config = FixedSparsityConfig(
            num_heads=num_attention_heads,
            block=config.sparsity_config.get("block", 16),
            different_layout_per_head=config.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_local_blocks=config.sparsity_config.get("num_local_blocks", 4),
            num_global_blocks=config.sparsity_config.get("num_global_blocks", 1),
            num_different_global_patterns=config.sparsity_config.get(
                "num_different_global_patterns", 1
            ),
            attention="unidirectional",
            horizontal_global_attention=False,
        )
    elif attention_type == "sparse_variable":
        sparsity_config = VariableSparsityConfig(
            num_heads=num_attention_heads,
            block=config.sparsity_config.get("block", 16),
            different_layout_per_head=config.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_random_blocks=config.sparsity_config.get("num_random_blocks", 0),
            local_window_blocks=config.sparsity_config.get(
                "local_window_blocks", [4]
            ),
            global_block_indices=config.sparsity_config.get(
                "global_block_indices", [0]
            ),
            global_block_end_indices=config.sparsity_config.get(
                "global_block_end_indices", None
            ),
            attention="unidirectional",
            horizontal_global_attention=False,
        )
    elif attention_type == "local":
        # can configure with `num_local_blocks` or `num_sliding_window_blocks`
        num_local_blocks = config.sparsity_config.get(
            "num_local_blocks",
            config.sparsity_config.get("num_sliding_window_blocks", 4),
        )
        sparsity_config = LocalSlidingWindowSparsityConfig(
            num_heads=num_attention_heads,
            block=config.sparsity_config.get("block", 16),
            num_sliding_window_blocks=num_local_blocks,
            attention="unidirectional",
        )
    elif attention_type == "bigbird":
        sparsity_config = BigBirdSparsityConfig(
            num_heads=num_attention_heads,
            block=config.sparsity_config.get("block", 16),
            different_layout_per_head=config.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_random_blocks=config.sparsity_config.get("num_random_blocks", 1),
            num_sliding_window_blocks=config.sparsity_config.get(
                "num_sliding_window_blocks", 3
            ),
            num_global_blocks=config.sparsity_config.get("num_global_blocks", 1),
            attention="unidirectional",
        )
    elif attention_type == "bslongformer":
        sparsity_config = BSLongformerSparsityConfig(
            num_heads=num_attention_heads,
            block=config.sparsity_config.get("block", 16),
            different_layout_per_head=config.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_sliding_window_blocks=config.sparsity_config.get(
                "num_sliding_window_blocks", 3
            ),
            global_block_indices=config.sparsity_config.get(
                "global_block_indices", [0]
            ),
            global_block_end_indices=config.sparsity_config.get(
                "global_block_end_indices", None
            ),
            attention="unidirectional",
        )
    else:
        raise ValueError(f"Attention type {attention_type} not recognized")
    return SparseSelfAttention(
        sparsity_config=sparsity_config,
        max_seq_length=config.max_position_embeddings,
        attn_mask_mode="add",
        mpu=mpu,
    )


def get_fusion_type(config):
    fusion_type = SoftmaxFusionTypes.none
    if config.scaled_upper_triang_masked_softmax_fusion:
        fusion_type = SoftmaxFusionTypes.upper_triang
    elif config.scaled_masked_softmax_fusion:
        fusion_type = SoftmaxFusionTypes.general
    return fusion_type
