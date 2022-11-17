"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    @th.cuda.amp.custom_fwd
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_length = length
        ctx.save_for_backward(*args)
        with th.no_grad():
            output_tensors = ctx.run_function(*args[:length])
        return output_tensors

    @staticmethod
    @th.cuda.amp.custom_bwd
    def backward(ctx, *output_grads):
        args = list(ctx.saved_tensors)

        # Filter for inputs that require grad. If none, exit early.
        input_indices = [i for (i, x) in enumerate(args) if x.requires_grad]
        if not input_indices:
            return (None, None) + tuple(None for _ in args)

        with th.enable_grad():
            for i in input_indices:
                if i < ctx.input_length:
                    # Not sure why the OAI code does this little
                    # dance. It might not be necessary.
                    args[i] = args[i].detach().requires_grad_()
                    args[i] = args[i].view_as(args[i])
            output_tensors = ctx.run_function(*args[:ctx.input_length])

        if isinstance(output_tensors, th.Tensor):
            output_tensors = [output_tensors]

        # Filter for outputs that require grad. If none, exit early.
        out_and_grads = [(o, g) for (o, g) in zip(output_tensors, output_grads) if o.requires_grad]
        if not out_and_grads:
            return (None, None) + tuple(None for _ in args)

        # Compute gradients on the filtered tensors.
        computed_grads = th.autograd.grad(
            [o for (o, g) in out_and_grads],
            [args[i] for i in input_indices],
            [g for (o, g) in out_and_grads]
        )

        # Reassemble the complete gradient tuple.
        input_grads = [None for _ in args]
        for (i, g) in zip(input_indices, computed_grads):
            input_grads[i] = g
        return (None, None) + tuple(input_grads)
