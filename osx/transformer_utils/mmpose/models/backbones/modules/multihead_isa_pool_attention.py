# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Lang Huang, RainbowSecret from:
#   https://github.com/openseg-group/openseg.pytorch/blob/master/lib/models/modules/isa_block.py
# --------------------------------------------------------

import os
import pdb
import math
import torch
import torch.nn as nn

from .multihead_isa_attention import MHA_, PadBlock, LocalPermuteModule


class InterlacedPoolAttention(nn.Module):
    r"""interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, embed_dim, num_heads, window_size=7, rpe=True, **kwargs):
        super(InterlacedPoolAttention, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_rpe = rpe

        self.attn = MHA_(
            embed_dim, num_heads, rpe=rpe, window_size=window_size, **kwargs
        )
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)

    def forward(self, x, H, W, **kwargs):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # attention
        # pad
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.size())
        # attention
        out, _, _ = self.attn(
            x_permute, x_permute, x_permute, rpe=self.with_rpe, **kwargs
        )
        # reverse permutation
        out = self.permute_helper.rev_permute(out, x_pad.size())
        # de-pad, pooling with `ceil_mode=True` will do implicit padding, so we need to remove it, too
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C)
