# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Rao Fu, RainbowSecret
# --------------------------------------------------------

import os
import copy
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .transformer_block import TransformerBlock

from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    build_plugin_layer,
    constant_init,
    kaiming_init,
)


class BasicBlock(nn.Module):
    """Only replce the second 3x3 Conv with the TransformerBlocker"""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        mhsa_flag=False,
        num_heads=1,
        num_halo_block=1,
        num_mlp_ratio=4,
        num_sr_ratio=1,
        with_rpe=False,
        with_ffn=True,
    ):
        super(BasicBlock, self).__init__()
        norm_cfg = copy.deepcopy(norm_cfg)

        self.in_channels = inplanes
        self.out_channels = planes
        self.stride = stride
        self.with_cp = with_cp
        self.downsample = downsample

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=1,
            dilation=1,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)

        if not mhsa_flag:
            self.conv2 = build_conv_layer(
                conv_cfg, planes, planes, 3, padding=1, bias=False
            )
            self.add_module(self.norm2_name, norm2)
        else:
            self.conv2 = TransformerBlock(
                planes,
                num_heads=num_heads,
                mlp_ratio=num_mlp_ratio,
                sr_ratio=num_sr_ratio,
                input_resolution=num_resolution,
                with_rpe=with_rpe,
                with_ffn=with_ffn,
            )

        self.relu = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out
