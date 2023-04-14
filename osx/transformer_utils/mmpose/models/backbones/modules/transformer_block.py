# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Rao Fu, RainbowSecret
# --------------------------------------------------------

import os
import math
import logging
import torch
import torch.nn as nn
from functools import partial

from .multihead_isa_pool_attention import InterlacedPoolAttention

from mmcv.cnn import build_conv_layer, build_norm_layer

BN_MOMENTUM = 0.1


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "drop_prob={}".format(self.drop_prob)


class MlpDWBN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = build_conv_layer(
            conv_cfg,
            in_features,
            hidden_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act1 = act_layer()
        self.norm1 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.dw3x3 = build_conv_layer(
            conv_cfg,
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
        )
        self.act2 = dw_act_layer()
        self.norm2 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.fc2 = build_conv_layer(
            conv_cfg,
            hidden_features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act3 = act_layer()
        self.norm3 = build_norm_layer(norm_cfg, out_features)[1]
        # self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x, H, W):
        if len(x.shape) == 3:
            B, N, C = x.shape
            if N == (H * W + 1):
                cls_tokens = x[:, 0, :]
                x_ = x[:, 1:, :].permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            else:
                x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

            x_ = self.fc1(x_)
            x_ = self.norm1(x_)
            x_ = self.act1(x_)
            x_ = self.dw3x3(x_)
            x_ = self.norm2(x_)
            x_ = self.act2(x_)
            # x_ = self.drop(x_)
            x_ = self.fc2(x_)
            x_ = self.norm3(x_)
            x_ = self.act3(x_)
            # x_ = self.drop(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1).contiguous()
            if N == (H * W + 1):
                x = torch.cat((cls_tokens.unsqueeze(1), x_), dim=1)
            else:
                x = x_
            return x

        elif len(x.shape) == 4:
            x = self.fc1(x)
            x = self.norm1(x)
            x = self.act1(x)
            x = self.dw3x3(x)
            x = self.norm2(x)
            x = self.act2(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.norm3(x)
            x = self.act3(x)
            x = self.drop(x)
            return x

        else:
            raise RuntimeError("Unsupported input shape: {}".format(x.shape))


class GeneralTransformerBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.attn = InterlacedPoolAttention(
            self.dim, num_heads=num_heads, window_size=window_size, dropout=attn_drop
        )

        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.out_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = MlpDWBN(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            out_features=self.out_dim,
            act_layer=act_layer,
            dw_act_layer=act_layer,
            drop=drop,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        B, C, H, W = x.size()
        # reshape
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        # Attention
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # reshape
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return x
