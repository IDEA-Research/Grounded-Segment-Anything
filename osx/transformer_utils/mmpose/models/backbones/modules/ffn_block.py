# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Rao Fu, RainbowSecret
# --------------------------------------------------------

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpDW(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        if N == (H * W + 1):
            cls_tokens = x[:, 0, :]
            x_ = x[:, 1:, :].permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        else:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

        x_ = self.fc1(x_)
        x_ = self.act1(x_)
        x_ = self.dw3x3(x_)
        x_ = self.act2(x_)
        x_ = self.drop(x_)
        x_ = self.fc2(x_)
        x_ = self.drop(x_)
        x_ = x_.reshape(B, C, -1).permute(0, 2, 1).contiguous()

        if N == (H * W + 1):
            x = torch.cat((cls_tokens.unsqueeze(1), x_), dim=1)
        else:
            x = x_

        return x


class MlpDWBN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
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
        x_ = self.drop(x_)
        x_ = self.fc2(x_)
        x_ = self.norm3(x_)
        x_ = self.act3(x_)
        x_ = self.drop(x_)
        x_ = x_.reshape(B, C, -1).permute(0, 2, 1).contiguous()

        if N == (H * W + 1):
            x = torch.cat((cls_tokens.unsqueeze(1), x_), dim=1)
        else:
            x = x_

        return x


class MlpDWBN2D(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
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
