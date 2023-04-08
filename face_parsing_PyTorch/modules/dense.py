from collections import OrderedDict

import torch
import torch.nn as nn

from .bn import ABN


class DenseModule(nn.Module):
    def __init__(self, in_channels, growth, layers, bottleneck_factor=4, norm_act=ABN, dilation=1):
        super(DenseModule, self).__init__()
        self.in_channels = in_channels
        self.growth = growth
        self.layers = layers

        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        for i in range(self.layers):
            self.convs1.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(in_channels)),
                ("conv", nn.Conv2d(in_channels, self.growth * bottleneck_factor, 1, bias=False))
            ])))
            self.convs3.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.growth * bottleneck_factor)),
                ("conv", nn.Conv2d(self.growth * bottleneck_factor, self.growth, 3, padding=dilation, bias=False,
                                   dilation=dilation))
            ])))
            in_channels += self.growth

    @property
    def out_channels(self):
        return self.in_channels + self.growth * self.layers

    def forward(self, x):
        inputs = [x]
        for i in range(self.layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs1[i](x)
            x = self.convs3[i](x)
            inputs += [x]

        return torch.cat(inputs, dim=1)
