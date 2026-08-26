# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two convolutional layers used by the U-Net encoder and decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        # Historical checkpoints include these parameters even though forward
        # uses conv_block, so retain them for state-dictionary compatibility.
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class UpBlock(nn.Module):
    """Upsample a decoder feature map and combine its skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
    ):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UnetModel(nn.Module):
    """U-Net architecture used by historical wind model checkpoints."""

    def __init__(self, opts: Mapping[str, Any]):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.in_channels = opts["input_channels"]
        self.out_channels = opts["first_layer_filters"]
        self.net_depth = opts["net_depth"]
        self.num_classes = opts["num_classes"]

        for _ in range(self.net_depth):
            conv = ConvBlock(self.in_channels, self.out_channels)
            self.downblocks.append(conv)
            self.in_channels, self.out_channels = (
                self.out_channels,
                2 * self.out_channels,
            )

        self.middle_conv = ConvBlock(self.in_channels, self.out_channels)

        self.in_channels, self.out_channels = (
            self.out_channels,
            self.out_channels // 2,
        )
        for _ in range(self.net_depth):
            upconv = UpBlock(self.in_channels, self.out_channels)
            self.upblocks.append(upconv)
            self.in_channels, self.out_channels = (
                self.out_channels,
                self.out_channels // 2,
            )

        self.seg_layer = nn.Conv2d(
            2 * self.out_channels,
            self.num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for downblock in self.downblocks:
            skip_connections.append(downblock(x))
            x = self.pool(skip_connections[-1])

        x = self.middle_conv(x)
        for upblock in self.upblocks:
            x = upblock(x, skip_connections.pop())

        return self.seg_layer(x)
