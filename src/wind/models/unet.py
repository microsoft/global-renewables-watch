# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""U-Net segmentation model."""

import argparse
from typing import Any, Dict, Union

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Pair of convolutions used by both the encoder and decoder of the U-Net."""

    def __init__(
        self,
        inchannels: int,
        outchannels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize the convolutional block.

        Args:
            inchannels: Number of input channels.
            outchannels: Number of output channels.
            kernel_size: Size of the convolutional kernels.
            stride: Stride of the convolutions.
            padding: Padding applied by the convolutions.
            bias: Whether the convolutions learn an additive bias.
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                inchannels,
                outchannels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannels),
            nn.Conv2d(
                outchannels,
                outchannels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Input tensor of shape (batch, inchannels, height, width).

        Returns:
            Tensor of shape (batch, outchannels, height, width).
        """
        return self.conv_block(x)


class UpBlock(nn.Module):
    """Decoder block that upsamples its input and merges it with a skip connection."""

    def __init__(
        self,
        inchannels: int,
        outchannels: int,
        kernel_size: int = 2,
        stride: int = 2,
    ) -> None:
        """Initialize the upsampling block.

        Args:
            inchannels: Number of input channels.
            outchannels: Number of output channels.
            kernel_size: Size of the transposed convolutional kernel.
            stride: Stride of the transposed convolution.
        """
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            inchannels, outchannels, kernel_size=kernel_size, stride=stride
        )
        self.conv = ConvBlock(inchannels, outchannels)

    def forward(self, x: torch.Tensor, skips: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Input tensor from the previous decoder stage.
            skips: Matching encoder output to concatenate with.

        Returns:
            Tensor of shape (batch, outchannels, 2 * height, 2 * width).
        """
        x = self.upconv(x)
        x = torch.cat([skips, x], 1)
        return self.conv(x)


class UnetModel(nn.Module):
    """U-Net model used for wind turbine segmentation."""

    def __init__(self, opts: Union[Dict[str, Any], argparse.Namespace]) -> None:
        """Initialize the U-Net.

        Args:
            opts: Options mapping (or argparse Namespace) containing
                ``input_channels``, ``first_layer_filters``, ``net_depth``, and
                ``num_classes``. Checkpoints store these under the ``params``
                key.
        """
        super().__init__()
        if not isinstance(opts, dict):
            opts = vars(opts)

        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.in_channels = opts["input_channels"]
        self.out_channels = opts["first_layer_filters"]
        self.net_depth = opts["net_depth"]
        self.num_classes = opts["num_classes"]

        # down transformations
        for _ in range(self.net_depth):
            self.downblocks.append(ConvBlock(self.in_channels, self.out_channels))
            self.in_channels, self.out_channels = (
                self.out_channels,
                2 * self.out_channels,
            )

        # midpoint
        self.middle_conv = ConvBlock(self.in_channels, self.out_channels)

        # up transformations
        self.in_channels, self.out_channels = (
            self.out_channels,
            self.out_channels // 2,
        )
        for _ in range(self.net_depth):
            self.upblocks.append(UpBlock(self.in_channels, self.out_channels))
            self.in_channels, self.out_channels = (
                self.out_channels,
                self.out_channels // 2,
            )

        self.seg_layer = nn.Conv2d(
            2 * self.out_channels, self.num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel logits for a batch of images.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Logits of shape (batch, num_classes, height, width).
        """
        encoder_outputs = []
        for op in self.downblocks:
            encoder_outputs.append(op(x))
            x = self.pool(encoder_outputs[-1])

        x = self.middle_conv(x)

        for op in self.upblocks:
            x = op(x, encoder_outputs.pop())

        return self.seg_layer(x)
