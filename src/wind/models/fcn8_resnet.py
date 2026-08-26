# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""FCN8 segmentation model with a ResNet-50 backbone."""

import torch
import torch.nn as nn
import torchvision


class FCN8(nn.Module):
    """Fully convolutional network with a ResNet-50 backbone.

    Predictions are formed by summing 1x1 convolutional scores computed at
    output strides of 32, 16, and 8, then bilinearly upsampling the result back
    to the input resolution.
    """

    def __init__(self, n_classes: int = 1, pretrained_backbone: bool = False) -> None:
        """Initialize the FCN8 model.

        Args:
            n_classes: Number of output channels to predict.
            pretrained_backbone: If True, initialize the ResNet-50 backbone with
                ImageNet weights. This is only useful when training from scratch
                -- these weights are overwritten when loading a checkpoint.
        """
        super().__init__()
        self.n_classes = n_classes

        weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        resnet50_32s = torchvision.models.resnet50(weights=weights)
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion

        # The classification head is unused, drop its parameters
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        self.score_32s = nn.Conv2d(
            512 * resnet_block_expansion_rate, self.n_classes, kernel_size=1
        )
        self.score_16s = nn.Conv2d(
            256 * resnet_block_expansion_rate, self.n_classes, kernel_size=1
        )
        self.score_8s = nn.Conv2d(
            128 * resnet_block_expansion_rate, self.n_classes, kernel_size=1
        )

        # The batch norm layers are frozen, matching how the released
        # checkpoints were trained
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel logits for a batch of images.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Logits of shape (batch, n_classes, height, width).
        """
        # The frozen batch norm layers always use their running statistics
        self.resnet50_32s.eval()
        input_spatial_dim = x.size()[2:]

        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)

        x = self.resnet50_32s.layer2(x)
        logits_8s = self.score_8s(x)

        x = self.resnet50_32s.layer3(x)
        logits_16s = self.score_16s(x)

        x = self.resnet50_32s.layer4(x)
        logits_32s = self.score_32s(x)

        logits_16s = logits_16s + nn.functional.interpolate(
            logits_32s,
            size=logits_16s.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        logits_8s = logits_8s + nn.functional.interpolate(
            logits_16s,
            size=logits_8s.size()[2:],
            mode="bilinear",
            align_corners=True,
        )

        logits_upsampled = nn.functional.interpolate(
            logits_8s, size=input_spatial_dim, mode="bilinear", align_corners=True
        )
        return logits_upsampled
