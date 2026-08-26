# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
from torch import nn
from torchvision.models import resnet50


class FCN8(nn.Module):
    """FCN8 segmentation model with a ResNet-50 backbone."""

    def __init__(self):
        super().__init__()
        self.n_classes = 1

        self.resnet50_32s = resnet50(weights=None)
        expansion = self.resnet50_32s.layer1[0].expansion
        self.resnet50_32s.fc = nn.Sequential()

        self.score_32s = nn.Conv2d(512 * expansion, self.n_classes, kernel_size=1)
        self.score_16s = nn.Conv2d(256 * expansion, self.n_classes, kernel_size=1)
        self.score_8s = nn.Conv2d(128 * expansion, self.n_classes, kernel_size=1)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_spatial_dim = x.shape[2:]

        self.resnet50_32s.eval()
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

        logits_16s += nn.functional.interpolate(
            logits_32s,
            size=logits_16s.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        logits_8s += nn.functional.interpolate(
            logits_16s,
            size=logits_8s.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        return nn.functional.interpolate(
            logits_8s,
            size=input_spatial_dim,
            mode="bilinear",
            align_corners=True,
        )
