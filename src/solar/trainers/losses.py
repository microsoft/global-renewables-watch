# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn


class DiceWBCELoss(nn.Module):
    """Dice weighted binary crossentropy loss."""

    def __init__(self, batch=True):
        super(DiceWBCELoss, self).__init__()
        self.batch = batch
        self.bce_loss = WeightedBCELoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_pred * y_true)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b


class DiceLoss(nn.Module):
    """Dice loss"""

    def __init__(self, batch=True):
        super(DiceLoss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            print(y_pred.shape)
            print(y_true.shape)
            intersection = torch.sum(y_pred * y_true)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred, y_true)


class WeightedBCELoss(nn.Module):
    """Weighted BCE loss."""

    def __init__(self, pos_class_weight=0.7, batch=True):
        super(WeightedBCELoss, self).__init__()
        assert pos_class_weight < 1.0 and pos_class_weight > 0.0, (
            "class weight value must be between 0 and 1"
        )
        weights = [1.0 - pos_class_weight, pos_class_weight]
        class_weights = torch.FloatTensor(weights)
        self.bce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, y_pred, y_true):
        loss = self.bce_loss(y_pred, y_true)
        return loss
