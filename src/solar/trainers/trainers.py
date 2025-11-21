# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from typing import Any, Dict, cast

import kornia.augmentation as K
import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, JaccardIndex, MetricCollection

from .losses import DiceWBCELoss, WeightedBCELoss

cmap = matplotlib.colors.ListedColormap(
    [
        (0, 0, 0, 0),  # Class 0, background
        (0, 0, 1, 1),  # Class 1, solar
    ]
)
rasterio_cmap = {0: (0, 0, 0, 0), 1: (0, 0, 255, 255)}


class SegmentationTask(pl.LightningModule):
    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=3,
                classes=2,
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=4,
                classes=2,
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

        if self.hparams["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()
        elif self.hparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass", classes=[0, 1])
        elif self.hparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=None, normalized=True
            )
        elif self.hparams["loss"] == "wbce":
            pos_class_weight = self.hparams["wbce_weight"]  # defaults to 0.7
            self.loss = WeightedBCELoss(pos_class_weight)
        elif self.hparams["loss"] == "dicewbce":
            self.loss = DiceWBCELoss()
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

        augmentation_list = [
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        ]

        if (
            "no_sharpness_augmentation" in kwargs
            and kwargs["no_sharpness_augmentation"]
        ):
            pass
        else:
            augmentation_list.append(K.RandomSharpness(p=0.5))
            pass

        if "color_jitter" in kwargs and kwargs["color_jitter"]:
            augmentation_list.append(
                K.ColorJitter(
                    p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                )
            )

        self.train_augmentations = K.AugmentationSequential(
            *augmentation_list,
            data_keys=["input", "mask"],
        )

        self.train_metrics = MetricCollection(
            [
                Accuracy(task="multiclass", num_classes=2, ignore_index=None),
                JaccardIndex(task="multiclass", num_classes=2, ignore_index=None),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """Training step - reports average accuracy and average IoU.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["mask"]
        with torch.no_grad():
            x, y = self.train_augmentations(x, y)
        y = y.long().squeeze()

        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)
        self.log_dict(self.train_metrics)

        return cast(Tensor, loss)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step - reports average accuracy and average IoU.

        Logs the first 10 validation samples to tensorboard as images with 3 subplots
        showing the image, mask, and predictions.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        if batch["mask"].shape[0] == 1:
            y = batch["mask"][0].long()
        else:
            y = batch["mask"].squeeze().long()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        tps = ((y == 1) & (y_hat_hard == 1)).sum()
        fps = ((y == 0) & (y_hat_hard == 1)).sum()
        fns = ((y == 1) & (y_hat_hard == 0)).sum()
        precision = tps / (tps + fps + 1e-5)
        recall = tps / (tps + fns + 1e-5)
        f1 = (2 * precision * recall) / (precision + recall + 1e-5)

        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

        self.log("val_loss", loss)
        self.val_metrics(y_hat_hard, y)
        self.log_dict(self.val_metrics)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                x[0].cpu().numpy(), 0, 3
            )
            mask = y[0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            axs[0].imshow(np.clip(img, 0, 1))
            axs[0].axis("off")
            axs[1].imshow(mask + 1, vmin=0, vmax=3, cmap=cmap, interpolation="none")
            axs[1].axis("off")
            axs[2].imshow(pred + 1, vmin=0, vmax=3, cmap=cmap, interpolation="none")
            axs[2].axis("off")
            plt.tight_layout()
            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"].long().squeeze()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        self.test_metrics(y_hat_hard, y)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        if self.hparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"],
            )
        elif self.hparams["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"],
            )
        elif self.hparams["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"],
            )
        elif self.hparams["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"],
                amsgrad=False,
            )
        else:
            raise ValueError(
                f"Optimizer '{self.hparams['optimizer']}' is not supported."
            )

        if self.hparams["scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["learning_rate_schedule_patience"],
                eta_min=1e-6,
            )
        elif self.hparams["scheduler"] == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.hparams["learning_rate_schedule_patience"],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_start(self) -> None:
        lr = self.optimizers().param_groups[0]["lr"]
        self.logger.experiment.add_scalar("lr", lr, self.current_epoch)
