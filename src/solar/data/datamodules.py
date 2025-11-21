# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from typing import Any, Callable, Dict, List, Optional

import lightning.pytorch as pl
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from .datasets import TileDataset, stack_samples
from .samplers import GridGeoSampler, RandomGeoSampler


def preprocess(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single sample from the Dataset."""
    sample["image"] = sample["image"][:3] / 255.0
    sample["image"] = sample["image"].float()

    if "mask" in sample:
        sample["mask"] = sample["mask"].float()

    return sample


def pad_to(
    size: int = 512, image_value: int = 0, mask_value: int = 0
) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
    """Returns a function to perform a padding transform on a single sample.
    Args:
        size: output image size
        image_value: value to pad image with
        mask_value: value to pad mask with
    Returns:
        function to perform padding
    """

    def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        _, height, width = sample["image"].shape
        assert height <= size and width <= size

        height_pad = size - height
        width_pad = size - width

        sample["image"] = F.pad(
            sample["image"],
            (0, width_pad, 0, height_pad),
            mode="constant",
            value=image_value,
        )
        if "mask" in sample:
            sample["mask"] = F.pad(
                sample["mask"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=mask_value,
            )
        return sample

    return pad_inner


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_fns: Dict[str, List[str]],
        mask_fns: Dict[str, List[str]],
        batch_size: int = 64,
        patch_size: int = 256,
        num_workers: int = 4,
        train_batches_per_epoch=512,
        valid_batches_per_epoch=32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.train_patches_per_epoch = train_batches_per_epoch * batch_size
        self.valid_patches_per_epoch = valid_batches_per_epoch * batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        transforms = Compose([preprocess, pad_to(self.patch_size)])

        self.train_dataset = TileDataset(
            self.image_fns["train"],
            self.mask_fns["train"],
            transforms=transforms,
            sanity_check=False,
        )

        self.val_dataset = TileDataset(
            self.image_fns["valid"],
            self.mask_fns["valid"],
            transforms=transforms,
            sanity_check=False,
        )

        self.test_dataset = TileDataset(
            self.image_fns["test"],
            self.mask_fns["test"],
            transforms=transforms,
            sanity_check=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""

        sampler = RandomGeoSampler(
            self.image_fns["train"], self.train_patches_per_epoch, self.patch_size
        )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        sampler = GridGeoSampler(
            self.image_fns["valid"],
            list(range(len(self.image_fns["valid"]))),
            224,
            224,
        )

        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        sampler = GridGeoSampler(
            self.image_fns["test"], list(range(len(self.image_fns["test"]))), 224, 224
        )

        return DataLoader(
            self.test_dataset,
            sampler=sampler,
            batch_size=16,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )
