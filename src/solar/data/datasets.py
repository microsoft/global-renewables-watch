# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import collections
import random

import cv2
import mercantile
import numpy as np
import rasterio
import rasterio.windows
import requests
import shapely.geometry
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _list_dict_to_dict_list(samples):
    """Convert a list of dictionaries to a dictionary of lists.

    Args:
        samples: a list of dictionaries

    Returns:
        a dictionary of lists
    """
    collated = collections.defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            collated[key].append(value)
    return collated


def stack_samples(samples):
    """Stack a list of samples along a new axis.

    Useful for forming a mini-batch of samples to pass to
    :class:`torch.utils.data.DataLoader`.

    Args:
        samples: list of samples

    Returns:
        a single sample
    """
    collated = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.stack(value)
    return collated


class TileDataset(Dataset):
    def __init__(self, image_fns, mask_fns, transforms=None, sanity_check=False):
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        if self.mask_fns is not None:
            assert len(image_fns) == len(mask_fns)

        # Check to make sure that all the image and mask tile pairs are the same size
        # as a sanity check
        if sanity_check and mask_fns is not None:
            for image_fn, mask_fn in zip(image_fns, mask_fns):
                with rasterio.open(image_fn[0]) as f:
                    image_height, image_width = f.shape
                with rasterio.open(mask_fn) as f:
                    mask_height, mask_width = f.shape
                assert image_height == mask_height
                assert image_width == mask_width

        self.transforms = transforms

    def __getitem__(self, index):
        i, y, x, patch_size = index
        assert 0 <= i < len(self.image_fns)

        sample = {
            "y": y,
            "x": x,
        }

        window = rasterio.windows.Window(x, y, patch_size, patch_size)

        # Load imagery
        stack = []
        for j in range(len(self.image_fns[i])):
            image_fn = self.image_fns[i][j]
            with rasterio.open(image_fn) as f:
                image = f.read(window=window)
                stack.append(image)
        stack = np.concatenate(stack, axis=0).astype(np.int32)
        sample["image"] = torch.from_numpy(stack)

        # Load mask
        if self.mask_fns is not None:
            mask_fn = self.mask_fns[i]
            with rasterio.open(mask_fn) as f:
                mask = f.read(window=window)
            sample["mask"] = torch.from_numpy(mask)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class InferenceDataset(Dataset):
    def __init__(self, input_fns, output_fns=None, transforms=None):
        super().__init__()
        self.input_fns = input_fns
        self.output_fns = output_fns
        self.transforms = transforms

    def __getitem__(self, index):
        input_fn = self.input_fns[index]
        with rasterio.open(input_fn) as f:
            image = f.read()
            affine = f.transform
            input_height, input_width = f.shape
        assert input_height == 4096 and input_width == 4096
        sample = {
            "image": torch.from_numpy(image),
            "input_fn": input_fn,
            "transform": np.array(
                [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f]
            ),
        }
        if self.output_fns is not None:
            sample["output_fn"] = self.output_fns[index]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.input_fns)
