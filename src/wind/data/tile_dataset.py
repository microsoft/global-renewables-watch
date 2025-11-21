# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import rasterio
import torch
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from torch.utils.data.dataset import Dataset


class TileInferenceDataset(Dataset):
    def __init__(
        self,
        fn,
        chip_size,
        stride,
        transform=None,
        windowed_sampling=False,
        verbose=False,
    ):
        """A torch Dataset for sampling a grid of chips that covers an input tile.

        If `chip_size` doesn't divide the height of the tile evenly (which is what is likely to happen) then we will sample an additional row of chips that are aligned to the bottom of the file.
        We do a similar operation if `chip_size` doesn't divide the width of the tile evenly -- by appending an additional column.

        Note: without a `transform` we will return chips in (height, width, channels) format in whatever the tile's dtype is.

        Args:
            fn: The path to the file to sample from (this can be anything that rasterio.open(...) knows how to read).
            chip_size: The size of chips to return (chips will be squares).
            stride: How much we move the sliding window to sample the next chip. If this is is less than `chip_size` then we will get overlapping windows, if it is > `chip_size` then some parts of the tile will not be sampled.
            transform: A torchvision Transform to apply on each chip.
            windowed_sample: If `True` we will use rasterio.windows.Window to sample chips without every loading the entire file into memory, else, we will load the entire tile up-front and index into it to sample chips.
            verbose: Flag to control printing stuff.
        """
        self.fn = fn
        self.chip_size = chip_size

        self.transform = transform
        self.windowed_sampling = windowed_sampling
        self.verbose = verbose

        with rasterio.open(self.fn) as f:
            height, width = f.height, f.width
            self.num_channels = f.count
            self.dtype = f.profile["dtype"]
            if not windowed_sampling:  # if we aren't using windowed sampling, then go ahead and read in all of the data
                self.data = np.rollaxis(f.read()[:3], 0, 3)

        self.chip_coordinates = []  # upper left coordinate (y,x), of each chip that this Dataset will return
        for y in list(range(0, height - self.chip_size, stride)) + [
            height - self.chip_size
        ]:
            for x in list(range(0, width - self.chip_size, stride)) + [
                width - self.chip_size
            ]:
                self.chip_coordinates.append((y, x))
        self.num_chips = len(self.chip_coordinates)

        if self.verbose:
            print(
                "Constructed TileInferenceDataset -- we have %d by %d file with %d channels with a dtype of %s. We are sampling %d chips from it."
                % (height, width, self.num_channels, self.dtype, self.num_chips)
            )

    def __getitem__(self, idx):
        """
        Returns:
            A tuple (chip, (y,x)): `chip` is the chip that we sampled from the larger tile. (y,x) are the indices of the upper left corner of the chip.
        """
        y, x = self.chip_coordinates[idx]

        if self.windowed_sampling:
            try:
                with rasterio.Env():
                    with rasterio.open(self.fn) as f:
                        img = np.rollaxis(
                            f.read(
                                window=rasterio.windows.Window(
                                    x, y, self.chip_size, self.chip_size
                                )
                            )[:3],
                            0,
                            3,
                        )
            except RasterioIOError as e:  # NOTE(caleb): I put this here to catch weird errors that I was seeing occasionally when trying to read from COGS - I don't remember the details though
                print("Reading %d failed, returning 0's" % (idx))
                img = np.zeros(
                    (self.chip_size, self.chip_size, self.num_channels), dtype=np.uint8
                )
        else:
            img = self.data[y : y + self.chip_size, x : x + self.chip_size]

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array((y, x))

    def __len__(self):
        return self.num_chips
