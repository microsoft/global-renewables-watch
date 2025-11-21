# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import rasterio
from torch.utils.data import Sampler


class RandomGeoSampler(Sampler):
    def __init__(self, image_fns, length, patch_size):
        self.tile_sample_weights = []
        self.tile_heights = []
        self.tile_widths = []
        self.length = length
        self.patch_size = patch_size

        for image_fn in image_fns:
            with rasterio.open(image_fn[0]) as f:
                image_height, image_width = f.shape
            self.tile_sample_weights.append(image_height * image_width)
            self.tile_heights.append(image_height)
            self.tile_widths.append(image_width)

        self.tile_sample_weights = np.array(self.tile_sample_weights)
        self.tile_sample_weights = (
            self.tile_sample_weights / self.tile_sample_weights.sum()
        )
        self.num_tiles = len(self.tile_sample_weights)

    def __iter__(self):
        for _ in range(len(self)):
            i = np.random.choice(self.num_tiles, p=self.tile_sample_weights)

            max_y_size = max(self.tile_heights[i] - self.patch_size, 1)
            max_x_size = max(self.tile_widths[i] - self.patch_size, 1)

            y = np.random.randint(0, max_y_size)
            x = np.random.randint(0, max_x_size)

            yield (i, y, x, self.patch_size)

    def __len__(self):
        return self.length


class GridGeoSampler(Sampler):
    def __init__(
        self,
        image_fns,
        image_fn_indices,
        patch_size=256,
        stride=256,
    ):
        self.image_fn_indices = image_fn_indices
        self.patch_size = patch_size

        # tuples of the form (i, y, x, patch_size) that index into a CustomTileDataset
        self.indices = []
        for i in self.image_fn_indices:
            with rasterio.open(image_fns[i][0]) as f:
                height, width = f.height, f.width

            if patch_size > height and patch_size > width:
                self.indices.append((i, 0, 0, self.patch_size))
            else:
                for y in list(range(0, height - patch_size, stride)) + [
                    height - patch_size
                ]:
                    for x in list(range(0, width - patch_size, stride)) + [
                        width - patch_size
                    ]:
                        self.indices.append((i, y, x, self.patch_size))
        self.num_chips = len(self.indices)

    def __iter__(self):
        for index in self.indices:
            yield index

    def __len__(self):
        return self.num_chips
