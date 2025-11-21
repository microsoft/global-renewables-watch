# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys

import fiona
import numpy as np
import rasterio
import rasterio.mask
import torch
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from shapely.geometry import Point, box
from torch.utils.data.dataset import IterableDataset
from torchvision import transforms


class StreamingGeospatialDataset(IterableDataset):
    def __init__(
        self,
        imagery_fns,
        label_fns=None,
        groups=None,
        subtile_bounds=None,
        chip_size=256,
        num_chips_per_tile=200,
        vector_labels_fn=None,
        windowed_sampling=False,
        image_transform=None,
        label_transform=None,
        nodata_check=None,
        verbose=False,
    ):
        """A torch Dataset for randomly sampling chips from a list of tiles. When used in conjunction with a DataLoader that has `num_workers>1` this Dataset will assign each worker to sample chips from disjoint sets of tiles.
        Args:
            imagery_fns: A list of filenames (or URLS -- anything that `rasterio.open()` can read) pointing to imagery tiles.
            label_fns: A list of filenames of the same size as `imagery_fns` pointing to label mask tiles or `None` if the Dataset should operate in "imagery only mode". Note that we expect `imagery_fns[i]` and `label_fns[i]` to have the same dimension and coordinate system.
            groups: Optional: A list of integers of the same size as `imagery_fns` that gives the "group" membership of each tile. This can be used to normalize imagery from different groups differently.
            subtile_bounds: Boundary limits to sample patches from a tile. Useful when a single tile is available for training, val and test. Format: [x_min, x_max, y_min, y_max]
            chip_size: Desired size of chips (in pixels).
            num_chips_per_tile: Desired number of chips to sample for each tile.
            windowed_sampling: Flag indicating whether we should sample each chip with a read using `rasterio.windows.Window` or whether we should read the whole tile into memory, then sample chips.
            image_transform: A function to apply to each image chip object. If this is `None`, then the only transformation applied to the loaded imagery will be to convert it to a `torch.Tensor`. If this is not `None`, then the function should return a `Torch.tensor`. Further, if `groups` is not `None` then the transform function should expect the imagery as the first argument and the group as the second argument.
            label_transform: Similar to image_transform, but applied to label chips.
            nodata_check: A method that will check an `(image_chip)` or `(image_chip, label_chip)` (if `label_fns` are provided) and return whether or not the chip should be skipped. This can be used, for example, to skip chips that contain nodata values.
            verbose: If `False` we will be quiet.
        """

        if label_fns is None or vector_labels_fn is not None:
            self.fns = imagery_fns
            self.use_labels = False
        else:
            self.fns = list(zip(imagery_fns, label_fns))
            self.use_labels = True

        self.groups = groups
        self.vector_labels_fn = vector_labels_fn

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling = windowed_sampling
        self.subtile_bounds = subtile_bounds

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if (
            worker_info is None
        ):  # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random.shuffle(self.fns)  # in place
        # NOTE: A warning, when different workers are created they will all have the same numpy random seed, however will have different torch random seeds. If you want to use numpy random functions, seed appropriately.
        # seed = torch.randint(low=0,high=2**32-1,size=(1,)).item()
        # np.random.seed(seed) # when different workers spawn, they have the same numpy random seed...

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id + 1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):
            label_fn = None
            if self.use_labels:
                img_fn, label_fn = self.fns[idx]
            else:
                img_fn = self.fns[idx]

            if self.groups is not None:
                group = self.groups[idx]
            else:
                group = None

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, label_fn, group)

    def stream_chips(self):
        for img_fn, label_fn, group in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            label_fp = rasterio.open(label_fn, "r") if self.use_labels else None

            height, width = img_fp.shape
            if height <= self.chip_size or width <= self.chip_size:
                continue
            if (
                self.use_labels
            ):  # garantee that our label mask has the same dimensions as our imagery
                t_height, t_width = label_fp.shape
                assert height == t_height and width == t_width

            try:
                # If we aren't in windowed sampling mode then we should read the entire tile up front
                if not self.windowed_sampling:
                    img_data = np.rollaxis(img_fp.read()[:3], 0, 3)
                    if self.use_labels:
                        label_data = (
                            label_fp.read().squeeze()
                        )  # assume the label geotiff has a single channel
                    elif self.vector_labels_fn is not None:
                        with fiona.open(self.vector_labels_fn, "r") as shapefile:
                            shapes = [feature["geometry"] for feature in shapefile]

                            bounds = img_fp.bounds
                            geom = box(*bounds)

                            matches = []
                            for shape in shapes:
                                if shape["type"] == "MultiPoint":
                                    pts = shape["coordinates"]
                                    for pt_shape in pts:
                                        pt = Point(pt_shape)
                                        if geom.contains(pt):
                                            matches.append(pt_shape)
                                else:
                                    pt = Point(shape["coordinates"])
                                    if geom.contains(pt):
                                        matches.append(shape["coordinates"])

                            label_data = np.zeros(
                                (img_fp.read().shape[1], img_fp.read().shape[2])
                            )
                            # print(matches)
                            for i, (lon, lat) in enumerate(matches):
                                py, px = img_fp.index(lon, lat)
                                label_data[py, px] = 1

                i = 0
                while i < self.num_chips_per_tile:
                    # Select the top left pixel of our chip randomly
                    if self.subtile_bounds is None:
                        x = np.random.randint(0, width - self.chip_size)
                        y = np.random.randint(0, height - self.chip_size)
                    else:
                        x = np.random.randint(
                            self.subtile_bounds[0],
                            self.subtile_bounds[1] - self.chip_size,
                        )
                        y = np.random.randint(
                            self.subtile_bounds[2],
                            self.subtile_bounds[3] - self.chip_size,
                        )

                    # Read imagery / labels
                    img = None
                    labels = None
                    if self.windowed_sampling:
                        img = np.rollaxis(
                            img_fp.read(
                                window=Window(x, y, self.chip_size, self.chip_size)
                            ),
                            0,
                            3,
                        )
                        if self.use_labels:
                            labels = label_fp.read(
                                window=Window(x, y, self.chip_size, self.chip_size)
                            ).squeeze()
                    else:
                        img = img_data[
                            y : y + self.chip_size, x : x + self.chip_size, :
                        ]
                        if self.use_labels or (self.vector_labels_fn is not None):
                            labels = label_data[
                                y : y + self.chip_size, x : x + self.chip_size
                            ]

                    # Check for no data
                    if self.nodata_check is not None:
                        if self.use_labels:
                            skip_chip = self.nodata_check(img, labels)
                        else:
                            skip_chip = self.nodata_check(img)

                        if skip_chip:  # The current chip has been identified as invalid by the `nodata_check(...)` method
                            num_skipped_chips += 1
                            continue

                    # Transform the imagery
                    if self.image_transform is not None:
                        if self.groups is None:
                            img = self.image_transform(img)
                        else:
                            img = self.image_transform(img, group)
                    else:
                        img = torch.from_numpy(img).squeeze()

                    # Transform the labels
                    if self.use_labels:
                        if self.label_transform is not None:
                            if self.groups is None:
                                labels = self.label_transform(labels)
                            else:
                                labels = self.label_transform(labels, group)
                        else:
                            labels = torch.from_numpy(labels).squeeze()

                    i += 1
                    # Note, that img should be a torch "Double" type (i.e. a np.float32) and labels should be a torch "Long" type (i.e. np.int64)
                    if self.use_labels or (self.vector_labels_fn is not None):
                        yield img, labels
                    else:
                        yield img
            except RasterioIOError as e:  # NOTE(caleb): I put this here to catch weird errors that I was seeing occasionally when trying to read from COGS - I don't remember the details though
                print("WARNING: Reading %s failed, skipping..." % (img_fn))

            # Close file pointers
            img_fp.close()
            if self.use_labels:
                label_fp.close()

            if num_skipped_chips > 0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())
