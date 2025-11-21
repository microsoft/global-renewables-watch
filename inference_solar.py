# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os

import numpy as np
import rasterio
import torch
from affine import Affine
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.solar.data.datamodules import preprocess
from src.solar.data.datasets import InferenceDataset
from src.solar.trainers.trainers import SegmentationTask


def get_args() -> argparse.Namespace:
    """Set up the argument parser.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-fn",
        required=True,
        type=str,
        help="Path to the model checkpoint (.ckpt format)",
    )
    parser.add_argument(
        "--input-fn",
        required=True,
        type=str,
        help="Path to the raster to run the model on",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to write prediction tiles to",
    )
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output tiles if they exist",
    )
    command_group.add_argument(
        "--skip",
        action="store_true",
        help="Skip output tiles if they exist",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use for inference (CPU is used if not set)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size to use during inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of workers to use in the dataloader",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    assert os.path.exists(args.model_fn), f"Model file not found: {args.model_fn}"
    assert args.model_fn.endswith(".ckpt"), "Model file must be .ckpt format"
    assert os.path.exists(args.input_fn), f"Input file not found: {args.input_fn}"

    print("Starting solar inference...")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)

    input_fns = [args.input_fn]
    output_fns = [
        os.path.join(
            args.output_dir,
            os.path.basename(args.input_fn).split("?")[0].replace(".tif", "_solar.tif"),
        )
    ]

    output_fn = output_fns[0]
    if os.path.exists(output_fn):
        if args.skip:
            if args.verbose:
                print(f"Skipping existing file: {output_fn}")
            return
        elif not args.overwrite:
            print(
                f"Output file already exists: {output_fn}. Use --overwrite or --skip."
            )
            return

    device = torch.device(
        f"cuda:{args.gpu}"
        if (args.gpu is not None) and torch.cuda.is_available()
        else "cpu"
    )
    if args.verbose:
        print(f"Using device: {device}")

    task = SegmentationTask.load_from_checkpoint(args.model_fn)
    task.freeze()
    model = task.model
    model = model.eval().to(device)

    ds = InferenceDataset(input_fns, output_fns, transforms=preprocess)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    enumerator = tqdm(dl) if args.verbose else dl

    for batch in enumerator:
        images = batch["image"].to(device)
        t_output_fns = batch["output_fn"]
        t_transforms = batch["transform"]
        batch_size = images.shape[0]

        with torch.inference_mode():
            t_batch_output = model(images).argmax(dim=1).cpu().numpy().astype(np.uint8)

        for i in range(batch_size):
            transform = t_transforms[i]
            output_fn = t_output_fns[i]
            profile = {
                "driver": "GTiff",
                "dtype": "uint8",
                "nodata": 0,
                "width": 4096,
                "height": 4096,
                "count": 1,
                "crs": "EPSG:3857",
                "transform": Affine(*transform),
                "blockxsize": 512,
                "blockysize": 512,
                "tiled": True,
                "compress": "lzw",
                "predictor": 2,
                "interleave": "pixel",
            }

            with rasterio.open(output_fn, "w", **profile) as f:
                f.write(t_batch_output[i], 1)

    if args.verbose:
        print(f"Saved predictions to: {output_fn}")


if __name__ == "__main__":
    args = get_args()
    main(args)
