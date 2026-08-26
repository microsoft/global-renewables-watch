# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os

import numpy as np
import rasterio
import torch
import torchvision
from tqdm import tqdm

from src.wind.data.tile_dataset import TileInferenceDataset
from src.wind.models.fcn8_resnet import FCN8
from src.wind.models.unet import UnetModel

NUM_WORKERS = 8
CHIP_SIZE = 256
PADDING = 4
assert PADDING % 2 == 0, "PADDING must be even"
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING


def get_args() -> argparse.Namespace:
    """Set up the argument parser.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-fn",
        type=str,
        required=True,
        help="Path to the raster to run the model on",
    )
    parser.add_argument(
        "--model-fn",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth.tar format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
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
        default=256,
        help="Batch size to use during inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers to use in the dataloader",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Threshold to use for classification",
    )
    parser.add_argument(
        "--sample-stats",
        action="store_true",
        help="Calculate the means and stds on the fly",
    )
    parser.add_argument(
        "--input-means",
        type=str,
        default="70.3314, 75.5732546, 49.84334",
        help="Per channel means to use (comma delimited)",
    )
    parser.add_argument(
        "--input-stds",
        type=str,
        default="35.947722, 22.148079, 19.78486",
        help="Per channel stds to use (comma delimited)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    assert os.path.exists(args.input_fn), f"Input file not found: {args.input_fn}"
    assert os.path.exists(args.model_fn), f"Model file not found: {args.model_fn}"

    if args.verbose:
        print("Starting wind turbine inference...")

    device = torch.device(
        f"cuda:{args.gpu}"
        if (args.gpu is not None) and torch.cuda.is_available()
        else "cpu"
    )
    if args.verbose:
        print(f"Using device: {device}")

    if not args.sample_stats:
        input_means = list(map(float, args.input_means.split(",")))
        input_stds = list(map(float, args.input_stds.split(",")))
    else:
        raise NotImplementedError("Sample stats calculation is not yet implemented")
    checkpoint = torch.load(args.model_fn, map_location="cpu")
    opts = checkpoint["params"]

    if opts["model"] == "unet":
        model = UnetModel(opts)
    elif opts["model"] == "fcn":
        model = FCN8()
    else:
        raise NotImplementedError(
            f"Model type '{opts['model']}' not supported. Available options: unet, fcn"
        )

    model.load_state_dict(checkpoint["model"])
    model = model.eval().to(device)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    output_fn = os.path.join(
        args.output_dir,
        os.path.basename(args.input_fn).split("?")[0].replace(".tif", "_wind.tif"),
    )

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

    with rasterio.open(args.input_fn) as f:
        input_width, input_height = f.width, f.height
        input_profile = f.profile.copy()

    transform_set = torchvision.transforms.Compose(
        [
            lambda x: np.rollaxis(x.astype(np.float32), 2, 0),
            lambda x: torch.from_numpy(x),
            torchvision.transforms.Normalize(input_means, input_stds),
        ]
    )

    dataset = TileInferenceDataset(
        args.input_fn,
        chip_size=CHIP_SIZE,
        stride=CHIP_STRIDE,
        transform=transform_set,
        verbose=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    output = np.zeros((input_height, input_width), dtype=np.uint8)
    for i, (data, coords) in enumerate(
        tqdm(dataloader) if args.verbose else dataloader
    ):
        data = data.to(device)
        with torch.inference_mode():
            if opts["model"] == "fcn":
                t_output = model(data).sigmoid()
                t_output = (t_output[:, 0, :, :] > args.threshold).cpu().numpy()
            else:
                t_output = model(data).argmax(axis=1).cpu().numpy()

        for j in range(t_output.shape[0]):
            y, x = coords[j]
            output[
                y + HALF_PADDING : y + CHIP_SIZE - HALF_PADDING,
                x + HALF_PADDING : x + CHIP_SIZE - HALF_PADDING,
            ] = t_output[
                j,
                HALF_PADDING:-HALF_PADDING,
                HALF_PADDING:-HALF_PADDING,
            ]

    output_profile = input_profile.copy()
    output_profile["dtype"] = "uint8"
    output_profile["count"] = 1
    output_profile["nodata"] = 0
    with rasterio.open(output_fn, "w", **output_profile) as f:
        f.write(output, 1)

    if args.verbose:
        print(f"Saved predictions to: {output_fn}")


if __name__ == "__main__":
    args = get_args()
    main(args)
