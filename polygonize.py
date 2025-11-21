# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import multiprocessing
import os

import fiona
import fiona.transform
import rasterio
import rasterio.features
import shapely.geometry
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    """Set up the argument parser.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing model prediction GeoTIFFs (*_solar.tif and/or *_wind.tif)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to write GeoJSON results to",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they exist",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=10000.0,
        help="Minimum area in square meters to include in output (for solar only)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()
    return args


def wrapper(args):
    """Wrapper function for multiprocessing."""
    return get_features_from_fn(*args)


def get_features_from_fn(input_fn, output_fn, min_area, overwrite, is_wind):
    """Extract polygon features from a prediction raster file.

    Args:
        input_fn: Path to input GeoTIFF
        output_fn: Path to output GeoJSON
        min_area: Minimum area in square meters
        overwrite: Whether to overwrite existing files
        is_wind: If True, extract centroids; if False, extract polygons

    Returns:
        None
    """
    if os.path.exists(output_fn) and not overwrite:
        return

    features = []
    with rasterio.open(input_fn) as f:
        assert f.crs.to_string() == "EPSG:3857", f"Expected EPSG:3857, got {f.crs}"
        for feature in rasterio.features.dataset_features(f, bidx=1, geographic=False):
            if feature["properties"]["val"] == 1:
                geom = fiona.transform.transform_geom(
                    "epsg:3857", "epsg:6933", feature["geometry"]
                )
                shape = shapely.geometry.shape(geom)
                area = shape.area
                del feature["bbox"]
                del feature["id"]
                del feature["properties"]["val"]

                if area >= min_area and not is_wind:
                    feature["geometry"] = fiona.transform.transform_geom(
                        "epsg:6933", "epsg:3857", geom
                    )
                    feature["properties"]["area"] = area
                    feature["properties"]["filename"] = os.path.basename(input_fn)
                    features.append(feature)
                elif is_wind:
                    # For wind, use centroid as a Point
                    centroid = shape.centroid
                    feature["geometry"] = fiona.transform.transform_geom(
                        "epsg:6933", "epsg:3857", shapely.geometry.mapping(centroid)
                    )
                    feature["properties"]["area"] = area
                    feature["properties"]["filename"] = os.path.basename(input_fn)
                    features.append(feature)

    if len(features) == 0:
        print(f"No features found for {input_fn}")
        return

    schema = {
        "geometry": "Point" if is_wind else "Polygon",
        "properties": {
            "filename": "str",
            "area": "float",
        },
    }

    with fiona.open(
        output_fn, "w", driver="GeoJSON", crs="EPSG:3857", schema=schema
    ) as f:
        f.writerecords(features)


def main(args: argparse.Namespace) -> None:
    """Main function for polygonization.

    Args:
        args: Parsed command-line arguments
    """
    assert os.path.exists(args.input_dir) and os.path.isdir(args.input_dir), (
        f"Input directory not found: {args.input_dir}"
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    elif not args.overwrite:
        existing_files = [
            f for f in os.listdir(args.output_dir) if f.endswith(".geojson")
        ]
        if existing_files:
            print(
                f"Output directory contains {len(existing_files)} existing files. Use --overwrite to replace them."
            )

    if args.verbose:
        print("Starting polygonization...")

    # Find and categorize all input files
    tasks = []
    solar_count = 0
    wind_count = 0

    for fn in sorted(os.listdir(args.input_dir)):
        if fn.endswith("_solar.tif"):
            input_fn = os.path.join(args.input_dir, fn)
            output_fn = os.path.join(
                args.output_dir, fn.replace("_solar.tif", "_solar.geojson")
            )
            tasks.append((input_fn, output_fn, args.min_area, args.overwrite, False))
            solar_count += 1
        elif fn.endswith("_wind.tif"):
            input_fn = os.path.join(args.input_dir, fn)
            output_fn = os.path.join(
                args.output_dir, fn.replace("_wind.tif", "_wind.geojson")
            )
            tasks.append((input_fn, output_fn, args.min_area, args.overwrite, True))
            wind_count += 1

    if len(tasks) == 0:
        print(
            f"No prediction files (*_solar.tif or *_wind.tif) found in {args.input_dir}"
        )
        return

    if args.verbose:
        print(f"Found {solar_count} solar and {wind_count} wind prediction file(s)")
        print(
            f"Running parallel polygonization with {args.num_workers} worker(s) (min area: {args.min_area} m²)"
        )
        if solar_count > 0:
            print("  - Solar: extracting polygons")
        if wind_count > 0:
            print("  - Wind: extracting centroids")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(wrapper, tasks),
                total=len(tasks),
                disable=not args.verbose,
            )
        )

    if args.verbose:
        print(f"Polygonization complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    args = get_args()
    main(args)
