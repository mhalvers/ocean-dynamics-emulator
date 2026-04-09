from __future__ import annotations

import argparse
from pathlib import Path

from ode.data.hycom import prepare_hycom_dataset
from ode.data.visualization import animate_surface_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Animate raw HYCOM SSH and surface current data.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw HYCOM ssh and u-v NetCDF files.")
    parser.add_argument("--output-path", required=True, help="Output GIF path.")
    parser.add_argument("--start-date", default=None, help="Optional inclusive start date filter.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive end date filter.")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for the GIF.")
    parser.add_argument("--quiver-stride", type=int, default=None, help="Optional stride for current vectors. Defaults to an automatic value.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepared = prepare_hycom_dataset(
        args.input_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    saved_path = animate_surface_dataset(
        prepared.dataset,
        Path(args.output_path),
        fps=args.fps,
        quiver_stride=args.quiver_stride,
    )
    print(f"Animated {prepared.unique_timestamps} raw timesteps")
    print(f"Saved raw-data animation to {saved_path}")


if __name__ == "__main__":
    main()