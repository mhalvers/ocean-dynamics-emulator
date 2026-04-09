from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ode.inference import format_time_value, plot_point_timeseries, plot_prediction, predict_window


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one forecast from a saved ode checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved training checkpoint.")
    parser.add_argument("--split", default="val", choices=("train", "val", "all"), help="Dataset split used to choose the forecast window.")
    parser.add_argument("--sample-index", type=int, default=0, help="Window index within the selected split. Negative indices count from the end.")
    parser.add_argument("--lead-steps", type=int, default=None, help="Number of forecast steps to roll forward autoregressively. Defaults to the checkpoint training horizon.")
    parser.add_argument("--forecast-step", type=int, default=0, help="Forecast step to visualize within the rolled-out forecast sequence.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda, or mps.")
    parser.add_argument("--figure-path", default=None, help="Optional output path for a prediction figure.")
    parser.add_argument("--timeseries-path", default=None, help="Optional output path for a pointwise time-series plot.")
    parser.add_argument("--random-seed", type=int, default=None, help="Optional RNG seed used when choosing the random spatial point for the time-series plot.")
    parser.add_argument("--show", action="store_true", help="Display the prediction figure interactively.")
    return parser


def _format_times(values: np.ndarray) -> str:
    return ", ".join(format_time_value(value) for value in values)


def main() -> None:
    args = build_parser().parse_args()
    prediction = predict_window(
        args.checkpoint,
        split=args.split,
        sample_index=args.sample_index,
        lead_steps=args.lead_steps,
        device=args.device,
    )

    print(f"Checkpoint: {prediction.checkpoint_path}")
    print(f"Data store: {prediction.data_store or 'NetCDF inputs'}")
    print(
        f"Split: {prediction.split} | sample {prediction.relative_sample_index} "
        f"(absolute index {prediction.absolute_sample_index})"
    )
    print(f"Lead steps: {prediction.lead_steps}")
    print(f"Input times: {_format_times(prediction.input_times)}")
    print(f"Target times: {_format_times(prediction.target_times)}")
    print(f"Prediction MSE: {prediction.mse:.6f}")

    if args.figure_path or args.show:
        saved_path = plot_prediction(
            prediction,
            forecast_step=args.forecast_step,
            figure_path=Path(args.figure_path) if args.figure_path else None,
            show=args.show,
        )
        if saved_path is not None:
            print(f"Saved figure to {saved_path}")

    if args.timeseries_path:
        timeseries_path, selection = plot_point_timeseries(
            prediction,
            seed=args.random_seed,
            figure_path=Path(args.timeseries_path),
        )
        coordinate_text = ", ".join(
            f"{dim}={format_time_value(value)}" for dim, value in zip(prediction.spatial_dims, selection.point_coordinates)
        )
        print(f"Random point index: {selection.point_index[0]}, {selection.point_index[1]}")
        print(f"Random point coordinates: {coordinate_text}")
        if timeseries_path is not None:
            print(f"Saved point time series to {timeseries_path}")

if __name__ == "__main__":
    main()