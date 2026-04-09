from __future__ import annotations

import argparse
from pathlib import Path

from ode.config import DataConfig, ModelConfig, OptimizationConfig, TrainingConfig, load_training_config
from ode.data.hycom import prepare_hycom_zarr
from ode.data.netcdf import convert_netcdf_to_zarr
from ode.data.pull import pull_data
from ode.data.thredds import DEFAULT_THREDDS_VARIABLES, ThreddsSubsetRequest, pull_thredds_catalog, resolve_thredds_request_window
from ode.training.engine import fit, save_checkpoint


def _print_download_results(results) -> None:
    downloaded_count = 0
    skipped_count = 0
    for result in results:
        status = getattr(result, "status", "downloaded")
        path = getattr(result, "path", result)
        if status == "downloaded":
            downloaded_count += 1
            print(f"Pulled {path}")
        else:
            skipped_count += 1
            print(f"Skipped existing {path}")

    if skipped_count and not downloaded_count:
        print("No new files were downloaded. Use --overwrite to re-download existing files.")


def _parse_chunks(values: list[str]) -> dict[str, int]:
    chunks: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid chunk specification '{value}'. Use name=size.")
        name, size = value.split("=", maxsplit=1)
        chunks[name] = int(size)
    return chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SSH and surface current forecasting models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull_parser = subparsers.add_parser("pull", help="Pull remote or file-based data into a local cache directory.")
    pull_parser.add_argument("--url", action="append", default=[], help="URL to download into the output directory.")
    pull_parser.add_argument("--manifest", default=None, help="JSON manifest with url, optional path, and optional sha256 fields.")
    pull_parser.add_argument("--output-dir", required=True, help="Local directory used as the download root.")
    pull_parser.add_argument("--overwrite", action="store_true", help="Overwrite files that already exist.")

    thredds_parser = subparsers.add_parser("pull-thredds", help="Pull a NetCDF subset from a THREDDS NetcdfSubset catalog.")
    thredds_parser.add_argument("--catalog-url", required=True, help="THREDDS catalog HTML or XML URL.")
    thredds_parser.add_argument("--dataset-id", default=None, help="Optional THREDDS dataset id when the catalog contains multiple datasets.")
    thredds_parser.add_argument("--output-dir", required=True, help="Local directory used as the download root.")
    thredds_parser.add_argument("--output-name", default=None, help="Optional output filename for the downloaded NetCDF subset.")
    thredds_parser.add_argument("--variable", action="append", default=[], help="Variable to include in the subset request. Defaults to ssh, u, and v.")
    thredds_parser.add_argument("--day", default=None, help="UTC day to download as YYYY-MM-DD. HYCOM NCSS recommends one day per request.")
    thredds_parser.add_argument("--start-date", default=None, help="Inclusive UTC start date for one-day-at-a-time pulls, clamped to available data.")
    thredds_parser.add_argument("--end-date", default=None, help="Inclusive UTC end date for one-day-at-a-time pulls, clamped to available data.")
    thredds_parser.add_argument("--time", default=None, help="Single ISO-8601 UTC timestamp to download.")
    thredds_parser.add_argument("--north", type=float, default=None, help="North latitude bound.")
    thredds_parser.add_argument("--south", type=float, default=None, help="South latitude bound.")
    thredds_parser.add_argument("--east", type=float, default=None, help="East longitude bound.")
    thredds_parser.add_argument("--west", type=float, default=None, help="West longitude bound.")
    thredds_parser.add_argument("--vert-coord", default=None, help="Optional vertical coordinate override. Defaults to 0.0 for HYCOM u and v surface-current pulls.")
    thredds_parser.add_argument("--accept", default="netcdf4", help="NCSS output format. Defaults to netcdf4.")
    thredds_parser.add_argument("--horiz-stride", type=int, default=1, help="Horizontal stride for NCSS requests.")
    thredds_parser.add_argument("--overwrite", action="store_true", help="Overwrite files that already exist.")

    prepare_parser = subparsers.add_parser("prepare-hycom", help="Merge raw HYCOM ssh and surface-current files into a training-ready Zarr store.")
    prepare_parser.add_argument("--input-dir", required=True, help="Directory containing raw HYCOM ssh and u-v daily NetCDF files.")
    prepare_parser.add_argument("--output", required=True, help="Output Zarr path for the prepared training dataset.")
    prepare_parser.add_argument("--start-date", default=None, help="Optional inclusive start date filter for prepared HYCOM daily files.")
    prepare_parser.add_argument("--end-date", default=None, help="Optional inclusive end date filter for prepared HYCOM daily files.")
    prepare_parser.add_argument("--chunk", action="append", default=[], help="Chunk specification, e.g. time=32.")
    prepare_parser.add_argument("--engine", default=None, help="Optional xarray backend engine.")

    convert_parser = subparsers.add_parser("convert", help="Convert NetCDF files into a local Zarr store.")
    convert_parser.add_argument("--input", nargs="+", required=True, help="Input NetCDF paths or globs.")
    convert_parser.add_argument("--output", required=True, help="Output Zarr store path.")
    convert_parser.add_argument("--chunk", action="append", default=[], help="Chunk specification, e.g. time=24.")
    convert_parser.add_argument("--engine", default=None, help="Optional xarray backend engine.")

    train_parser = subparsers.add_parser("train", help="Train the PCA-LSTM forecaster.")
    train_parser.add_argument("--config", default=None, help="Optional JSON config file.")
    train_parser.add_argument("--input", nargs="*", default=[], help="Input NetCDF or Zarr paths.")
    train_parser.add_argument("--zarr-path", default=None, help="Local Zarr store used for training.")
    train_parser.add_argument("--variable", action="append", default=[], help="Training variable name.")
    train_parser.add_argument("--time-dim", default="time", help="Dataset time dimension.")
    train_parser.add_argument("--spatial-dim", action="append", default=[], help="Spatial dimensions in order.")
    train_parser.add_argument("--input-steps", type=int, default=6, help="Number of input timesteps.")
    train_parser.add_argument("--output-steps", type=int, default=1, help="Number of forecast timesteps.")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    train_parser.add_argument("--num-workers", type=int, default=0, help="PyTorch dataloader workers.")
    train_parser.add_argument("--train-fraction", type=float, default=0.8, help="Leading fraction of windows used for training.")
    train_parser.add_argument("--chunk", action="append", default=[], help="Chunk specification, e.g. time=24.")
    train_parser.add_argument("--engine", default=None, help="Optional xarray backend engine.")
    train_parser.add_argument("--pca-components", type=int, default=32, help="Number of leading principal components used as the LSTM state input.")
    train_parser.add_argument("--lstm-hidden-size", type=int, default=128, help="Hidden size of the PCA-sequence LSTM.")
    train_parser.add_argument("--lstm-layers", type=int, default=2, help="Number of stacked LSTM layers.")
    train_parser.add_argument("--lstm-dropout", type=float, default=0.0, help="Dropout between LSTM layers when lstm-layers > 1.")
    train_parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=1e-5, help="Optimizer weight decay.")
    train_parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or mps.")
    train_parser.add_argument("--checkpoint-path", default=None, help="Optional checkpoint output path.")

    return parser


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    if args.config:
        return load_training_config(args.config)

    variables = tuple(args.variable) if args.variable else ("ssh", "u", "v")
    spatial_dims = tuple(args.spatial_dim) if args.spatial_dim else None
    return TrainingConfig(
        data=DataConfig(
            paths=list(args.input),
            zarr_path=args.zarr_path,
            variables=variables,
            time_dim=args.time_dim,
            spatial_dims=spatial_dims,
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_fraction=args.train_fraction,
            chunks=_parse_chunks(args.chunk),
            engine=args.engine,
        ),
        model=ModelConfig(
            pca_components=args.pca_components,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
        ),
        optimization=OptimizationConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        ),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pull":
        results = pull_data(
            output_dir=args.output_dir,
            urls=args.url,
            manifest_path=args.manifest,
            overwrite=args.overwrite,
        )
        _print_download_results(results)
        return

    if args.command == "pull-thredds":
        request = ThreddsSubsetRequest(
            catalog_url=args.catalog_url,
            dataset_id=args.dataset_id,
            output_dir=args.output_dir,
            output_name=args.output_name,
            variables=tuple(args.variable) or DEFAULT_THREDDS_VARIABLES,
            day=args.day,
            start_date=args.start_date,
            end_date=args.end_date,
            time=args.time,
            north=args.north,
            south=args.south,
            east=args.east,
            west=args.west,
            vert_coord=args.vert_coord,
            accept=args.accept,
            horiz_stride=args.horiz_stride,
            overwrite=args.overwrite,
        )
        if args.start_date and args.end_date:
            window = resolve_thredds_request_window(request)
            if window.effective_start is not None and window.effective_end is not None:
                requested_text = f"{args.start_date} to {args.end_date}"
                effective_text = f"{window.effective_start.isoformat()} to {window.effective_end.isoformat()}"
                if requested_text != effective_text:
                    print(f"Clamped requested range {requested_text} to available range {effective_text}.")

        results = pull_thredds_catalog(request)
        _print_download_results(results)
        return

    if args.command == "prepare-hycom":
        output_path, prepared = prepare_hycom_zarr(
            args.input_dir,
            args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            chunks=_parse_chunks(args.chunk),
            engine=args.engine,
        )
        print(f"Prepared HYCOM training store at {output_path}")
        print(
            f"Merged {len(prepared.ssh_paths)} daily ssh files and {len(prepared.uv_paths)} daily u-v files "
            f"into {prepared.unique_timestamps} unique timestamps."
        )
        return

    if args.command == "convert":
        chunks = _parse_chunks(args.chunk)
        output_path = convert_netcdf_to_zarr(args.input, args.output, chunks=chunks, engine=args.engine)
        print(f"Wrote Zarr store to {output_path}")
        return

    if args.command == "train":
        config = _build_training_config(args)
        result = fit(config)
        print(f"Training complete on {result.device}. Final train loss: {result.history['train_loss'][-1]:.6f}")
        if result.history["val_loss"]:
            print(f"Final val loss: {result.history['val_loss'][-1]:.6f}")
        if args.checkpoint_path:
            checkpoint = save_checkpoint(result, config, Path(args.checkpoint_path))
            print(f"Saved checkpoint to {checkpoint}")
        if result.zarr_path:
            print(f"Training data store: {result.zarr_path}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
