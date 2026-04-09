# ode

`ode` is a library for forecasting ocean surface currents and sea surface height.
Currently, it pulls HYCOM subsets, prepares local training stores, and trains
a LSTM-PCA forecasting model.  

The current repository workflow is:

1. Pull raw daily HYCOM subsets from THREDDS/NCSS.
2. Merge daily `ssh` and `u-v` files into one training-ready Zarr store.
3. Optionally inspect one timestamp with the notebook-style viewer script.
4. Train the PCA-LSTM model from CLI flags or a JSON config file.

## Features

- Local data pulling by URL or manifest into a reusable cache directory
- HYCOM THREDDS/NCSS pulling with daily range expansion, bbox support, retries, and clear skipped/downloaded reporting
- NetCDF ingestion through `xarray`
- Local Zarr conversion for repeatable training runs
- HYCOM-specific raw-to-training preparation with merge, rename, squeeze, sort, and deduplicate steps
- Sliding-window PyTorch dataset for SSH and surface current fields
- PCA-plus-LSTM forecasting model over leading field components
- Training loop and checkpoint saving
- Notebook-style Python viewer for one HYCOM timestamp
- CLI for pulling, preparation, conversion, and training

## Install

```bash
pip install -e .
```

For tests:

```bash
pip install -e .[dev]
```

## Pull Data Locally

Download individual files directly into a local cache directory:

```bash
ode pull \
  --url https://example.org/ocean/ssh_2024-01-01.nc \
  --url https://example.org/ocean/currents_2024-01-01.nc \
  --output-dir data/raw
```

Or use a manifest when you want reproducible local layouts and checksum verification:

```json
{
  "files": [
    {
      "url": "https://example.org/ocean/ssh_2024-01-01.nc",
      "path": "2024/01/ssh_2024-01-01.nc",
      "sha256": "<expected sha256>"
    }
  ]
}
```

```bash
ode pull \
  --manifest manifests/training-data.json \
  --output-dir data/raw
```

For the HYCOM THREDDS server you provided, use the dedicated NCSS pull path so the package can generate a local NetCDF subset directly from the catalog:

```bash
ode pull-thredds \
  --catalog-url 'https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5' \
  --day 2019-03-03 \
  --north 31.9606 \
  --south 18.0916 \
  --west -98.0 \
  --east -76.4 \
  --output-dir data/raw/hycom
```

If you omit `--variable`, the THREDDS pull defaults to `ssh`, `u`, and `v`, with `u` and `v` taken from the surface layer.
Because HYCOM serves `ssh` as a 2D field and `u`/`v` as 3D fields, the puller writes separate compatible NetCDF files for each day: one `ssh` file and one `u-v` surface-current file.

When a request overlaps files you already downloaded, the CLI will report those files as skipped unless you add `--overwrite`.

For a whole inclusive date range, the puller will request one day at a time and clamp the requested window to the dataset's available dates:

```bash
ode pull-thredds \
  --catalog-url 'https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5' \
  --start-date 2019-01-01 \
  --end-date 2019-03-03 \
  --north 21 \
  --south 20 \
  --west -82 \
  --east -80 \
  --output-dir data/raw/hycom
```

Notes for HYCOM:

- The catalog exposes OPENDAP and NetcdfSubset services, but local training files should be pulled through NetcdfSubset.
- HYCOM explicitly recommends subsetting no more than one day at a time on `ncss.hycom.org`, so the CLI accepts `--day` for that workflow.
- The inclusive `--start-date` and `--end-date` range mode expands to one request per available day and clamps the request to the dataset metadata window.
- THREDDS pulls default to `ssh`, `u`, and `v` when `--variable` is omitted.
- HYCOM `u` and `v` pulls default to `vertCoord=0.0`, so the data pull uses surface currents unless you explicitly override it.
- The current HYCOM source window exposed by this dataset is `2014-04-02` through `2019-03-03`.

## Prepare HYCOM For Training

The raw HYCOM pull layout is split into daily `ssh` files and daily `u-v` surface-current files. Before training, merge them into a single training-ready Zarr store:

```bash
ode prepare-hycom \
  --input-dir data/raw/hycom \
  --start-date 2019-01-01 \
  --end-date 2019-03-03 \
  --output data/processed/hycom_training_2019q1.zarr \
  --chunk time=32 \
  --chunk lat=28 \
  --chunk lon=51
```

This preparation step:

- merges daily `ssh` and `u-v` files
- squeezes the surface `Depth` dimension out of `u` and `v`
- renames HYCOM dimensions from `MT`, `Latitude`, and `Longitude` to `time`, `lat`, and `lon`
- sorts and deduplicates overlapping timestamps between daily pulls
- optionally filters the prepared dataset to a clean inclusive date range inside a larger raw directory

For the current dataset in this repository, this preparation step has been used to produce a store at `data/processed/hycom_training_2019q1.zarr`.

## Inspect One HYCOM Timestamp

The repository includes a notebook-style Python script at `notebooks/view_hycom_snapshot.py` for VS Code's Interactive Window. It opens one daily `ssh` file and one daily `u-v` file for the same day, merges them, and plots:

- SSH with current vectors overlaid as a quiver plot
- surface `u`
- surface `v`

Run it with VS Code cell execution or as a normal Python script after editing `TARGET_DAY` near the top of the file.

## Animate Raw HYCOM Data

If you want an animation of the observed/raw fields instead of the forecast, use the raw-data animation script. It renders `ssh` with `pcolormesh` and overlays `u` plus `v` as quiver arrows without a quiver colorbar:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/animate_raw_hycom.py \
  --input-dir data/raw/hycom \
  --start-date 2019-02-01 \
  --end-date 2019-02-07 \
  --output-path checkpoints/hycom_raw_animation.gif \
  --fps 2
```

## Convert NetCDF to Zarr

Use the generic converter when your training source is not the HYCOM split daily layout handled by `prepare-hycom`:

```bash
ode convert \
  --input data/day_*.nc \
  --output data/cache/training.zarr \
  --chunk time=24 \
  --chunk lat=64 \
  --chunk lon=64
```

## Train The PCA-LSTM Model

You can train directly from CLI flags:

```bash
ode train \
  --zarr-path data/processed/hycom_training_2019q1.zarr \
  --variable ssh \
  --variable u \
  --variable v \
  --time-dim time \
  --spatial-dim lat \
  --spatial-dim lon \
  --pca-components 16 \
  --lstm-hidden-size 64 \
  --lstm-layers 2 \
  --lstm-dropout 0.1 \
  --input-steps 6 \
  --output-steps 1 \
  --batch-size 8 \
  --train-fraction 0.8 \
  --epochs 50 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --device auto \
  --checkpoint-path checkpoints/pca_lstm_hycom_full.pt
```

The repository also includes a reusable JSON config file at `configs/hycom_full_train.json`:

```bash
ode train \
  --config configs/hycom_full_train.json \
  --checkpoint-path checkpoints/pca_lstm_hycom_full.pt
```

The trainer prints the final train and validation losses and saves a checkpoint when `--checkpoint-path` is provided. Checkpoints contain:

- model weights
- epoch-level training history
- the resolved training device
- the training config payload
- the Zarr path used for training

## Inspect One Saved Forecast

Use the checkpoint prediction script to load a saved model, run one forecast window, print the forecast timestamps, and optionally save a quick comparison figure:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/predict_checkpoint.py \
  --checkpoint checkpoints/pca_lstm_hycom_full_rerun.pt \
  --split val \
  --sample-index 0 \
  --forecast-step 0 \
  --figure-path checkpoints/pca_lstm_hycom_full_rerun_prediction.png
```

The figure keeps scalar variables such as `ssh` as heatmaps and combines `u` plus `v` into a single surface-current quiver row for target, prediction, and error.

To inspect one random grid point as a time series, save a second plot that shows the lookback window plus actual versus forecast values at that point:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/predict_checkpoint.py \
  --checkpoint checkpoints/pca_lstm_hycom_full_rerun.pt \
  --split val \
  --sample-index 0 \
  --lead-steps 7 \
  --timeseries-path checkpoints/pca_lstm_hycom_full_rerun_point_timeseries.png \
  --random-seed 7
```

That plot picks one spatial point, shows the lookback history used as model input, and overlays the forecast against the actual future values for each variable.

To produce a 7-day lead forecast from the current 1-day model, roll the checkpoint forward autoregressively and plot the seventh day:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/predict_checkpoint.py \
  --checkpoint checkpoints/pca_lstm_hycom_full_rerun.pt \
  --split val \
  --sample-index 0 \
  --lead-steps 7 \
  --forecast-step 6 \
  --figure-path checkpoints/pca_lstm_hycom_full_rerun_day7_prediction.png
```

That command uses each predicted day as part of the next input window until it reaches a 7-day lead time.

The current trainer records one average train loss and one average validation loss per epoch. On the current `hycom_training_2019q1.zarr` store, the full config run completed 50 epochs on `mps` and reached a final train loss near `0.0044` and final validation loss near `0.0160`.

## Assumptions

- The training model projects full fields onto the top N principal components, trains an LSTM on coefficient sequences, and reconstructs predicted fields back to the original grid.
- The model expects exactly two spatial dimensions, such as `lat` and `lon` or `y` and `x`.
- Training variables should share the same grid and time axis.
- NetCDF files should be combinable by coordinates.
- Zarr storage is used as a local cache and training source when provided.
- The `pull` command accepts plain URLs and `file://` URLs, and manifest paths must stay relative to the chosen output directory.
- The `pull-thredds` command resolves a THREDDS NetcdfSubset endpoint from the catalog URL and writes local NetCDF subset files for later conversion to Zarr.

## Test

```bash
python -m pytest -q
```
