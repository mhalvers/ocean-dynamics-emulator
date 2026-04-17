# ode

`ode` is a library for forecasting ocean surface currents and sea surface height.
Currently, it pulls HYCOM subsets, prepares local training stores, and trains
a normalized PCA-LSTM forecasting model.  

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
- Channel-normalized PCA-plus-LSTM forecasting model over leading field components
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
  --start-date 2014-04-02 \
  --end-date 2019-03-03 \
  --output data/processed/hycom_training_2014-04-02_to_2019-03-03.zarr \
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

For the current dataset in this repository, the training store currently used by the canonical config is `data/processed/hycom_training_2014-04-02_to_2019-03-03.zarr`, which contains the full available daily SSH, u, and v fields from the current HYCOM `expt_32.5` source on the repository's training grid.

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
  --zarr-path data/processed/hycom_training_2014-04-02_to_2019-03-03.zarr \
  --variable ssh \
  --variable u \
  --variable v \
  --target-variable ssh \
  --time-dim time \
  --spatial-dim lat \
  --spatial-dim lon \
  --pca-components 32 \
  --lstm-hidden-size 128 \
  --lstm-layers 2 \
  --lstm-dropout 0.1 \
  --input-steps 14 \
  --output-steps 7 \
  --batch-size 32 \
  --train-fraction 0.8 \
  --epochs 100 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --device auto \
  --checkpoint-path checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03.pt
```

The repository also includes a reusable JSON config file at `configs/hycom_full_train.json`:

```bash
ode train \
  --config configs/hycom_full_train.json \
  --checkpoint-path checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03.pt
```

The trainer prints the final train and validation losses and saves a checkpoint when `--checkpoint-path` is provided. The canonical config now trains a direct 7-step model from 14 days of lookback, uses `ssh`, `u`, and `v` as inputs, predicts only `ssh`, and normalizes each variable before PCA fitting and LSTM forecasting. Checkpoints contain:

- model weights
- epoch-level training history
- the resolved training device
- the training config payload
- the Zarr path used for training

The canonical config also trains `ssh` as a residual over the last observed `ssh` input frame. The model still outputs absolute `ssh`, but its learned target is the forecast correction over the persistence baseline.

The training config also supports date-based splits. When `train_end_date`, `val_start_date`, or `val_end_date` are set, training windows are chosen by forecast target dates instead of a leading sample fraction. The canonical config now uses:

- `train_end_date=2018-11-18`
- `val_start_date=2018-11-19`
- `val_end_date=2018-12-18`

That keeps the validation target window recent while leaving the December 2018 to January 2019 benchmark targets outside the training and validation split.

An autoregressive decoder variant of the same PCA-LSTM backbone is available through `configs/hycom_autoregressive_decoder_train.json`. It keeps the same PCA encoder idea but replaces the one-shot 7-step coefficient readout with a stepwise decoder LSTM in target PCA space.

## Track Experiments

The repository includes a lightweight local experiment-tracking workflow built around four pieces:

- a fixed benchmark spec at `benchmarks/ssh_standard_windows_v1.json`
- a benchmark evaluator at `scripts/evaluate_benchmark.py`
- a tracked training wrapper at `scripts/run_tracked_experiment.py`
- a local run registry at `experiments/registry.jsonl`

The benchmark spec defines exact input start dates on a standardized evaluation store. That means experiment comparisons do not drift when the training store length changes.

For future runs, prefer the built-in tracked workflow from the main CLI so training, benchmark evaluation, and registry logging happen in one command:

```bash
ode experiment \
  --config configs/hycom_full_train.json \
  --benchmark benchmarks/ssh_standard_windows_v1.json
```

That command trains the model, writes a dedicated run directory under `experiments/runs/`, evaluates the saved checkpoint on the fixed benchmark, and appends one row to `experiments/registry.jsonl`.

To evaluate an existing checkpoint on the fixed benchmark windows:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/evaluate_benchmark.py \
  --checkpoint checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03.pt \
  --benchmark benchmarks/ssh_standard_windows_v1.json \
  --output experiments/manual_eval_ssh_target.json
```

To train a new run, save its checkpoint and training summary into a dedicated run directory, evaluate it on the benchmark, and append one record to the registry in one step:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/run_tracked_experiment.py \
  --config configs/hycom_full_train.json \
  --benchmark benchmarks/ssh_standard_windows_v1.json
```

The standalone script remains available when you want the same tracked workflow outside the `ode` console entry point.

That command creates a run directory under `experiments/runs/<run_id>/` with:

- `checkpoint.pt`
- `config.json`
- `benchmark.json`
- `training_summary.json`
- `benchmark_metrics.json`
- `run_manifest.json`

The registry file records the run id, git commit, training store, model hyperparameters, final train and validation loss, and benchmark metrics such as overall benchmark MSE and SSH benchmark MSE. The generated run directories and registry are local outputs and are ignored by git.

## Inspect One Saved Forecast

Use the checkpoint prediction script to load a saved model, run one forecast window, print the forecast timestamps, and optionally save a quick comparison figure:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/predict_checkpoint.py \
  --checkpoint checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03.pt \
  --split val \
  --sample-index 0 \
  --forecast-step 6 \
  --figure-path checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03_day7_prediction.png
```

When the checkpoint predicts only `ssh`, the figure contains only the SSH target, prediction, and error panels. If a checkpoint predicts `u` and `v` as targets as well, the figure adds the surface-current quiver row.

To inspect one random grid point as a time series, save a second plot that shows the lookback window plus actual versus forecast values at that point:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/predict_checkpoint.py \
  --checkpoint checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03.pt \
  --split val \
  --sample-index 0 \
  --timeseries-path checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03_point_timeseries.png \
  --random-seed 7
```

That plot picks one spatial point, shows the lookback history used as model input, and overlays the forecast against the actual future values for each variable.

To inspect the seventh day from the direct 7-step model, plot `forecast-step 6` from the saved checkpoint:

```bash
/Users/mark/projects/ocean_dynamics_emulator/.venv/bin/python scripts/predict_checkpoint.py \
  --checkpoint checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03.pt \
  --split val \
  --sample-index 0 \
  --forecast-step 6 \
  --figure-path checkpoints/pca_lstm_ssh_target_direct7_normalized_pca32_hidden128_100epochs_2014-04-02_to_2019-03-03_day7_prediction.png
```

If you request more than 7 steps, the script can only roll the checkpoint forward autoregressively when the checkpoint predicts the same variables it consumes as inputs. The canonical SSH-only-target config does not support that longer autoregressive rollout.

The current trainer records one average train loss and one average validation loss per epoch. Final metrics depend on the selected store span, lookback window, and checkpoint settings.

## Experimental Findings: LSTM Architectures

Over a series of systematic experiments, we evaluated various LSTM improvements to reduce forecast error. Here are the key findings:

### Baseline Performance

A **minimal 1-layer PCA-LSTM with direct 7-step output** (no autoregressive, no scheduled sampling) achieved:
- **SSH RMSE: 0.0830**
- Skill vs persistence: +0.21

This serves as the vanilla baseline for all improvements.

### Scheduled Sampling for Autoregressive Decoder

Implementing an autoregressive decoder with scheduled sampling significantly improved results:
- **SSH RMSE: 0.0749** (-9.8% vs baseline, **+16.1% vs naive autoregressive**)
- Skill vs persistence: +0.36 (notably better than persistence)
- Teacher forcing ratio schedule: 1.0 → 0.2 over 100 epochs
- Key insight: Gradually transitioning from ground-truth targets to model predictions during training reduces exposure bias and improves generalization

### Capacity Experiments

Increasing model capacity (PCA 64 components, hidden size 192) with the same scheduled sampling:
- **SSH RMSE: 0.0823** (worse than baseline scheduled sampling)
- Skill vs persistence: +0.22
- Key insight: The bottleneck was not model capacity but rather the training procedure itself; over-parameterization hurt performance when the training procedure wasn't optimized

### Residual Three-Layer Encoder

Implementing a three-layer encoder with skip connections (inspired by literature on residual LSTM architectures):
- **SSH RMSE: 0.1845** (-146% regression vs baseline)
- Skill vs persistence: -0.36 (significantly worse)
- Key insight: The residual topology that works for deeper networks was not effective in this PCA-compressed latent space; the added complexity introduced optimization difficulties

### Summary

For this ocean forecasting task, **scheduled sampling for autoregressive decoders is the most effective improvement**, reducing SSH RMSE from 0.0892 (naive autoregressive) to 0.0749 (with scheduling). The key finding is that **training procedure has greater impact than model capacity or architectural complexity**.

The results suggest that:
1. Exposure bias in autoregressive models is critical—scheduled sampling directly addresses it
2. Capacity increases and complex residual architectures provide no benefit and can hurt when the training procedure isn't adjusted accordingly
3. Spatial field correlations in PCA-compressed space are limited—full spatial field models (ConvLSTM) should be explored next

Detailed comparison table available in [experiments/LSTM_COMPARISON.md](experiments/LSTM_COMPARISON.md)

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
