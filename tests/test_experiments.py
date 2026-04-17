from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from ode.config import DataConfig, ModelConfig, OptimizationConfig, TrainingConfig
from ode.experiments import evaluate_checkpoint_against_benchmark, record_existing_checkpoint, run_tracked_experiment
from ode.training.engine import fit, save_checkpoint


def _write_datetime_netcdf(path: Path) -> None:
    time = np.array([np.datetime64("2019-01-01") + np.timedelta64(offset, "D") for offset in range(12)])
    lat = np.linspace(-1.0, 1.0, 4)
    lon = np.linspace(120.0, 122.0, 5)
    base = np.arange(time.size * lat.size * lon.size, dtype=np.float32).reshape(time.size, lat.size, lon.size)
    dataset = xr.Dataset(
        data_vars={
            "ssh": (("time", "lat", "lon"), base),
            "u": (("time", "lat", "lon"), base * 0.1),
            "v": (("time", "lat", "lon"), base * -0.1),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    dataset.to_netcdf(path)


def test_evaluate_checkpoint_against_benchmark_returns_summary(tmp_path) -> None:
    netcdf_path = tmp_path / "benchmark.nc"
    zarr_path = tmp_path / "benchmark.zarr"
    checkpoint_path = tmp_path / "benchmark.pt"
    benchmark_path = tmp_path / "benchmark.json"
    _write_datetime_netcdf(netcdf_path)

    config = TrainingConfig(
        data=DataConfig(
            paths=[str(netcdf_path)],
            zarr_path=str(zarr_path),
            variables=("ssh", "u", "v"),
            input_variables=("ssh", "u", "v"),
            target_variables=("ssh",),
            time_dim="time",
            spatial_dims=("lat", "lon"),
            input_steps=3,
            output_steps=2,
            batch_size=2,
            train_fraction=0.75,
            chunks={"time": 2},
        ),
        model=ModelConfig(pca_components=4, lstm_hidden_size=8, lstm_layers=1, lstm_dropout=0.0),
        optimization=OptimizationConfig(epochs=1, learning_rate=1e-3, device="cpu"),
    )

    result = fit(config)
    save_checkpoint(result, config, checkpoint_path)
    benchmark_path.write_text(
        json.dumps(
            {
                "name": "synthetic_ssh_benchmark",
                "zarr_path": str(zarr_path),
                "lead_steps": 2,
                "windows": [{"input_start": "2019-01-05", "label": "window_a"}, {"input_start": "2019-01-06"}],
            }
        )
    )

    metrics = evaluate_checkpoint_against_benchmark(checkpoint_path, benchmark_path, device="cpu")

    assert metrics["benchmark_name"] == "synthetic_ssh_benchmark"
    assert metrics["summary"]["window_count"] == 2
    assert "ssh" in metrics["summary"]["variables"]
    assert metrics["summary"]["variables"]["ssh"]["per_lead"][0]["lead_step"] == 1
    assert metrics["windows"][0]["label"] == "window_a"


def test_run_tracked_experiment_writes_run_artifacts(tmp_path) -> None:
    netcdf_path = tmp_path / "tracked.nc"
    zarr_path = tmp_path / "tracked.zarr"
    config_path = tmp_path / "config.json"
    benchmark_path = tmp_path / "benchmark.json"
    runs_dir = tmp_path / "experiments" / "runs"
    registry_path = tmp_path / "experiments" / "registry.jsonl"
    _write_datetime_netcdf(netcdf_path)

    config = {
        "data": {
            "paths": [str(netcdf_path)],
            "zarr_path": str(zarr_path),
            "variables": ["ssh", "u", "v"],
            "input_variables": ["ssh", "u", "v"],
            "target_variables": ["ssh"],
            "time_dim": "time",
            "spatial_dims": ["lat", "lon"],
            "input_steps": 3,
            "output_steps": 2,
            "batch_size": 2,
            "train_fraction": 0.75,
            "chunks": {"time": 2}
        },
        "model": {
            "pca_components": 4,
            "lstm_hidden_size": 8,
            "lstm_layers": 1,
            "lstm_dropout": 0.0
        },
        "optimization": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "device": "cpu"
        }
    }
    config_path.write_text(json.dumps(config))
    benchmark_path.write_text(
        json.dumps(
            {
                "name": "synthetic_ssh_benchmark",
                "zarr_path": str(zarr_path),
                "lead_steps": 2,
                "windows": [{"input_start": "2019-01-05"}, {"input_start": "2019-01-06"}],
            }
        )
    )

    manifest = run_tracked_experiment(
        config_path,
        benchmark_path=benchmark_path,
        runs_dir=runs_dir,
        registry_path=registry_path,
        device="cpu",
        run_id="synthetic-run",
    )

    run_dir = runs_dir / "synthetic-run"
    assert manifest["run_id"] == "synthetic-run"
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "benchmark.json").exists()
    assert (run_dir / "training_summary.json").exists()
    assert (run_dir / "benchmark_metrics.json").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert registry_path.exists()
    assert "synthetic-run" in registry_path.read_text()


def test_record_existing_checkpoint_writes_run_artifacts(tmp_path) -> None:
    netcdf_path = tmp_path / "existing.nc"
    zarr_path = tmp_path / "existing.zarr"
    checkpoint_path = tmp_path / "existing.pt"
    config_path = tmp_path / "config.json"
    benchmark_path = tmp_path / "benchmark.json"
    runs_dir = tmp_path / "experiments" / "runs"
    registry_path = tmp_path / "experiments" / "registry.jsonl"
    _write_datetime_netcdf(netcdf_path)

    config = TrainingConfig(
        data=DataConfig(
            paths=[str(netcdf_path)],
            zarr_path=str(zarr_path),
            variables=("ssh", "u", "v"),
            input_variables=("ssh", "u", "v"),
            target_variables=("ssh",),
            time_dim="time",
            spatial_dims=("lat", "lon"),
            input_steps=3,
            output_steps=2,
            batch_size=2,
            train_fraction=0.75,
            chunks={"time": 2},
        ),
        model=ModelConfig(pca_components=4, lstm_hidden_size=8, lstm_layers=1, lstm_dropout=0.0),
        optimization=OptimizationConfig(epochs=1, learning_rate=1e-3, device="cpu"),
    )
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "paths": [str(netcdf_path)],
                    "zarr_path": str(zarr_path),
                    "variables": ["ssh", "u", "v"],
                    "input_variables": ["ssh", "u", "v"],
                    "target_variables": ["ssh"],
                    "time_dim": "time",
                    "spatial_dims": ["lat", "lon"],
                    "input_steps": 3,
                    "output_steps": 2,
                    "batch_size": 2,
                    "train_fraction": 0.75,
                    "chunks": {"time": 2},
                },
                "model": {
                    "pca_components": 4,
                    "lstm_hidden_size": 8,
                    "lstm_layers": 1,
                    "lstm_dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.00001,
                    "device": "cpu",
                },
            }
        )
    )
    benchmark_path.write_text(
        json.dumps(
            {
                "name": "synthetic_ssh_benchmark",
                "zarr_path": str(zarr_path),
                "lead_steps": 2,
                "windows": [{"input_start": "2019-01-05"}, {"input_start": "2019-01-06"}],
            }
        )
    )

    result = fit(config)
    save_checkpoint(result, config, checkpoint_path)

    manifest = record_existing_checkpoint(
        checkpoint_path,
        config_path=config_path,
        benchmark_path=benchmark_path,
        runs_dir=runs_dir,
        registry_path=registry_path,
        device="cpu",
        run_id="existing-run",
    )

    run_dir = runs_dir / "existing-run"
    assert manifest["run_id"] == "existing-run"
    assert manifest["original_checkpoint_path"] == str(checkpoint_path)
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "benchmark.json").exists()
    assert (run_dir / "training_summary.json").exists()
    assert (run_dir / "benchmark_metrics.json").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert registry_path.exists()
    assert "existing-run" in registry_path.read_text()