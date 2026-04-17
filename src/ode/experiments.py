from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ode.config import TrainingConfig, load_training_config
from ode.inference import _build_model_from_checkpoint, _roll_forward
from ode.training.engine import _build_window_dataset, _resolve_device, fit, save_checkpoint


@dataclass(slots=True)
class BenchmarkWindow:
    input_start: str
    label: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkWindow":
        return cls(input_start=payload["input_start"], label=payload.get("label"))


@dataclass(slots=True)
class BenchmarkSpec:
    name: str
    zarr_path: str
    lead_steps: int
    windows: tuple[BenchmarkWindow, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkSpec":
        windows = tuple(BenchmarkWindow.from_dict(window) for window in payload.get("windows", []))
        if not windows:
            raise ValueError("Benchmark specs must define at least one evaluation window.")
        return cls(
            name=payload["name"],
            zarr_path=payload["zarr_path"],
            lead_steps=int(payload["lead_steps"]),
            windows=windows,
        )


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    payload = json.loads(Path(path).read_text())
    return BenchmarkSpec.from_dict(payload)


def save_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def _safe_skill_score(model_mse: float, baseline_mse: float) -> float | None:
    if baseline_mse <= 0.0:
        return None
    return 1.0 - (model_mse / baseline_mse)


def _resolve_absolute_index(time_values: np.ndarray, input_start: str) -> int:
    target_time = np.datetime64(input_start)
    matches = np.where(time_values == target_time)[0]
    if matches.size == 0:
        raise ValueError(f"Benchmark input_start={input_start} was not found in the dataset time coordinate.")
    return int(matches[0])


def _predict_tensor(model, inputs: torch.Tensor, *, lead_steps: int, device: str, can_roll_forward: bool) -> torch.Tensor:
    if lead_steps <= model.output_steps:
        with torch.no_grad():
            return model(inputs.unsqueeze(0).to(device)).cpu().squeeze(0)[:lead_steps]
    if not can_roll_forward:
        raise ValueError(
            "Benchmark lead_steps exceed the checkpoint horizon, but autoregressive rollout is unavailable "
            "because target variables do not match input variables."
        )
    return _roll_forward(model, inputs, lead_steps, device)


def _build_persistence_baseline(
    inputs: torch.Tensor,
    *,
    input_variables: tuple[str, ...],
    target_variables: tuple[str, ...],
    lead_steps: int,
) -> torch.Tensor:
    input_index_by_name = {name: index for index, name in enumerate(input_variables)}
    repeated_channels: list[torch.Tensor] = []
    for target_name in target_variables:
        if target_name not in input_index_by_name:
            raise ValueError(
                f"Cannot build a persistence baseline for target variable '{target_name}' because it is not present in inputs."
            )
        repeated_channels.append(inputs[-1, input_index_by_name[target_name]].unsqueeze(0))
    baseline = torch.cat(repeated_channels, dim=0)
    return baseline.unsqueeze(0).repeat(lead_steps, 1, 1, 1)


def _compute_metric_block(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    persistence: torch.Tensor,
    *,
    target_variables: tuple[str, ...],
) -> dict[str, Any]:
    prediction_errors = predictions - targets
    persistence_errors = persistence - targets

    overall_mse = float(torch.mean(prediction_errors**2).item())
    persistence_mse = float(torch.mean(persistence_errors**2).item())
    per_variable: dict[str, Any] = {}
    for variable_index, variable_name in enumerate(target_variables):
        variable_errors = prediction_errors[:, variable_index]
        variable_persistence_errors = persistence_errors[:, variable_index]
        variable_mse = float(torch.mean(variable_errors**2).item())
        variable_persistence_mse = float(torch.mean(variable_persistence_errors**2).item())

        per_lead: list[dict[str, Any]] = []
        for lead_index in range(predictions.shape[0]):
            lead_mse = float(torch.mean(variable_errors[lead_index] ** 2).item())
            lead_persistence_mse = float(torch.mean(variable_persistence_errors[lead_index] ** 2).item())
            per_lead.append(
                {
                    "lead_step": lead_index + 1,
                    "mse": lead_mse,
                    "rmse": lead_mse**0.5,
                    "persistence_mse": lead_persistence_mse,
                    "persistence_rmse": lead_persistence_mse**0.5,
                    "skill_vs_persistence": _safe_skill_score(lead_mse, lead_persistence_mse),
                }
            )

        per_variable[variable_name] = {
            "overall_mse": variable_mse,
            "overall_rmse": variable_mse**0.5,
            "persistence_mse": variable_persistence_mse,
            "persistence_rmse": variable_persistence_mse**0.5,
            "skill_vs_persistence": _safe_skill_score(variable_mse, variable_persistence_mse),
            "per_lead": per_lead,
        }

    return {
        "overall_mse": overall_mse,
        "overall_rmse": overall_mse**0.5,
        "persistence_mse": persistence_mse,
        "persistence_rmse": persistence_mse**0.5,
        "skill_vs_persistence": _safe_skill_score(overall_mse, persistence_mse),
        "variables": per_variable,
    }


def _aggregate_window_metrics(windows: list[dict[str, Any]], *, target_variables: tuple[str, ...], lead_steps: int) -> dict[str, Any]:
    window_count = len(windows)
    overall_mse = float(sum(window["overall_mse"] for window in windows) / window_count)
    persistence_mse = float(sum(window["persistence_mse"] for window in windows) / window_count)

    variable_summary: dict[str, Any] = {}
    for variable_name in target_variables:
        variable_mse = float(sum(window["variables"][variable_name]["overall_mse"] for window in windows) / window_count)
        variable_persistence_mse = float(
            sum(window["variables"][variable_name]["persistence_mse"] for window in windows) / window_count
        )
        per_lead: list[dict[str, Any]] = []
        for lead_index in range(lead_steps):
            lead_mse = float(
                sum(window["variables"][variable_name]["per_lead"][lead_index]["mse"] for window in windows) / window_count
            )
            lead_persistence_mse = float(
                sum(
                    window["variables"][variable_name]["per_lead"][lead_index]["persistence_mse"]
                    for window in windows
                )
                / window_count
            )
            per_lead.append(
                {
                    "lead_step": lead_index + 1,
                    "mse": lead_mse,
                    "rmse": lead_mse**0.5,
                    "persistence_mse": lead_persistence_mse,
                    "persistence_rmse": lead_persistence_mse**0.5,
                    "skill_vs_persistence": _safe_skill_score(lead_mse, lead_persistence_mse),
                }
            )

        variable_summary[variable_name] = {
            "overall_mse": variable_mse,
            "overall_rmse": variable_mse**0.5,
            "persistence_mse": variable_persistence_mse,
            "persistence_rmse": variable_persistence_mse**0.5,
            "skill_vs_persistence": _safe_skill_score(variable_mse, variable_persistence_mse),
            "per_lead": per_lead,
        }

    return {
        "window_count": window_count,
        "overall_mse": overall_mse,
        "overall_rmse": overall_mse**0.5,
        "persistence_mse": persistence_mse,
        "persistence_rmse": persistence_mse**0.5,
        "skill_vs_persistence": _safe_skill_score(overall_mse, persistence_mse),
        "variables": variable_summary,
    }


def evaluate_checkpoint_against_benchmark(
    checkpoint_path: str | Path,
    benchmark_path: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = TrainingConfig.from_dict(checkpoint["config"])
    benchmark = load_benchmark_spec(benchmark_path)
    config.data.zarr_path = benchmark.zarr_path

    dataset, data_store = _build_window_dataset(config)
    resolved_device = _resolve_device(device)
    model = _build_model_from_checkpoint(config, dataset, checkpoint, resolved_device)
    time_values = np.asarray(dataset.input_array[dataset.time_dim].values)
    can_roll_forward = dataset.input_variables == dataset.target_variables

    window_metrics: list[dict[str, Any]] = []
    for window in benchmark.windows:
        absolute_index = _resolve_absolute_index(time_values, window.input_start)
        max_index = absolute_index + config.data.input_steps + benchmark.lead_steps
        if max_index > int(dataset.input_array.sizes[dataset.time_dim]):
            raise ValueError(
                f"Benchmark window starting at {window.input_start} exceeds the available dataset horizon for lead_steps={benchmark.lead_steps}."
            )

        input_slice = dataset.input_array.isel({dataset.time_dim: slice(absolute_index, absolute_index + config.data.input_steps)})
        target_slice = dataset.target_array.isel(
            {dataset.time_dim: slice(absolute_index + config.data.input_steps, absolute_index + config.data.input_steps + benchmark.lead_steps)}
        )
        inputs = torch.as_tensor(np.asarray(input_slice.values, dtype=np.float32))
        targets = torch.as_tensor(np.asarray(target_slice.values, dtype=np.float32))
        predictions = _predict_tensor(
            model,
            inputs,
            lead_steps=benchmark.lead_steps,
            device=resolved_device,
            can_roll_forward=can_roll_forward,
        )
        persistence = _build_persistence_baseline(
            inputs,
            input_variables=tuple(dataset.input_variables),
            target_variables=tuple(dataset.target_variables),
            lead_steps=benchmark.lead_steps,
        )
        metric_block = _compute_metric_block(
            predictions,
            targets,
            persistence,
            target_variables=tuple(dataset.target_variables),
        )
        window_metrics.append(
            {
                "label": window.label or window.input_start,
                "input_start": window.input_start,
                "absolute_index": absolute_index,
                "target_start": str(time_values[absolute_index + config.data.input_steps]),
                "target_end": str(time_values[absolute_index + config.data.input_steps + benchmark.lead_steps - 1]),
                **metric_block,
            }
        )

    return {
        "benchmark_name": benchmark.name,
        "benchmark_path": str(Path(benchmark_path)),
        "checkpoint_path": str(Path(checkpoint_path)),
        "evaluated_data_store": data_store,
        "lead_steps": benchmark.lead_steps,
        "input_variables": list(dataset.input_variables),
        "target_variables": list(dataset.target_variables),
        "summary": _aggregate_window_metrics(
            window_metrics,
            target_variables=tuple(dataset.target_variables),
            lead_steps=benchmark.lead_steps,
        ),
        "windows": window_metrics,
    }


def append_registry_entry(path: str | Path, entry: dict[str, Any]) -> Path:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return registry_path


def _build_default_run_id(config: TrainingConfig) -> str:
    target_slug = "-".join(config.data.resolved_target_variables())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{timestamp}_{target_slug}_pca{config.model.pca_components}_"
        f"h{config.model.lstm_hidden_size}_e{config.optimization.epochs}"
    )


def _resolve_git_metadata(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _build_registry_entry(
    *,
    run_id: str,
    run_dir: Path,
    config_path: str | Path | None,
    benchmark_path: str | Path | None,
    checkpoint_path: str | Path,
    training_data_store: str | None,
    config: TrainingConfig,
    final_train_loss: float | None,
    final_val_loss: float | None,
    benchmark_metrics: dict[str, Any] | None,
    original_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path.cwd()
    git_commit, git_dirty = _resolve_git_metadata(repo_root)
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "run_dir": str(run_dir),
        "config_path": str(Path(config_path)) if config_path is not None else None,
        "benchmark_path": str(Path(benchmark_path)) if benchmark_path is not None else None,
        "checkpoint_path": str(Path(checkpoint_path)),
        "original_checkpoint_path": str(Path(original_checkpoint_path)) if original_checkpoint_path is not None else None,
        "training_data_store": training_data_store,
        "input_variables": list(config.data.resolved_input_variables()),
        "target_variables": list(config.data.resolved_target_variables()),
        "input_steps": config.data.input_steps,
        "output_steps": config.data.output_steps,
        "pca_components": config.model.pca_components,
        "lstm_hidden_size": config.model.lstm_hidden_size,
        "lstm_layers": config.model.lstm_layers,
        "lstm_dropout": config.model.lstm_dropout,
        "autoregressive_decoder": config.model.autoregressive_decoder,
        "epochs": config.optimization.epochs,
        "learning_rate": config.optimization.learning_rate,
        "weight_decay": config.optimization.weight_decay,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "benchmark_overall_mse": benchmark_metrics["summary"]["overall_mse"] if benchmark_metrics is not None else None,
        "benchmark_overall_rmse": benchmark_metrics["summary"]["overall_rmse"] if benchmark_metrics is not None else None,
        "benchmark_ssh_mse": (
            benchmark_metrics["summary"]["variables"]["ssh"]["overall_mse"]
            if benchmark_metrics is not None and "ssh" in benchmark_metrics["summary"]["variables"]
            else None
        ),
    }


def run_tracked_experiment(
    config_path: str | Path,
    *,
    benchmark_path: str | Path | None = None,
    runs_dir: str | Path = "experiments/runs",
    registry_path: str | Path = "experiments/registry.jsonl",
    device: str = "auto",
    run_id: str | None = None,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    resolved_run_id = run_id or _build_default_run_id(config)
    run_dir = Path(runs_dir) / resolved_run_id
    if run_dir.exists():
        raise FileExistsError(f"Experiment run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(run_dir / "config.json", asdict(config))
    if benchmark_path is not None:
        benchmark = load_benchmark_spec(benchmark_path)
        save_json(run_dir / "benchmark.json", asdict(benchmark))

    result = fit(config)
    checkpoint_path = save_checkpoint(result, config, run_dir / "checkpoint.pt")
    final_train_loss = float(result.history["train_loss"][-1]) if result.history["train_loss"] else None
    final_val_loss = float(result.history["val_loss"][-1]) if result.history["val_loss"] else None
    training_summary = {
        "device": result.device,
        "training_data_store": result.zarr_path,
        "checkpoint_path": str(checkpoint_path),
        "train_loss_history": result.history["train_loss"],
        "val_loss_history": result.history["val_loss"],
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
    save_json(run_dir / "training_summary.json", training_summary)

    benchmark_metrics = None
    if benchmark_path is not None:
        benchmark_metrics = evaluate_checkpoint_against_benchmark(checkpoint_path, benchmark_path, device=device)
        save_json(run_dir / "benchmark_metrics.json", benchmark_metrics)

    registry_entry = _build_registry_entry(
        run_id=resolved_run_id,
        run_dir=run_dir,
        config_path=config_path,
        benchmark_path=benchmark_path,
        checkpoint_path=checkpoint_path,
        training_data_store=result.zarr_path,
        config=config,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        benchmark_metrics=benchmark_metrics,
    )
    append_registry_entry(registry_path, registry_entry)
    save_json(run_dir / "run_manifest.json", registry_entry)
    return registry_entry


def record_existing_checkpoint(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
    runs_dir: str | Path = "experiments/runs",
    registry_path: str | Path = "experiments/registry.jsonl",
    device: str = "auto",
    run_id: str | None = None,
) -> dict[str, Any]:
    source_checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainingConfig.from_dict(checkpoint["config"])

    resolved_run_id = run_id or source_checkpoint_path.stem
    run_dir = Path(runs_dir) / resolved_run_id
    if run_dir.exists():
        raise FileExistsError(f"Experiment run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(run_dir / "config.json", asdict(config))
    copied_checkpoint_path = run_dir / "checkpoint.pt"
    shutil.copy2(source_checkpoint_path, copied_checkpoint_path)

    benchmark_metrics = None
    if benchmark_path is not None:
        benchmark = load_benchmark_spec(benchmark_path)
        save_json(run_dir / "benchmark.json", asdict(benchmark))
        benchmark_metrics = evaluate_checkpoint_against_benchmark(copied_checkpoint_path, benchmark_path, device=device)
        save_json(run_dir / "benchmark_metrics.json", benchmark_metrics)

    train_history = checkpoint.get("history", {}).get("train_loss", [])
    val_history = checkpoint.get("history", {}).get("val_loss", [])
    final_train_loss = float(train_history[-1]) if train_history else None
    final_val_loss = float(val_history[-1]) if val_history else None
    training_summary = {
        "device": checkpoint.get("device"),
        "training_data_store": checkpoint.get("zarr_path"),
        "checkpoint_path": str(copied_checkpoint_path),
        "source_checkpoint_path": str(source_checkpoint_path),
        "train_loss_history": train_history,
        "val_loss_history": val_history,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
    save_json(run_dir / "training_summary.json", training_summary)

    registry_entry = _build_registry_entry(
        run_id=resolved_run_id,
        run_dir=run_dir,
        config_path=config_path,
        benchmark_path=benchmark_path,
        checkpoint_path=copied_checkpoint_path,
        training_data_store=checkpoint.get("zarr_path"),
        config=config,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        benchmark_metrics=benchmark_metrics,
        original_checkpoint_path=source_checkpoint_path,
    )
    append_registry_entry(registry_path, registry_entry)
    save_json(run_dir / "run_manifest.json", registry_entry)
    return registry_entry


__all__ = [
    "BenchmarkSpec",
    "BenchmarkWindow",
    "append_registry_entry",
    "evaluate_checkpoint_against_benchmark",
    "load_benchmark_spec",
    "record_existing_checkpoint",
    "run_tracked_experiment",
    "save_json",
]