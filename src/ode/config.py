from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    paths: list[str] = field(default_factory=list)
    zarr_path: str | None = None
    variables: tuple[str, ...] = ("ssh", "u", "v")
    input_variables: tuple[str, ...] | None = None
    target_variables: tuple[str, ...] | None = None
    residual_targets: tuple[str, ...] = ()
    time_dim: str = "time"
    spatial_dims: tuple[str, str] | None = None
    input_steps: int = 6
    output_steps: int = 1
    batch_size: int = 4
    num_workers: int = 0
    train_fraction: float = 0.8
    train_end_date: str | None = None
    val_start_date: str | None = None
    val_end_date: str | None = None
    chunks: dict[str, int] = field(default_factory=dict)
    engine: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataConfig":
        return cls(
            paths=list(payload.get("paths", [])),
            zarr_path=payload.get("zarr_path"),
            variables=tuple(payload.get("variables", ("ssh", "u", "v"))),
            input_variables=tuple(payload["input_variables"]) if payload.get("input_variables") else None,
            target_variables=tuple(payload["target_variables"]) if payload.get("target_variables") else None,
            residual_targets=tuple(payload.get("residual_targets", ())),
            time_dim=payload.get("time_dim", "time"),
            spatial_dims=tuple(payload["spatial_dims"]) if payload.get("spatial_dims") else None,
            input_steps=int(payload.get("input_steps", 6)),
            output_steps=int(payload.get("output_steps", 1)),
            batch_size=int(payload.get("batch_size", 4)),
            num_workers=int(payload.get("num_workers", 0)),
            train_fraction=float(payload.get("train_fraction", 0.8)),
            train_end_date=payload.get("train_end_date"),
            val_start_date=payload.get("val_start_date"),
            val_end_date=payload.get("val_end_date"),
            chunks={key: int(value) for key, value in payload.get("chunks", {}).items()},
            engine=payload.get("engine"),
        )

    def resolved_input_variables(self) -> tuple[str, ...]:
        return self.input_variables or self.variables

    def resolved_target_variables(self) -> tuple[str, ...]:
        return self.target_variables or self.resolved_input_variables()

    def resolved_residual_targets(self) -> tuple[str, ...]:
        return self.residual_targets


@dataclass(slots=True)
class ModelConfig:
    pca_components: int = 32
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.0
    autoregressive_decoder: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(
            pca_components=int(payload.get("pca_components", 32)),
            lstm_hidden_size=int(payload.get("lstm_hidden_size", 128)),
            lstm_layers=int(payload.get("lstm_layers", 2)),
            lstm_dropout=float(payload.get("lstm_dropout", 0.0)),
            autoregressive_decoder=bool(payload.get("autoregressive_decoder", False)),
        )


@dataclass(slots=True)
class OptimizationConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OptimizationConfig":
        return cls(
            epochs=int(payload.get("epochs", 10)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-5)),
            device=payload.get("device", "auto"),
        )


@dataclass(slots=True)
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        return cls(
            data=DataConfig.from_dict(payload.get("data", {})),
            model=ModelConfig.from_dict(payload.get("model", {})),
            optimization=OptimizationConfig.from_dict(payload.get("optimization", {})),
        )


def load_training_config(path: str | Path) -> TrainingConfig:
    payload = json.loads(Path(path).read_text())
    return TrainingConfig.from_dict(payload)
