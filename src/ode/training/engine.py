from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from ode.config import TrainingConfig
from ode.data.dataset import ForecastWindowDataset
from ode.data.netcdf import open_training_dataset
from ode.models.pca_lstm import PCALSTMForecaster, compute_principal_components
from ode.training.losses import mse_loss


@dataclass(slots=True)
class TrainingResult:
    model: nn.Module
    history: dict[str, list[float]]
    device: str
    zarr_path: str | None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_window_dataset(config: TrainingConfig) -> tuple[ForecastWindowDataset, str | None]:
    dataset, store_path = open_training_dataset(
        paths=config.data.paths,
        zarr_path=config.data.zarr_path,
        chunks=config.data.chunks,
        engine=config.data.engine,
    )
    window_dataset = ForecastWindowDataset(
        dataset,
        variables=config.data.variables,
        input_steps=config.data.input_steps,
        output_steps=config.data.output_steps,
        time_dim=config.data.time_dim,
        spatial_dims=config.data.spatial_dims,
    )
    return window_dataset, str(store_path) if store_path else None


def split_sample_indices(sample_count: int, train_fraction: float) -> tuple[list[int], list[int]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if sample_count == 1:
        return [0], []

    train_count = int(sample_count * train_fraction)
    train_count = min(max(train_count, 1), sample_count - 1)
    train_indices = list(range(train_count))
    val_indices = list(range(train_count, sample_count))
    return train_indices, val_indices


def _build_dataloaders(dataset: ForecastWindowDataset, config: TrainingConfig) -> tuple[DataLoader, DataLoader | None]:
    train_indices, val_indices = split_sample_indices(len(dataset), config.data.train_fraction)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
    return train_loader, val_loader


def _fit_pca_statistics(dataset: ForecastWindowDataset, config: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fields = torch.as_tensor(dataset.array.values, dtype=torch.float32)
    total_timesteps = int(fields.shape[0])
    minimum_train_timesteps = config.data.input_steps + config.data.output_steps
    train_timesteps = max(int(total_timesteps * config.data.train_fraction), minimum_train_timesteps)
    train_timesteps = min(train_timesteps, total_timesteps)
    pca_source = fields[:train_timesteps]
    return compute_principal_components(pca_source, config.model.pca_components)


def fit(config: TrainingConfig) -> TrainingResult:
    dataset, zarr_path = _build_window_dataset(config)
    train_loader, val_loader = _build_dataloaders(dataset, config)

    if len(dataset.spatial_dims) != 2:
        raise ValueError("The PCA-LSTM forecaster currently supports exactly two spatial dimensions.")

    pca_mean, pca_components, _ = _fit_pca_statistics(dataset, config)

    model = PCALSTMForecaster(
        input_steps=config.data.input_steps,
        in_channels=len(config.data.variables),
        spatial_shape=tuple(int(dataset.array.sizes[dim]) for dim in dataset.spatial_dims),
        output_steps=config.data.output_steps,
        pca_mean=pca_mean,
        pca_components=pca_components,
        lstm_hidden_size=config.model.lstm_hidden_size,
        lstm_layers=config.model.lstm_layers,
        lstm_dropout=config.model.lstm_dropout,
    )
    device = _resolve_device(config.optimization.device)
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for _ in range(config.optimization.epochs):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1
        history["train_loss"].append(train_loss_total / max(train_batches, 1))

        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    predictions = model(inputs)
                    loss = mse_loss(predictions, targets)
                    val_loss_total += float(loss.item())
                    val_batches += 1
            history["val_loss"].append(val_loss_total / max(val_batches, 1))

    return TrainingResult(model=model, history=history, device=device, zarr_path=zarr_path)


def save_checkpoint(result: TrainingResult, config: TrainingConfig, checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "history": result.history,
            "device": result.device,
            "zarr_path": result.zarr_path,
            "config": asdict(config),
        },
        checkpoint,
    )
    return checkpoint
