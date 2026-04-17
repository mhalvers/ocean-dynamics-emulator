from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from ode.config import TrainingConfig
from ode.data.dataset import ForecastWindowDataset
from ode.data.netcdf import open_training_dataset
from ode.models.pca_lstm import PCALSTMForecaster, compute_channel_statistics, compute_principal_components
from ode.models.conv_lstm import ConvLSTMForecaster
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
        input_variables=config.data.resolved_input_variables(),
        target_variables=config.data.resolved_target_variables(),
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


def _resolve_time_values(dataset: ForecastWindowDataset) -> np.ndarray:
    time_values = np.asarray(dataset.input_array[dataset.time_dim].values)
    if not np.issubdtype(time_values.dtype, np.datetime64):
        raise ValueError("Date-based splits require a datetime64 time coordinate.")
    return time_values


def _resolve_date_split_bounds(config: TrainingConfig) -> tuple[np.datetime64 | None, np.datetime64 | None, np.datetime64 | None]:
    train_end = np.datetime64(config.data.train_end_date) if config.data.train_end_date else None
    val_start = np.datetime64(config.data.val_start_date) if config.data.val_start_date else None
    val_end = np.datetime64(config.data.val_end_date) if config.data.val_end_date else None

    if val_end is not None and val_start is None:
        raise ValueError("val_end_date requires val_start_date.")
    if train_end is not None and val_start is not None and val_start <= train_end:
        raise ValueError("val_start_date must be later than train_end_date to avoid overlapping date-based splits.")
    if val_start is not None and val_end is not None and val_end < val_start:
        raise ValueError("val_end_date must be later than or equal to val_start_date.")
    return train_end, val_start, val_end


def resolve_split_sample_indices(
    dataset: ForecastWindowDataset,
    config: TrainingConfig,
    *,
    output_steps: int | None = None,
) -> tuple[list[int], list[int]]:
    resolved_output_steps = config.data.output_steps if output_steps is None else int(output_steps)
    sample_count = int(dataset.input_array.sizes[dataset.time_dim]) - config.data.input_steps - resolved_output_steps + 1
    if sample_count <= 0:
        raise ValueError(
            "Not enough timesteps to create forecast windows with the configured "
            f"input_steps={config.data.input_steps} and output_steps={resolved_output_steps}."
        )

    if not any((config.data.train_end_date, config.data.val_start_date, config.data.val_end_date)):
        return split_sample_indices(sample_count, config.data.train_fraction)

    time_values = _resolve_time_values(dataset)
    train_end, val_start, val_end = _resolve_date_split_bounds(config)
    sample_indices = np.arange(sample_count)
    target_start_indices = sample_indices + config.data.input_steps
    target_end_indices = target_start_indices + resolved_output_steps - 1
    target_start_times = time_values[target_start_indices]
    target_end_times = time_values[target_end_indices]

    if train_end is not None:
        train_mask = target_end_times <= train_end
    elif val_start is not None:
        train_mask = target_end_times < val_start
    else:
        train_mask = np.ones(sample_count, dtype=bool)

    if val_start is not None:
        val_mask = target_start_times >= val_start
    elif train_end is not None:
        val_mask = target_start_times > train_end
    else:
        val_mask = np.zeros(sample_count, dtype=bool)

    if val_end is not None:
        val_mask &= target_end_times <= val_end

    train_indices = sample_indices[train_mask].tolist()
    val_indices = sample_indices[val_mask].tolist()
    return train_indices, val_indices


def resolve_train_timestep_count(dataset: ForecastWindowDataset, config: TrainingConfig) -> int:
    total_timesteps = int(dataset.input_array.sizes[dataset.time_dim])
    minimum_train_timesteps = config.data.input_steps + config.data.output_steps
    if not any((config.data.train_end_date, config.data.val_start_date, config.data.val_end_date)):
        train_timesteps = max(int(total_timesteps * config.data.train_fraction), minimum_train_timesteps)
        return min(train_timesteps, total_timesteps)

    time_values = _resolve_time_values(dataset)
    train_end, val_start, _ = _resolve_date_split_bounds(config)
    if train_end is not None:
        train_timesteps = int(np.searchsorted(time_values, train_end, side="right"))
    elif val_start is not None:
        train_timesteps = int(np.searchsorted(time_values, val_start, side="left"))
    else:
        train_timesteps = total_timesteps

    if train_timesteps < minimum_train_timesteps:
        raise ValueError(
            "Date-based split leaves too few timesteps for PCA fitting with the configured "
            f"input_steps={config.data.input_steps} and output_steps={config.data.output_steps}."
        )
    return min(train_timesteps, total_timesteps)


def resolve_residual_target_input_indices(dataset: ForecastWindowDataset, config: TrainingConfig) -> torch.Tensor:
    residual_targets = config.data.resolved_residual_targets()
    if not residual_targets:
        return torch.full((len(dataset.target_variables),), -1, dtype=torch.int64)

    target_index_by_name = {name: index for index, name in enumerate(dataset.target_variables)}
    input_index_by_name = {name: index for index, name in enumerate(dataset.input_variables)}
    residual_input_indices = torch.full((len(dataset.target_variables),), -1, dtype=torch.int64)
    for name in residual_targets:
        if name not in target_index_by_name:
            raise ValueError(f"Residual target '{name}' is not present in target_variables.")
        if name not in input_index_by_name:
            raise ValueError(f"Residual target '{name}' is not present in input_variables, so no persistence baseline is available.")
        residual_input_indices[target_index_by_name[name]] = input_index_by_name[name]
    return residual_input_indices


def build_residual_target_training_fields(dataset: ForecastWindowDataset, config: TrainingConfig) -> torch.Tensor:
    residual_input_indices = resolve_residual_target_input_indices(dataset, config)
    if not torch.any(residual_input_indices >= 0):
        return torch.as_tensor(dataset.target_array.values, dtype=torch.float32)

    train_indices, _ = resolve_split_sample_indices(dataset, config)
    if not train_indices:
        raise ValueError("The training split is empty for the configured dataset.")

    input_fields = torch.as_tensor(dataset.input_array.values, dtype=torch.float32)
    target_fields = torch.as_tensor(dataset.target_array.values, dtype=torch.float32)
    train_index_tensor = torch.as_tensor(train_indices, dtype=torch.int64)
    target_start_indices = train_index_tensor + config.data.input_steps
    output_offsets = torch.arange(config.data.output_steps, dtype=torch.int64)
    target_index_matrix = target_start_indices.unsqueeze(1) + output_offsets.unsqueeze(0)
    residual_fields = target_fields[target_index_matrix].clone()
    baseline_input_indices = train_index_tensor + config.data.input_steps - 1
    baseline_inputs = input_fields[baseline_input_indices]

    for target_index, input_index in enumerate(residual_input_indices.tolist()):
        if input_index < 0:
            continue
        residual_fields[:, :, target_index] = residual_fields[:, :, target_index] - baseline_inputs[:, input_index].unsqueeze(1)
    return residual_fields.reshape(-1, residual_fields.shape[2], residual_fields.shape[3], residual_fields.shape[4])


def _build_dataloaders(dataset: ForecastWindowDataset, config: TrainingConfig) -> tuple[DataLoader, DataLoader | None]:
    train_indices, val_indices = resolve_split_sample_indices(dataset, config)

    if not train_indices:
        raise ValueError("The training split is empty for the configured dataset.")

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


def _fit_projection_statistics(
    fields: torch.Tensor,
    *,
    train_timesteps: int,
    pca_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source = fields[:train_timesteps]
    channel_mean, channel_std = compute_channel_statistics(source)
    normalized_source = (source - channel_mean.view(1, -1, 1, 1)) / channel_std.view(1, -1, 1, 1)
    pca_mean, pca_components_tensor, explained_variance = compute_principal_components(normalized_source, pca_components)
    return pca_mean, pca_components_tensor, explained_variance, channel_mean, channel_std


def _fit_pca_statistics(
    dataset: ForecastWindowDataset, config: TrainingConfig
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    input_fields = torch.as_tensor(dataset.input_array.values, dtype=torch.float32)
    target_fields = build_residual_target_training_fields(dataset, config)
    train_timesteps = resolve_train_timestep_count(dataset, config)
    input_stats = _fit_projection_statistics(
        input_fields,
        train_timesteps=train_timesteps,
        pca_components=config.model.pca_components,
    )
    target_stats = _fit_projection_statistics(
        target_fields,
        train_timesteps=train_timesteps,
        pca_components=config.model.pca_components,
    )
    return input_stats, target_stats


def _teacher_forcing_ratio_for_epoch(config: TrainingConfig, epoch_index: int) -> float:
    start_ratio = float(config.model.teacher_forcing_start_ratio)
    end_ratio = float(config.model.teacher_forcing_end_ratio)
    if not (0.0 <= start_ratio <= 1.0 and 0.0 <= end_ratio <= 1.0):
        raise ValueError("teacher_forcing_start_ratio and teacher_forcing_end_ratio must be between 0 and 1.")
    if config.optimization.epochs <= 1:
        return end_ratio

    progress = epoch_index / float(config.optimization.epochs - 1)
    return start_ratio + (end_ratio - start_ratio) * progress


def fit(config: TrainingConfig) -> TrainingResult:
    dataset, zarr_path = _build_window_dataset(config)
    train_loader, val_loader = _build_dataloaders(dataset, config)

    if len(dataset.spatial_dims) != 2:
        raise ValueError("The forecaster currently supports exactly two spatial dimensions.")

    if config.model.use_conv_lstm:
        input_fields = torch.as_tensor(dataset.input_array.values, dtype=torch.float32)
        target_fields = build_residual_target_training_fields(dataset, config)
        train_timesteps = resolve_train_timestep_count(dataset, config)
        
        input_channel_mean, input_channel_std = compute_channel_statistics(input_fields[:train_timesteps])
        target_channel_mean, target_channel_std = compute_channel_statistics(target_fields[:train_timesteps])

        model = ConvLSTMForecaster(
            input_steps=config.data.input_steps,
            in_channels=len(dataset.input_variables),
            out_channels=len(dataset.target_variables),
            spatial_shape=tuple(int(dataset.input_array.sizes[dim]) for dim in dataset.spatial_dims),
            output_steps=config.data.output_steps,
            input_channel_mean=input_channel_mean,
            input_channel_std=input_channel_std,
            target_channel_mean=target_channel_mean,
            target_channel_std=target_channel_std,
            target_residual_input_indices=resolve_residual_target_input_indices(dataset, config),
            lstm_hidden_size=config.model.lstm_hidden_size,
            lstm_layers=config.model.lstm_layers,
            lstm_dropout=config.model.lstm_dropout,
            autoregressive_decoder=config.model.autoregressive_decoder,
        )
    else:
        input_stats, target_stats = _fit_pca_statistics(dataset, config)
        input_pca_mean, input_pca_components, _, input_channel_mean, input_channel_std = input_stats
        target_pca_mean, target_pca_components, _, target_channel_mean, target_channel_std = target_stats

        model = PCALSTMForecaster(
            input_steps=config.data.input_steps,
            in_channels=len(dataset.input_variables),
            out_channels=len(dataset.target_variables),
            spatial_shape=tuple(int(dataset.input_array.sizes[dim]) for dim in dataset.spatial_dims),
            output_steps=config.data.output_steps,
            input_pca_mean=input_pca_mean,
            input_pca_components=input_pca_components,
            input_channel_mean=input_channel_mean,
            input_channel_std=input_channel_std,
            target_pca_mean=target_pca_mean,
            target_pca_components=target_pca_components,
            target_channel_mean=target_channel_mean,
            target_channel_std=target_channel_std,
            target_residual_input_indices=resolve_residual_target_input_indices(dataset, config),
            lstm_hidden_size=config.model.lstm_hidden_size,
            lstm_layers=config.model.lstm_layers,
            lstm_dropout=config.model.lstm_dropout,
            autoregressive_decoder=config.model.autoregressive_decoder,
            residual_encoder=config.model.residual_encoder,
        )
    
    device = _resolve_device(config.optimization.device)
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch_index in range(config.optimization.epochs):
        teacher_forcing_ratio = _teacher_forcing_ratio_for_epoch(config, epoch_index)
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            if config.model.autoregressive_decoder:
                predictions = model(
                    inputs,
                    teacher_forcing_targets=targets,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )
            else:
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
