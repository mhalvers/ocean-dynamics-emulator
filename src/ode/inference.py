from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from ode.config import TrainingConfig
from ode.models.pca_lstm import PCALSTMForecaster
from ode.models.conv_lstm import ConvLSTMForecaster
from ode.training.engine import _build_window_dataset, _fit_pca_statistics, _resolve_device, resolve_residual_target_input_indices, resolve_split_sample_indices, build_residual_target_training_fields, resolve_train_timestep_count, compute_channel_statistics


@dataclass(slots=True)
class ForecastPrediction:
    checkpoint_path: Path
    config: TrainingConfig
    split: str
    lead_steps: int
    relative_sample_index: int
    absolute_sample_index: int
    data_store: str | None
    input_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    spatial_dims: tuple[str, ...]
    spatial_coord_values: tuple[np.ndarray, ...]
    input_times: np.ndarray
    target_times: np.ndarray
    inputs: torch.Tensor
    targets: torch.Tensor
    predictions: torch.Tensor
    mse: float


@dataclass(slots=True)
class PointTimeseriesSelection:
    point_index: tuple[int, int]
    point_coordinates: tuple[object, object]


def format_time_value(value: object) -> str:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.datetime64):
        return np.datetime_as_string(array, unit="s")
    if array.ndim == 0:
        return str(array.item())
    return str(array)


def _resolve_lead_steps(config: TrainingConfig, lead_steps: int | None) -> int:
    resolved = config.data.output_steps if lead_steps is None else int(lead_steps)
    if resolved <= 0:
        raise ValueError("lead_steps must be a positive integer.")
    return resolved


def _roll_forward(model: PCALSTMForecaster, inputs: torch.Tensor, lead_steps: int, device: str) -> torch.Tensor:
    current_window = inputs.to(device)
    predicted_chunks: list[torch.Tensor] = []
    predicted_steps = 0

    with torch.no_grad():
        while predicted_steps < lead_steps:
            next_chunk = model(current_window.unsqueeze(0)).cpu().squeeze(0)
            predicted_chunks.append(next_chunk)
            predicted_steps += int(next_chunk.shape[0])
            current_window = torch.cat([current_window, next_chunk.to(device)], dim=0)[-model.input_steps :]

    return torch.cat(predicted_chunks, dim=0)[:lead_steps]


def _resolve_current_quiver_settings(*field_pairs: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    magnitude_limit = max(float(np.hypot(u_field, v_field).max()) for u_field, v_field in field_pairs)
    if not np.isfinite(magnitude_limit) or magnitude_limit <= 0.0:
        magnitude_limit = 1.0
    return 0.0, magnitude_limit, magnitude_limit


def _plot_quiver_panel(
    ax,
    *,
    u_field: np.ndarray,
    v_field: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    cmap: str,
    color_norm: Normalize,
    quiver_scale: float,
):
    height, width = u_field.shape
    stride = max(1, min(height, width) // 20)
    y_coords, x_coords = np.mgrid[0:height:stride, 0:width:stride]
    u_sample = u_field[::stride, ::stride]
    v_sample = v_field[::stride, ::stride]
    magnitude = np.hypot(u_sample, v_sample)

    quiver = ax.quiver(
        x_coords,
        y_coords,
        u_sample,
        v_sample,
        magnitude,
        cmap=cmap,
        norm=color_norm,
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
    )
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("auto")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return quiver


def _resolve_point_selection(
    prediction: ForecastPrediction,
    *,
    point_index: tuple[int, int] | None = None,
    seed: int | None = None,
) -> PointTimeseriesSelection:
    height, width = prediction.inputs.shape[-2:]
    if point_index is None:
        rng = np.random.default_rng(seed)
        resolved_point = (int(rng.integers(height)), int(rng.integers(width)))
    else:
        resolved_point = (int(point_index[0]), int(point_index[1]))

    row_index, column_index = resolved_point
    if row_index < 0 or row_index >= height or column_index < 0 or column_index >= width:
        raise IndexError(f"point_index={resolved_point} is out of range for spatial shape {(height, width)}.")

    coordinates: list[object] = []
    for axis_index, coord_values in enumerate(prediction.spatial_coord_values):
        coord_value = np.asarray(coord_values[resolved_point[axis_index]])
        coordinates.append(coord_value.item() if coord_value.ndim == 0 else coord_value)
    return PointTimeseriesSelection(point_index=resolved_point, point_coordinates=(coordinates[0], coordinates[1]))

def _resolve_prediction_index(split_indices: list[int], split: str, sample_index: int) -> tuple[int, int]:
    if split == "train":
        resolved_split_indices = split_indices
    elif split == "val":
        resolved_split_indices = split_indices
    elif split == "all":
        resolved_split_indices = split_indices
    else:
        raise ValueError("split must be one of: train, val, all.")

    if not resolved_split_indices:
        raise ValueError(f"The {split} split is empty for the configured dataset.")

    resolved_index = sample_index if sample_index >= 0 else len(resolved_split_indices) + sample_index
    if resolved_index < 0 or resolved_index >= len(resolved_split_indices):
        raise IndexError(
            f"sample_index={sample_index} is out of range for the {split} split with {len(resolved_split_indices)} samples."
        )
    return resolved_index, resolved_split_indices[resolved_index]


def _build_model_from_checkpoint(config: TrainingConfig, dataset, checkpoint: dict, device: str) -> PCALSTMForecaster | ConvLSTMForecaster:
    if config.model.use_conv_lstm:
        import torch
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
            residual_encoder=config.model.residual_encoder,
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_window(
    checkpoint_path: str | Path,
    *,
    split: str = "val",
    sample_index: int = 0,
    lead_steps: int | None = None,
    device: str = "auto",
) -> ForecastPrediction:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = TrainingConfig.from_dict(checkpoint["config"])
    dataset, data_store = _build_window_dataset(config)
    resolved_lead_steps = _resolve_lead_steps(config, lead_steps)
    requires_autoregressive_rollout = resolved_lead_steps > config.data.output_steps
    if requires_autoregressive_rollout and dataset.input_variables != dataset.target_variables:
        raise ValueError(
            "Autoregressive rollout beyond the direct forecast horizon requires target_variables to match input_variables."
        )

    required_horizon = max(config.data.output_steps, resolved_lead_steps)
    time_values = np.asarray(dataset.input_array[dataset.time_dim].values)
    if split == "all":
        sample_count = int(dataset.input_array.sizes[dataset.time_dim]) - config.data.input_steps - required_horizon + 1
        if sample_count <= 0:
            raise ValueError(
                "Not enough timesteps to make the requested forecast window with "
                f"input_steps={config.data.input_steps} and lead_steps={required_horizon}."
            )
        split_indices = list(range(sample_count))
    elif split == "train":
        split_indices = resolve_split_sample_indices(dataset, config, output_steps=required_horizon)[0]
    elif split == "val":
        split_indices = resolve_split_sample_indices(dataset, config, output_steps=required_horizon)[1]
    else:
        raise ValueError("split must be one of: train, val, all.")
    relative_sample_index, absolute_sample_index = _resolve_prediction_index(split_indices, split, sample_index)
    resolved_device = _resolve_device(device)
    model = _build_model_from_checkpoint(config, dataset, checkpoint, resolved_device)

    input_slice = dataset.input_array.isel({dataset.time_dim: slice(absolute_sample_index, absolute_sample_index + config.data.input_steps)})
    target_slice = dataset.target_array.isel(
        {
            dataset.time_dim: slice(
                absolute_sample_index + config.data.input_steps,
                absolute_sample_index + config.data.input_steps + resolved_lead_steps,
            )
        }
    )
    inputs = torch.as_tensor(np.asarray(input_slice.values, dtype=np.float32))
    targets = torch.as_tensor(np.asarray(target_slice.values, dtype=np.float32))
    input_times = time_values[absolute_sample_index : absolute_sample_index + config.data.input_steps]
    target_times = time_values[
        absolute_sample_index + config.data.input_steps : absolute_sample_index + config.data.input_steps + resolved_lead_steps
    ]

    if requires_autoregressive_rollout:
        predictions = _roll_forward(model, inputs, resolved_lead_steps, resolved_device)
    else:
        with torch.no_grad():
            predictions = model(inputs.unsqueeze(0).to(resolved_device)).cpu().squeeze(0)[:resolved_lead_steps]

    mse = float(torch.mean((predictions - targets) ** 2).item())
    return ForecastPrediction(
        checkpoint_path=Path(checkpoint_path),
        config=config,
        split=split,
        lead_steps=resolved_lead_steps,
        relative_sample_index=relative_sample_index,
        absolute_sample_index=absolute_sample_index,
        data_store=data_store,
        input_variables=tuple(dataset.input_variables),
        target_variables=tuple(dataset.target_variables),
        spatial_dims=tuple(dataset.spatial_dims),
        spatial_coord_values=tuple(np.asarray(dataset.input_array[dim].values) for dim in dataset.spatial_dims),
        input_times=input_times,
        target_times=target_times,
        inputs=inputs,
        targets=targets,
        predictions=predictions,
        mse=mse,
    )


def plot_prediction(
    prediction: ForecastPrediction,
    *,
    forecast_step: int = 0,
    figure_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    if forecast_step < 0 or forecast_step >= prediction.targets.shape[0]:
        raise IndexError(f"forecast_step={forecast_step} is out of range for {prediction.targets.shape[0]} forecast steps.")

    targets = prediction.targets[forecast_step].numpy()
    outputs = prediction.predictions[forecast_step].numpy()
    errors = outputs - targets
    variable_to_index = {name: index for index, name in enumerate(prediction.target_variables)}
    has_currents = "u" in variable_to_index and "v" in variable_to_index
    scalar_variables = [name for name in prediction.target_variables if name not in {"u", "v"}]
    row_count = len(scalar_variables) + int(has_currents)
    figure, axes = plt.subplots(row_count, 3, figsize=(12, 4 * row_count), constrained_layout=True, squeeze=False)

    row_index = 0
    for variable_name in scalar_variables:
        variable_index = variable_to_index[variable_name]
        target_slice = targets[variable_index]
        output_slice = outputs[variable_index]
        error_slice = errors[variable_index]
        field_min = min(float(target_slice.min()), float(output_slice.min()))
        field_max = max(float(target_slice.max()), float(output_slice.max()))
        error_limit = max(abs(float(error_slice.min())), abs(float(error_slice.max())))
        if error_limit == 0.0:
            error_limit = 1.0

        target_image = axes[row_index][0].imshow(target_slice, origin="lower", aspect="auto", vmin=field_min, vmax=field_max)
        output_image = axes[row_index][1].imshow(output_slice, origin="lower", aspect="auto", vmin=field_min, vmax=field_max)
        error_image = axes[row_index][2].imshow(
            error_slice,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
        )

        axes[row_index][0].set_title(f"{variable_name} target")
        axes[row_index][1].set_title(f"{variable_name} prediction")
        axes[row_index][2].set_title(f"{variable_name} error")
        for column_index, dim_name in enumerate((prediction.spatial_dims[1],) * 3):
            axes[row_index][column_index].set_xlabel(dim_name)
            axes[row_index][column_index].set_ylabel(prediction.spatial_dims[0])
        figure.colorbar(target_image, ax=axes[row_index][0])
        figure.colorbar(output_image, ax=axes[row_index][1])
        figure.colorbar(error_image, ax=axes[row_index][2])
        row_index += 1

    if has_currents:
        u_index = variable_to_index["u"]
        v_index = variable_to_index["v"]
        x_label = prediction.spatial_dims[1]
        y_label = prediction.spatial_dims[0]
        current_color_min, current_color_max, current_quiver_scale = _resolve_current_quiver_settings(
            (targets[u_index], targets[v_index]),
            (outputs[u_index], outputs[v_index]),
            (errors[u_index], errors[v_index]),
        )
        current_color_norm = Normalize(vmin=current_color_min, vmax=current_color_max)

        target_quiver = _plot_quiver_panel(
            axes[row_index][0],
            u_field=targets[u_index],
            v_field=targets[v_index],
            title="surface currents target",
            x_label=x_label,
            y_label=y_label,
            cmap="viridis",
            color_norm=current_color_norm,
            quiver_scale=current_quiver_scale,
        )
        prediction_quiver = _plot_quiver_panel(
            axes[row_index][1],
            u_field=outputs[u_index],
            v_field=outputs[v_index],
            title="surface currents prediction",
            x_label=x_label,
            y_label=y_label,
            cmap="viridis",
            color_norm=current_color_norm,
            quiver_scale=current_quiver_scale,
        )
        error_quiver = _plot_quiver_panel(
            axes[row_index][2],
            u_field=errors[u_index],
            v_field=errors[v_index],
            title="surface currents error",
            x_label=x_label,
            y_label=y_label,
            cmap="magma",
            color_norm=current_color_norm,
            quiver_scale=current_quiver_scale,
        )
        figure.colorbar(target_quiver, ax=axes[row_index][0])
        figure.colorbar(prediction_quiver, ax=axes[row_index][1])
        figure.colorbar(error_quiver, ax=axes[row_index][2])

    target_label = format_time_value(prediction.target_times[forecast_step])
    figure.suptitle(
        f"Lead step {forecast_step + 1}/{prediction.lead_steps} for {target_label} | MSE={prediction.mse:.6f}"
    )

    saved_path = None
    if figure_path is not None:
        saved_path = Path(figure_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
    return saved_path


def plot_point_timeseries(
    prediction: ForecastPrediction,
    *,
    point_index: tuple[int, int] | None = None,
    seed: int | None = None,
    figure_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Path | None, PointTimeseriesSelection]:
    selection = _resolve_point_selection(prediction, point_index=point_index, seed=seed)
    row_index, column_index = selection.point_index
    input_variable_to_index = {name: index for index, name in enumerate(prediction.input_variables)}
    target_variable_to_index = {name: index for index, name in enumerate(prediction.target_variables)}
    lookback_steps = prediction.inputs.shape[0]
    forecast_positions = np.arange(lookback_steps, lookback_steps + prediction.lead_steps)
    history_positions = np.arange(lookback_steps)
    tick_positions = np.concatenate([history_positions, forecast_positions])
    tick_labels = [format_time_value(value) for value in prediction.input_times] + [format_time_value(value) for value in prediction.target_times]

    figure, axes = plt.subplots(
        len(prediction.target_variables),
        1,
        figsize=(12, 3.5 * len(prediction.target_variables)),
        sharex=True,
        constrained_layout=True,
    )
    if len(prediction.target_variables) == 1:
        axes = [axes]

    for axis, variable_name in zip(axes, prediction.target_variables):
        target_index = target_variable_to_index[variable_name]
        actual_series = prediction.targets[:, target_index, row_index, column_index].numpy()
        forecast_series = prediction.predictions[:, target_index, row_index, column_index].numpy()

        if variable_name in input_variable_to_index:
            input_index = input_variable_to_index[variable_name]
            lookback_series = prediction.inputs[:, input_index, row_index, column_index].numpy()
            axis.plot(history_positions, lookback_series, color="0.2", marker="o", label="lookback")
            axis.plot(
                np.concatenate([[history_positions[-1]], forecast_positions]),
                np.concatenate([[lookback_series[-1]], actual_series]),
                color="tab:blue",
                marker="o",
                label="actual",
            )
            axis.plot(
                np.concatenate([[history_positions[-1]], forecast_positions]),
                np.concatenate([[lookback_series[-1]], forecast_series]),
                color="tab:orange",
                marker="o",
                linestyle="--",
                label="forecast",
            )
            axis.axvline(history_positions[-1], color="0.5", linestyle=":")
        else:
            axis.plot(forecast_positions, actual_series, color="tab:blue", marker="o", label="actual")
            axis.plot(
                forecast_positions,
                forecast_series,
                color="tab:orange",
                marker="o",
                linestyle="--",
                label="forecast",
            )
        axis.set_ylabel(variable_name)
        axis.legend(loc="best")

    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("time")
    coordinate_text = ", ".join(
        f"{dim}={format_time_value(value)}" for dim, value in zip(prediction.spatial_dims, selection.point_coordinates)
    )
    figure.suptitle(
        f"Point time series at {coordinate_text} (indices {selection.point_index[0]}, {selection.point_index[1]})"
    )

    saved_path = None
    if figure_path is not None:
        saved_path = Path(figure_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
    return saved_path, selection


__all__ = [
    "ForecastPrediction",
    "PointTimeseriesSelection",
    "_resolve_current_quiver_settings",
    "format_time_value",
    "plot_point_timeseries",
    "plot_prediction",
    "predict_window",
]