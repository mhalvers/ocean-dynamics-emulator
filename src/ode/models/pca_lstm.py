from __future__ import annotations

import torch
from torch import nn


def compute_principal_components(fields: torch.Tensor, num_components: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fields.ndim != 4:
        raise ValueError("PCA fitting expects fields shaped as [time, channels, height, width].")

    samples = fields.reshape(fields.shape[0], -1)
    max_rank = min(samples.shape[0], samples.shape[1])
    if num_components <= 0:
        raise ValueError("num_components must be a positive integer.")
    if num_components > max_rank:
        raise ValueError(f"num_components={num_components} exceeds the maximum PCA rank {max_rank}.")

    mean_vector = samples.mean(dim=0)
    centered = samples - mean_vector
    _, singular_values, right_vectors = torch.pca_lowrank(centered, q=num_components, center=False)
    components = right_vectors[:, :num_components].transpose(0, 1).contiguous()
    explained_variance = singular_values[:num_components] ** 2 / max(samples.shape[0] - 1, 1)
    return mean_vector, components, explained_variance


def compute_channel_statistics(fields: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    if fields.ndim != 4:
        raise ValueError("Channel statistics expect fields shaped as [time, channels, height, width].")

    channel_mean = fields.mean(dim=(0, 2, 3))
    channel_std = fields.std(dim=(0, 2, 3), unbiased=False).clamp_min(eps)
    return channel_mean, channel_std


class PCALSTMForecaster(nn.Module):
    def __init__(
        self,
        *,
        input_steps: int,
        in_channels: int,
        out_channels: int,
        spatial_shape: tuple[int, int],
        output_steps: int,
        input_pca_mean: torch.Tensor,
        input_pca_components: torch.Tensor,
        input_channel_mean: torch.Tensor,
        input_channel_std: torch.Tensor,
        target_pca_mean: torch.Tensor,
        target_pca_components: torch.Tensor,
        target_channel_mean: torch.Tensor,
        target_channel_std: torch.Tensor,
        target_residual_input_indices: torch.Tensor | None = None,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        autoregressive_decoder: bool = False,
    ) -> None:
        super().__init__()
        if lstm_hidden_size < 1:
            raise ValueError("lstm_hidden_size must be at least 1.")
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be at least 1.")

        height, width = spatial_shape
        input_flattened_size = in_channels * height * width
        target_flattened_size = out_channels * height * width
        if input_pca_mean.ndim != 1 or input_pca_mean.numel() != input_flattened_size:
            raise ValueError("input_pca_mean must be a 1D tensor matching input channels * height * width.")
        if input_pca_components.ndim != 2 or input_pca_components.shape[1] != input_flattened_size:
            raise ValueError("input_pca_components must have shape [num_components, input channels * height * width].")
        if input_channel_mean.ndim != 1 or input_channel_mean.numel() != in_channels:
            raise ValueError("input_channel_mean must be a 1D tensor matching the number of input channels.")
        if input_channel_std.ndim != 1 or input_channel_std.numel() != in_channels:
            raise ValueError("input_channel_std must be a 1D tensor matching the number of input channels.")
        if target_pca_mean.ndim != 1 or target_pca_mean.numel() != target_flattened_size:
            raise ValueError("target_pca_mean must be a 1D tensor matching target channels * height * width.")
        if target_pca_components.ndim != 2 or target_pca_components.shape[1] != target_flattened_size:
            raise ValueError("target_pca_components must have shape [num_components, target channels * height * width].")
        if target_channel_mean.ndim != 1 or target_channel_mean.numel() != out_channels:
            raise ValueError("target_channel_mean must be a 1D tensor matching the number of target channels.")
        if target_channel_std.ndim != 1 or target_channel_std.numel() != out_channels:
            raise ValueError("target_channel_std must be a 1D tensor matching the number of target channels.")
        if target_residual_input_indices is None:
            target_residual_input_indices = torch.full((out_channels,), -1, dtype=torch.int64)
        if target_residual_input_indices.ndim != 1 or target_residual_input_indices.numel() != out_channels:
            raise ValueError("target_residual_input_indices must be a 1D tensor matching the number of target channels.")

        input_num_components = int(input_pca_components.shape[0])
        target_num_components = int(target_pca_components.shape[0])
        dropout = lstm_dropout if lstm_layers > 1 else 0.0

        self.input_steps = input_steps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = spatial_shape
        self.output_steps = output_steps
        self.input_num_components = input_num_components
        self.target_num_components = target_num_components
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.autoregressive_decoder = autoregressive_decoder

        self.register_buffer("input_pca_mean", input_pca_mean.to(dtype=torch.float32))
        self.register_buffer("input_pca_components", input_pca_components.to(dtype=torch.float32))
        self.register_buffer("input_channel_mean", input_channel_mean.to(dtype=torch.float32))
        self.register_buffer("input_channel_std", input_channel_std.to(dtype=torch.float32))
        self.register_buffer("target_pca_mean", target_pca_mean.to(dtype=torch.float32))
        self.register_buffer("target_pca_components", target_pca_components.to(dtype=torch.float32))
        self.register_buffer("target_channel_mean", target_channel_mean.to(dtype=torch.float32))
        self.register_buffer("target_channel_std", target_channel_std.to(dtype=torch.float32))
        self.register_buffer("target_residual_input_indices", target_residual_input_indices.to(dtype=torch.int64))

        self.lstm = nn.LSTM(
            input_size=input_num_components,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True,
        )
        if autoregressive_decoder:
            self.decoder = nn.LSTM(
                input_size=target_num_components,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder_start_token = nn.Parameter(torch.zeros(target_num_components, dtype=torch.float32))
            self.readout = nn.Linear(lstm_hidden_size, target_num_components)
        else:
            self.readout = nn.Linear(lstm_hidden_size, output_steps * target_num_components)

    def _normalize_inputs(self, fields: torch.Tensor) -> torch.Tensor:
        view_shape = (1,) * (fields.ndim - 3) + (self.in_channels, 1, 1)
        return (fields - self.input_channel_mean.view(view_shape)) / self.input_channel_std.view(view_shape)

    def _denormalize_targets(self, fields: torch.Tensor) -> torch.Tensor:
        view_shape = (1,) * (fields.ndim - 3) + (self.out_channels, 1, 1)
        return fields * self.target_channel_std.view(view_shape) + self.target_channel_mean.view(view_shape)

    def _normalize_targets(self, fields: torch.Tensor) -> torch.Tensor:
        view_shape = (1,) * (fields.ndim - 3) + (self.out_channels, 1, 1)
        return (fields - self.target_channel_mean.view(view_shape)) / self.target_channel_std.view(view_shape)

    def _project_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized_inputs = self._normalize_inputs(inputs)
        flattened = normalized_inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
        centered = flattened - self.input_pca_mean.view(1, 1, -1)
        return centered @ self.input_pca_components.transpose(0, 1)

    def _reconstruct_targets(self, coefficients: torch.Tensor) -> torch.Tensor:
        reconstructed = coefficients @ self.target_pca_components
        reconstructed = reconstructed + self.target_pca_mean.view(1, 1, -1)
        height, width = self.spatial_shape
        normalized_fields = reconstructed.reshape(coefficients.shape[0], coefficients.shape[1], self.out_channels, height, width)
        return self._denormalize_targets(normalized_fields)

    def _project_targets(self, targets: torch.Tensor) -> torch.Tensor:
        normalized_targets = self._normalize_targets(targets)
        flattened = normalized_targets.reshape(targets.shape[0], targets.shape[1], -1)
        centered = flattened - self.target_pca_mean.view(1, 1, -1)
        return centered @ self.target_pca_components.transpose(0, 1)

    def _add_residual_baseline(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        residual_mask = self.target_residual_input_indices >= 0
        if not torch.any(residual_mask):
            return targets

        outputs = targets.clone()
        baseline_inputs = inputs[:, -1]
        residual_target_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
        for target_index in residual_target_indices.tolist():
            input_index = int(self.target_residual_input_indices[target_index].item())
            baseline = baseline_inputs[:, input_index].unsqueeze(1)
            outputs[:, :, target_index] = outputs[:, :, target_index] + baseline
        return outputs

    def _remove_residual_baseline(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        residual_mask = self.target_residual_input_indices >= 0
        if not torch.any(residual_mask):
            return targets

        outputs = targets.clone()
        baseline_inputs = inputs[:, -1]
        residual_target_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
        for target_index in residual_target_indices.tolist():
            input_index = int(self.target_residual_input_indices[target_index].item())
            baseline = baseline_inputs[:, input_index].unsqueeze(1)
            outputs[:, :, target_index] = outputs[:, :, target_index] - baseline
        return outputs

    def _decode_autoregressive(
        self,
        inputs: torch.Tensor,
        encoder_hidden: tuple[torch.Tensor, torch.Tensor],
        teacher_forcing_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        decoder_input = self.decoder_start_token.view(1, 1, -1).expand(batch_size, 1, -1)
        hidden = encoder_hidden
        predicted_coefficients: list[torch.Tensor] = []

        teacher_forcing_coefficients = None
        if teacher_forcing_targets is not None:
            adjusted_targets = self._remove_residual_baseline(inputs, teacher_forcing_targets)
            teacher_forcing_coefficients = self._project_targets(adjusted_targets)

        for step_index in range(self.output_steps):
            decoder_outputs, hidden = self.decoder(decoder_input, hidden)
            next_coefficients = self.readout(decoder_outputs[:, -1]).unsqueeze(1)
            predicted_coefficients.append(next_coefficients)
            if teacher_forcing_coefficients is not None and step_index + 1 < self.output_steps:
                decoder_input = teacher_forcing_coefficients[:, step_index : step_index + 1]
            else:
                decoder_input = next_coefficients

        return torch.cat(predicted_coefficients, dim=1)

    def forward(self, inputs: torch.Tensor, teacher_forcing_targets: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError("PCALSTMForecaster expects [batch, input_steps, channels, height, width] inputs.")

        batch, steps, channels, height, width = inputs.shape
        if steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, received {steps}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}.")
        if (height, width) != self.spatial_shape:
            raise ValueError(f"Expected spatial shape {self.spatial_shape}, received {(height, width)}.")

        coefficients = self._project_inputs(inputs)
        lstm_outputs, hidden = self.lstm(coefficients)
        if self.autoregressive_decoder:
            predicted_coefficients = self._decode_autoregressive(inputs, hidden, teacher_forcing_targets)
        else:
            predicted_coefficients = self.readout(lstm_outputs[:, -1]).reshape(batch, self.output_steps, self.target_num_components)
        reconstructed_targets = self._reconstruct_targets(predicted_coefficients)
        return self._add_residual_baseline(inputs, reconstructed_targets)