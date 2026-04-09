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


class PCALSTMForecaster(nn.Module):
    def __init__(
        self,
        *,
        input_steps: int,
        in_channels: int,
        spatial_shape: tuple[int, int],
        output_steps: int,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if lstm_hidden_size < 1:
            raise ValueError("lstm_hidden_size must be at least 1.")
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be at least 1.")

        height, width = spatial_shape
        flattened_size = in_channels * height * width
        if pca_mean.ndim != 1 or pca_mean.numel() != flattened_size:
            raise ValueError("pca_mean must be a 1D tensor matching channels * height * width.")
        if pca_components.ndim != 2 or pca_components.shape[1] != flattened_size:
            raise ValueError("pca_components must have shape [num_components, channels * height * width].")

        num_components = int(pca_components.shape[0])
        dropout = lstm_dropout if lstm_layers > 1 else 0.0

        self.input_steps = input_steps
        self.in_channels = in_channels
        self.spatial_shape = spatial_shape
        self.output_steps = output_steps
        self.num_components = num_components
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        self.register_buffer("pca_mean", pca_mean.to(dtype=torch.float32))
        self.register_buffer("pca_components", pca_components.to(dtype=torch.float32))

        self.lstm = nn.LSTM(
            input_size=num_components,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.readout = nn.Linear(lstm_hidden_size, output_steps * num_components)

    def _project(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
        centered = flattened - self.pca_mean.view(1, 1, -1)
        return centered @ self.pca_components.transpose(0, 1)

    def _reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        reconstructed = coefficients @ self.pca_components
        reconstructed = reconstructed + self.pca_mean.view(1, 1, -1)
        height, width = self.spatial_shape
        return reconstructed.reshape(coefficients.shape[0], coefficients.shape[1], self.in_channels, height, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError("PCALSTMForecaster expects [batch, input_steps, channels, height, width] inputs.")

        batch, steps, channels, height, width = inputs.shape
        if steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, received {steps}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}.")
        if (height, width) != self.spatial_shape:
            raise ValueError(f"Expected spatial shape {self.spatial_shape}, received {(height, width)}.")

        coefficients = self._project(inputs)
        lstm_outputs, _ = self.lstm(coefficients)
        predicted_coefficients = self.readout(lstm_outputs[:, -1]).reshape(batch, self.output_steps, self.num_components)
        return self._reconstruct(predicted_coefficients)