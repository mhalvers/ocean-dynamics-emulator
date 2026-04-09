from __future__ import annotations

import torch
from torch import nn


class TemporalConvForecaster(nn.Module):
    def __init__(
        self,
        *,
        input_steps: int,
        in_channels: int,
        output_steps: int,
        hidden_channels: int = 64,
        depth: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so spatial resolution is preserved.")

        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(input_steps * in_channels, hidden_channels, kernel_size, padding=padding),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, output_steps * in_channels, kernel_size=1))

        self.network = nn.Sequential(*layers)
        self.input_steps = input_steps
        self.in_channels = in_channels
        self.output_steps = output_steps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                "TemporalConvForecaster expects input tensors shaped as "
                "[batch, input_steps, channels, height, width]."
            )
        batch, steps, channels, height, width = inputs.shape
        if steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, received {steps}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}.")

        flattened = inputs.reshape(batch, steps * channels, height, width)
        outputs = self.network(flattened)
        return outputs.reshape(batch, self.output_steps, channels, height, width)
