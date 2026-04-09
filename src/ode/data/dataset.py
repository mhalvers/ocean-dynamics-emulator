from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


def _infer_spatial_dims(dataset: xr.Dataset, variables: Sequence[str], time_dim: str) -> tuple[str, ...]:
    variable_dims = dataset[variables[0]].dims
    spatial_dims = tuple(dim for dim in variable_dims if dim != time_dim)
    if len(spatial_dims) != 2:
        raise ValueError(
            "ForecastWindowDataset currently expects exactly two spatial dimensions "
            f"after excluding '{time_dim}', received {spatial_dims}."
        )
    return spatial_dims


class ForecastWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: xr.Dataset,
        *,
        variables: Sequence[str],
        input_steps: int,
        output_steps: int,
        time_dim: str = "time",
        spatial_dims: Sequence[str] | None = None,
    ) -> None:
        missing = [name for name in variables if name not in dataset.data_vars]
        if missing:
            raise KeyError(f"Missing variables in dataset: {missing}")
        if time_dim not in dataset.dims:
            raise KeyError(f"Missing time dimension '{time_dim}' in dataset.")
        if input_steps <= 0 or output_steps <= 0:
            raise ValueError("input_steps and output_steps must be positive integers.")

        resolved_spatial_dims = tuple(spatial_dims) if spatial_dims else _infer_spatial_dims(dataset, variables, time_dim)
        variable_stack = dataset[list(variables)].to_array(dim="channel")
        self.array = variable_stack.transpose(time_dim, "channel", *resolved_spatial_dims)
        self.variables = tuple(variables)
        self.time_dim = time_dim
        self.spatial_dims = resolved_spatial_dims
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_samples = int(self.array.sizes[time_dim]) - input_steps - output_steps + 1
        if self.num_samples <= 0:
            raise ValueError(
                "Not enough timesteps to create forecast windows with the configured "
                f"input_steps={input_steps} and output_steps={output_steps}."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)

        input_slice = self.array.isel({self.time_dim: slice(index, index + self.input_steps)})
        target_slice = self.array.isel(
            {self.time_dim: slice(index + self.input_steps, index + self.input_steps + self.output_steps)}
        )
        inputs = torch.from_numpy(np.asarray(input_slice.values, dtype=np.float32))
        targets = torch.from_numpy(np.asarray(target_slice.values, dtype=np.float32))
        return inputs, targets
