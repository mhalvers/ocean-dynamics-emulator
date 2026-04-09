from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import xarray as xr


def animate_surface_dataset(
    dataset: xr.Dataset,
    animation_path: str | Path,
    *,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    ssh_name: str = "ssh",
    u_name: str = "u",
    v_name: str = "v",
    fps: int = 2,
    quiver_stride: int | None = None,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be a positive integer.")

    animation_file = Path(animation_path)
    if animation_file.suffix.lower() != ".gif":
        raise ValueError("animation_path must use a .gif extension.")

    required = (ssh_name, u_name, v_name)
    missing = [name for name in required if name not in dataset.data_vars]
    if missing:
        raise KeyError(f"Missing required variables for animation: {missing}")
    for dim_name in (time_dim, lat_dim, lon_dim):
        if dim_name not in dataset.dims:
            raise KeyError(f"Missing required dimension '{dim_name}' for animation.")

    ssh = np.asarray(dataset[ssh_name].values)
    u = np.asarray(dataset[u_name].values)
    v = np.asarray(dataset[v_name].values)
    time_values = np.asarray(dataset[time_dim].values)
    latitudes = np.asarray(dataset[lat_dim].values)
    longitudes = np.asarray(dataset[lon_dim].values)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    stride = quiver_stride if quiver_stride is not None else max(1, min(len(latitudes), len(longitudes)) // 20)
    if stride <= 0:
        raise ValueError("quiver_stride must be a positive integer.")

    ssh_min = float(ssh.min())
    ssh_max = float(ssh.max())
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    mesh = axis.pcolormesh(longitudes, latitudes, ssh[0], shading="auto", cmap="viridis", vmin=ssh_min, vmax=ssh_max)
    quiver = axis.quiver(
        lon_grid[::stride, ::stride],
        lat_grid[::stride, ::stride],
        u[0, ::stride, ::stride],
        v[0, ::stride, ::stride],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    axis.set_xlabel(lon_dim)
    axis.set_ylabel(lat_dim)
    title = axis.set_title(f"Raw data at {np.datetime_as_string(time_values[0], unit='s') if np.issubdtype(time_values.dtype, np.datetime64) else time_values[0]}")
    figure.colorbar(mesh, ax=axis, label=ssh_name)

    def _format_time(frame_index: int) -> str:
        value = np.asarray(time_values[frame_index])
        if np.issubdtype(value.dtype, np.datetime64):
            return np.datetime_as_string(value, unit="s")
        return str(value.item() if value.ndim == 0 else value)

    def _update(frame_index: int):
        mesh.set_array(ssh[frame_index].ravel())
        quiver.set_UVC(u[frame_index, ::stride, ::stride], v[frame_index, ::stride, ::stride])
        title.set_text(f"Raw data at {_format_time(frame_index)}")
        return mesh, quiver, title

    animation = FuncAnimation(figure, _update, frames=ssh.shape[0], interval=max(int(1000 / fps), 1), blit=False)
    animation_file.parent.mkdir(parents=True, exist_ok=True)
    animation.save(animation_file, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return animation_file


__all__ = ["animate_surface_dataset"]