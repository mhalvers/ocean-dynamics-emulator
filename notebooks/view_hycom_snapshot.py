# %% [markdown]
# # View One HYCOM Timestamp
#
# This notebook-style Python script opens one HYCOM SSH file and one HYCOM
# surface-current file for the same day, merges them, and plots one timestamp.
#
# Run it in VS Code as a Python notebook script with the Interactive Window.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DATA_DIR = Path("data/raw/hycom")
TARGET_DAY = "2019-03-01"
TIME_INDEX = 0
QUIVER_STRIDE = 3


def _resolve_daily_files(data_dir: Path, day: str) -> tuple[Path, Path]:
    ssh_path = data_dir / f"GOMl0.04_expt_32.5_{day}_ssh.nc"
    uv_path = data_dir / f"GOMl0.04_expt_32.5_{day}_u-v.nc"
    if not ssh_path.exists():
        raise FileNotFoundError(f"Missing SSH file: {ssh_path}")
    if not uv_path.exists():
        raise FileNotFoundError(f"Missing surface current file: {uv_path}")
    return ssh_path, uv_path


def load_daily_dataset(data_dir: Path, day: str) -> xr.Dataset:
    ssh_path, uv_path = _resolve_daily_files(data_dir, day)
    ssh_ds = xr.open_dataset(ssh_path)
    uv_ds = xr.open_dataset(uv_path)
    uv_surface = uv_ds.squeeze("Depth", drop=True) if "Depth" in uv_ds.dims else uv_ds
    dataset = xr.merge([ssh_ds[["ssh"]], uv_surface[["u", "v"]]], compat="override")
    return dataset


dataset = load_daily_dataset(DATA_DIR, TARGET_DAY)
dataset

# %% [markdown]
# ## Inspect Available Timestamps
#
# Use `TIME_INDEX` above to choose which timestamp to display.

# %%
timestamps = dataset["MT"].values
print(f"Available timestamps for {TARGET_DAY}:")
for idx, timestamp in enumerate(timestamps):
    print(f"  {idx}: {timestamp}")

# %% [markdown]
# ## Select One Timestamp

# %%
snapshot = dataset.isel(MT=TIME_INDEX)
selected_timestamp = np.datetime_as_string(snapshot["MT"].values, unit="s")
print(f"Selected timestamp: {selected_timestamp}")
snapshot

# %% [markdown]
# ## Plot SSH, Surface U, and Surface V

# %%
latitude = snapshot["Latitude"].values
longitude = snapshot["Longitude"].values
lon_grid, lat_grid = np.meshgrid(longitude, latitude)

ssh = snapshot["ssh"].values
u = snapshot["u"].values
v = snapshot["v"].values
speed = np.hypot(u, v)

figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

ssh_plot = axes[0].pcolormesh(longitude, latitude, ssh, shading="auto", cmap="viridis")
axes[0].quiver(
    lon_grid[::QUIVER_STRIDE, ::QUIVER_STRIDE],
    lat_grid[::QUIVER_STRIDE, ::QUIVER_STRIDE],
    u[::QUIVER_STRIDE, ::QUIVER_STRIDE],
    v[::QUIVER_STRIDE, ::QUIVER_STRIDE],
    speed[::QUIVER_STRIDE, ::QUIVER_STRIDE],
    cmap="magma",
    scale=6,
)
axes[0].set_title(f"SSH\n{selected_timestamp}")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
figure.colorbar(ssh_plot, ax=axes[0], label="m")

u_plot = axes[1].pcolormesh(longitude, latitude, u, shading="auto", cmap="coolwarm")
axes[1].set_title("Surface U")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
figure.colorbar(u_plot, ax=axes[1], label="m/s")

v_plot = axes[2].pcolormesh(longitude, latitude, v, shading="auto", cmap="coolwarm")
axes[2].set_title("Surface V")
axes[2].set_xlabel("Longitude")
axes[2].set_ylabel("Latitude")
figure.colorbar(v_plot, ax=axes[2], label="m/s")

plt.show()
