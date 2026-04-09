from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr


HYCOM_FILENAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<day>\d{4}-\d{2}-\d{2})_(?P<kind>ssh|u-v)\.nc$")
HYCOM_RENAME_MAP = {
    "MT": "time",
    "Latitude": "lat",
    "Longitude": "lon",
}


@dataclass(slots=True)
class HycomPreparedDataset:
    dataset: xr.Dataset
    ssh_paths: list[Path]
    uv_paths: list[Path]
    unique_timestamps: int
def _collect_hycom_paths(
    input_dir: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[Path], list[Path]]:
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    ssh_by_day: dict[str, Path] = {}
    uv_by_day: dict[str, Path] = {}
    for path in sorted(directory.glob("*.nc")):
        match = HYCOM_FILENAME_RE.match(path.name)
        if not match:
            continue
        day = match.group("day")
        kind = match.group("kind")
        if kind == "ssh":
            ssh_by_day[day] = path
        else:
            uv_by_day[day] = path

    selected_days = sorted(set(ssh_by_day) | set(uv_by_day))
    if start_date:
        selected_days = [day for day in selected_days if day >= start_date]
    if end_date:
        selected_days = [day for day in selected_days if day <= end_date]
    ssh_by_day = {day: ssh_by_day[day] for day in selected_days if day in ssh_by_day}
    uv_by_day = {day: uv_by_day[day] for day in selected_days if day in uv_by_day}

    common_days = sorted(set(ssh_by_day) & set(uv_by_day))
    if not common_days:
        raise ValueError(f"No matching HYCOM ssh/u-v daily files were found in {directory}.")

    missing_ssh = sorted(set(uv_by_day) - set(ssh_by_day))
    missing_uv = sorted(set(ssh_by_day) - set(uv_by_day))
    if missing_ssh or missing_uv:
        raise ValueError(
            "HYCOM daily files are incomplete. "
            f"Missing ssh days: {missing_ssh or 'none'}. Missing u-v days: {missing_uv or 'none'}."
        )

    return [ssh_by_day[day] for day in common_days], [uv_by_day[day] for day in common_days]


def prepare_hycom_dataset(
    input_dir: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    chunks: Mapping[str, int] | None = None,
    engine: str | None = None,
) -> HycomPreparedDataset:
    ssh_paths, uv_paths = _collect_hycom_paths(input_dir, start_date=start_date, end_date=end_date)

    ssh_ds = xr.open_mfdataset([str(path) for path in ssh_paths], combine="by_coords", chunks=chunks or None, engine=engine)
    uv_ds = xr.open_mfdataset([str(path) for path in uv_paths], combine="by_coords", chunks=chunks or None, engine=engine)
    if "Depth" in uv_ds.dims:
        uv_ds = uv_ds.squeeze("Depth", drop=True)

    merged = xr.merge([ssh_ds[["ssh"]], uv_ds[["u", "v"]]], compat="override", join="inner")
    rename_map = {source: target for source, target in HYCOM_RENAME_MAP.items() if source in merged.dims or source in merged.coords}
    prepared = merged.rename(rename_map).sortby("time")

    time_values = prepared["time"].values
    _, unique_indices = np.unique(time_values, return_index=True)
    prepared = prepared.isel(time=np.sort(unique_indices))

    return HycomPreparedDataset(
        dataset=prepared,
        ssh_paths=ssh_paths,
        uv_paths=uv_paths,
        unique_timestamps=int(prepared.sizes["time"]),
    )


def prepare_hycom_zarr(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    chunks: Mapping[str, int] | None = None,
    engine: str | None = None,
) -> tuple[Path, HycomPreparedDataset]:
    prepared = prepare_hycom_dataset(
        input_dir,
        start_date=start_date,
        end_date=end_date,
        chunks=chunks,
        engine=engine,
    )
    dataset = prepared.dataset.chunk(chunks) if chunks else prepared.dataset

    zarr_path = Path(output_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(zarr_path, mode="w", consolidated=True)
    return zarr_path, prepared
