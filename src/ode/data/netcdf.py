from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import xarray as xr


def _normalize_paths(paths: Sequence[str | Path]) -> list[Path]:
    normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError("At least one NetCDF or Zarr path is required.")
    return normalized


def open_netcdf_dataset(
    paths: Sequence[str | Path],
    *,
    chunks: Mapping[str, int] | None = None,
    engine: str | None = None,
) -> xr.Dataset:
    normalized = _normalize_paths(paths)
    if len(normalized) == 1:
        return xr.open_dataset(normalized[0], chunks=chunks or None, engine=engine)

    return xr.open_mfdataset(
        [str(path) for path in normalized],
        combine="by_coords",
        chunks=chunks or None,
        engine=engine,
    )


def convert_netcdf_to_zarr(
    netcdf_paths: Sequence[str | Path],
    zarr_path: str | Path,
    *,
    chunks: Mapping[str, int] | None = None,
    engine: str | None = None,
) -> Path:
    dataset = open_netcdf_dataset(netcdf_paths, chunks=chunks, engine=engine)
    if chunks:
        dataset = dataset.chunk(chunks)

    store_path = Path(zarr_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(store_path, mode="w", consolidated=True)
    return store_path


def open_training_dataset(
    *,
    paths: Sequence[str | Path],
    zarr_path: str | Path | None,
    chunks: Mapping[str, int] | None = None,
    engine: str | None = None,
) -> tuple[xr.Dataset, Path | None]:
    if zarr_path:
        store_path = Path(zarr_path)
        if not store_path.exists():
            convert_netcdf_to_zarr(paths, store_path, chunks=chunks, engine=engine)
        return xr.open_zarr(store_path, consolidated=True), store_path

    normalized = _normalize_paths(paths)
    first_path = normalized[0]
    if len(normalized) == 1 and first_path.suffix == ".zarr":
        return xr.open_zarr(first_path, consolidated=True), first_path

    return open_netcdf_dataset(normalized, chunks=chunks, engine=engine), None
