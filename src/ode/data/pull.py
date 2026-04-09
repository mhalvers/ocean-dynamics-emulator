from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass(slots=True)
class DownloadSpec:
    url: str
    path: str | None = None
    sha256: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "DownloadSpec":
        url = payload.get("url")
        if not url:
            raise ValueError("Manifest entries must define a url.")
        return cls(url=url, path=payload.get("path"), sha256=payload.get("sha256"))


@dataclass(slots=True)
class DownloadResult:
    path: Path
    status: str


DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0


def load_download_manifest(path: str | Path) -> list[DownloadSpec]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        entries = payload.get("files")
        if entries is None:
            raise ValueError("Manifest objects must include a 'files' list.")
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("Manifest must be either a list or an object with a 'files' key.")
    return [DownloadSpec.from_dict(entry) for entry in entries]


def _filename_from_url(url: str) -> str:
    filename = Path(urlparse(url).path).name
    if not filename:
        raise ValueError(f"Unable to infer a filename from URL '{url}'. Provide an explicit path in the manifest.")
    return filename


def _resolve_destination(output_dir: str | Path, relative_path: str | None, url: str) -> Path:
    base_dir = Path(output_dir)
    candidate = Path(relative_path) if relative_path else Path(_filename_from_url(url))
    if candidate.is_absolute():
        raise ValueError(f"Download destination must be relative to the output directory, received '{candidate}'.")

    destination = (base_dir / candidate).resolve()
    output_root = base_dir.resolve()
    if output_root not in destination.parents and destination != output_root:
        raise ValueError(f"Download destination '{destination}' escapes output directory '{output_root}'.")
    return destination


def _iter_specs(
    urls: Sequence[str],
    manifest_path: str | Path | None,
    inline_specs: Sequence[DownloadSpec] = (),
) -> list[DownloadSpec]:
    specs = [DownloadSpec(url=url) for url in urls]
    specs.extend(inline_specs)
    if manifest_path:
        specs.extend(load_download_manifest(manifest_path))
    if not specs:
        raise ValueError("Provide at least one --url or a --manifest file.")
    return specs


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch for '{path}'. Expected {expected}, received {digest}.")


def _download_to_path(
    url: str,
    destination: Path,
    *,
    overwrite: bool,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> DownloadResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return DownloadResult(path=destination, status="skipped")

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(url) as response, NamedTemporaryFile(delete=False, dir=destination.parent) as handle:
                temp_path = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            temp_path.replace(destination)
            return DownloadResult(path=destination, status="downloaded")
        except HTTPError as exc:
            body_bytes = exc.read() if hasattr(exc, "read") else b""
            body = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
            last_error = RuntimeError(f"HTTP {exc.code} while downloading {url}: {body[:500]}")
            if not 500 <= exc.code < 600 or attempt >= max_retries:
                raise last_error from exc
        except URLError as exc:
            last_error = RuntimeError(f"Network error while downloading {url}: {exc.reason}")
            if attempt >= max_retries:
                raise last_error from exc

        time.sleep(retry_delay_seconds * (2**attempt))

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to download {url} for an unknown reason.")


def pull_data(
    *,
    output_dir: str | Path,
    urls: Sequence[str] = (),
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
    specs: Sequence[DownloadSpec] = (),
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> list[DownloadResult]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    downloaded: list[DownloadResult] = []
    for spec in _iter_specs(urls, manifest_path, specs):
        destination = _resolve_destination(output_root, spec.path, spec.url)
        result = _download_to_path(
            spec.url,
            destination,
            overwrite=overwrite,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        if spec.sha256:
            _verify_sha256(result.path, spec.sha256)
        downloaded.append(result)
    return downloaded
