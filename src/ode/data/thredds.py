from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import urlopen
from xml.etree import ElementTree

from ode.data.pull import DownloadSpec, pull_data

THREDDS_NAMESPACE = {"thredds": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}
DEFAULT_THREDDS_VARIABLES = ("ssh", "u", "v")


@dataclass(slots=True)
class ThreddsDatasetMetadata:
    dataset_path: str
    ncss_base: str
    available_start: date | None = None
    available_end: date | None = None
    vertical_variables: frozenset[str] = frozenset()


@dataclass(slots=True)
class ThreddsRequestWindow:
    available_start: date | None
    available_end: date | None
    effective_start: date | None
    effective_end: date | None


@dataclass(slots=True)
class ThreddsSubsetRequest:
    catalog_url: str
    output_dir: str
    variables: tuple[str, ...] = DEFAULT_THREDDS_VARIABLES
    day: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    time: str | None = None
    north: float | None = None
    south: float | None = None
    east: float | None = None
    west: float | None = None
    vert_coord: str | None = None
    accept: str = "netcdf4"
    horiz_stride: int = 1
    add_latlon: bool = True
    overwrite: bool = False
    output_name: str | None = None
    dataset_id: str | None = None


def _normalize_catalog_url(catalog_url: str) -> tuple[str, str | None]:
    parsed = urlparse(catalog_url)
    dataset_id = parse_qs(parsed.query).get("dataset", [None])[0]
    path = parsed.path
    if path.endswith(".html"):
        path = f"{path[:-5]}.xml"
    normalized = parsed._replace(path=path, query="")
    return urlunparse(normalized), dataset_id


def _read_catalog_xml(catalog_url: str) -> ElementTree.Element:
    with urlopen(catalog_url) as response:
        payload = response.read()
    return ElementTree.fromstring(payload)


def _read_ncss_dataset_xml(ncss_base: str, dataset_path: str) -> ElementTree.Element:
    dataset_url = f"{_join_url(ncss_base, dataset_path)}/dataset.xml"
    with urlopen(dataset_url) as response:
        payload = response.read()
    return ElementTree.fromstring(payload)


def _normalize_service_base(service_base: str, catalog_url: str) -> str:
    if service_base.startswith("//"):
        scheme = urlparse(catalog_url).scheme or "https"
        return f"{scheme}:{service_base}"
    return service_base


def _iter_datasets(root: ElementTree.Element) -> Iterable[ElementTree.Element]:
    return root.findall(".//thredds:dataset", THREDDS_NAMESPACE)


def _resolve_dataset_path(root: ElementTree.Element, dataset_id: str | None) -> str:
    leaf_datasets = [dataset for dataset in _iter_datasets(root) if dataset.get("urlPath")]
    if not leaf_datasets:
        raise ValueError("No downloadable datasets with urlPath were found in the THREDDS catalog.")

    if dataset_id is None:
        if len(leaf_datasets) == 1:
            return leaf_datasets[0].attrib["urlPath"]
        raise ValueError("The THREDDS catalog contains multiple datasets; provide an explicit dataset id.")

    for dataset in leaf_datasets:
        if dataset.get("ID") == dataset_id or dataset.get("name") == dataset_id or dataset.get("urlPath") == dataset_id:
            return dataset.attrib["urlPath"]
    raise ValueError(f"Dataset id '{dataset_id}' was not found in the THREDDS catalog.")


def _resolve_ncss_base(root: ElementTree.Element, catalog_url: str) -> str:
    for service in root.findall(".//thredds:service", THREDDS_NAMESPACE):
        if service.get("serviceType") == "NetcdfSubset":
            base = service.get("base")
            if not base:
                break
            return _normalize_service_base(base, catalog_url)
    raise ValueError("No NetcdfSubset service was found in the THREDDS catalog.")


def _join_url(base: str, suffix: str) -> str:
    return f"{base.rstrip('/')}/{suffix.lstrip('/')}"


def _parse_iso_date(value: str) -> date:
    if "T" in value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    return date.fromisoformat(value)


def _parse_day(day_value: str) -> tuple[str, str]:
    selected_day = date.fromisoformat(day_value)
    start = datetime.combine(selected_day, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _resolve_variables(request: ThreddsSubsetRequest) -> tuple[str, ...]:
    return request.variables or DEFAULT_THREDDS_VARIABLES


def _extract_time_span(root: ElementTree.Element) -> tuple[date | None, date | None]:
    time_span = root.find(".//TimeSpan")
    if time_span is None:
        time_span = root.find(".//timeSpan")
    if time_span is None:
        return None, None

    begin_text = time_span.findtext("begin")
    end_text = time_span.findtext("end")
    available_start = _parse_iso_date(begin_text) if begin_text else None
    available_end = _parse_iso_date(end_text) if end_text else None
    return available_start, available_end


def _extract_vertical_variables(root: ElementTree.Element) -> frozenset[str]:
    vertical_variables: set[str] = set()
    for element in root.iter():
        if element.tag.split("}")[-1] != "grid":
            continue
        variable_name = element.get("name")
        shape = element.get("shape", "")
        if variable_name and "Depth" in shape.split():
            vertical_variables.add(variable_name)
    return frozenset(vertical_variables)


def _resolve_dataset_metadata(request: ThreddsSubsetRequest) -> ThreddsDatasetMetadata:
    catalog_url, inferred_dataset_id = _normalize_catalog_url(request.catalog_url)
    dataset_id = request.dataset_id or inferred_dataset_id
    catalog_root = _read_catalog_xml(catalog_url)
    dataset_path = _resolve_dataset_path(catalog_root, dataset_id)
    ncss_base = _resolve_ncss_base(catalog_root, catalog_url)
    ncss_root = _read_ncss_dataset_xml(ncss_base, dataset_path)
    available_start, available_end = _extract_time_span(ncss_root)
    vertical_variables = _extract_vertical_variables(ncss_root)
    return ThreddsDatasetMetadata(
        dataset_path=dataset_path,
        ncss_base=ncss_base,
        available_start=available_start,
        available_end=available_end,
        vertical_variables=vertical_variables,
    )


def _validate_single_request_against_availability(
    request: ThreddsSubsetRequest,
    metadata: ThreddsDatasetMetadata,
) -> None:
    if metadata.available_start is None or metadata.available_end is None:
        return
    if request.day:
        selected_day = date.fromisoformat(request.day)
        if not metadata.available_start <= selected_day <= metadata.available_end:
            raise ValueError(
                f"Requested day {request.day} is outside the available range "
                f"{metadata.available_start.isoformat()} to {metadata.available_end.isoformat()}."
            )
    if request.time:
        selected_day = _parse_iso_date(request.time)
        if not metadata.available_start <= selected_day <= metadata.available_end:
            raise ValueError(
                f"Requested time {request.time} is outside the available range "
                f"{metadata.available_start.isoformat()} to {metadata.available_end.isoformat()}."
            )


def _resolve_requested_days(
    request: ThreddsSubsetRequest,
    metadata: ThreddsDatasetMetadata,
) -> list[str]:
    if request.day and request.time:
        raise ValueError("Specify either a day window or a single time, not both.")
    if request.day and (request.start_date or request.end_date):
        raise ValueError("Specify either --day or a --start-date/--end-date range, not both.")
    if request.time and (request.start_date or request.end_date):
        raise ValueError("Specify either --time or a --start-date/--end-date range, not both.")

    if not request.start_date and not request.end_date:
        return []
    if not request.start_date or not request.end_date:
        raise ValueError("Specify both --start-date and --end-date for THREDDS range pulls.")

    requested_start = date.fromisoformat(request.start_date)
    requested_end = date.fromisoformat(request.end_date)
    if requested_start > requested_end:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")

    clamped_start = requested_start
    clamped_end = requested_end
    if metadata.available_start is not None:
        clamped_start = max(clamped_start, metadata.available_start)
    if metadata.available_end is not None:
        clamped_end = min(clamped_end, metadata.available_end)
    if clamped_start > clamped_end:
        available_text = (
            f"{metadata.available_start.isoformat()} to {metadata.available_end.isoformat()}"
            if metadata.available_start is not None and metadata.available_end is not None
            else "the server's available time range"
        )
        raise ValueError(f"Requested date range does not overlap {available_text}.")

    return [
        (clamped_start + timedelta(days=offset)).isoformat()
        for offset in range((clamped_end - clamped_start).days + 1)
    ]


def resolve_thredds_request_window(request: ThreddsSubsetRequest) -> ThreddsRequestWindow:
    metadata = _resolve_dataset_metadata(request)
    request_days = _resolve_requested_days(request, metadata)
    effective_start = date.fromisoformat(request_days[0]) if request_days else None
    effective_end = date.fromisoformat(request_days[-1]) if request_days else None

    if not request_days:
        if request.day:
            selected_day = date.fromisoformat(request.day)
            _validate_single_request_against_availability(request, metadata)
            effective_start = selected_day
            effective_end = selected_day
        elif request.time:
            selected_day = _parse_iso_date(request.time)
            _validate_single_request_against_availability(request, metadata)
            effective_start = selected_day
            effective_end = selected_day

    return ThreddsRequestWindow(
        available_start=metadata.available_start,
        available_end=metadata.available_end,
        effective_start=effective_start,
        effective_end=effective_end,
    )


def _build_params(request: ThreddsSubsetRequest) -> list[tuple[str, str]]:
    variables = _resolve_variables(request)
    if not request.day and not request.time:
        raise ValueError("Specify --day, --time, or an inclusive --start-date/--end-date range for THREDDS pulls.")

    params: list[tuple[str, str]] = [("var", variable) for variable in variables]
    params.append(("horizStride", str(request.horiz_stride)))
    params.append(("accept", request.accept))
    if request.add_latlon:
        params.append(("addLatLon", "true"))

    if request.time:
        params.append(("time", request.time))
    else:
        time_start, time_end = _parse_day(request.day or "")
        params.append(("time_start", time_start))
        params.append(("time_end", time_end))

    bbox_values = (request.north, request.south, request.east, request.west)
    if all(value is None for value in bbox_values):
        params.append(("disableLLSubset", "on"))
        params.append(("disableProjSubset", "on"))
    elif any(value is None for value in bbox_values):
        raise ValueError("Provide north, south, east, and west together for spatial subsetting.")
    else:
        if request.north is not None and request.south is not None and request.north < request.south:
            raise ValueError("north must be greater than or equal to south for THREDDS bounding boxes.")
        if request.east is not None and request.west is not None and request.east < request.west:
            raise ValueError(
                "east must be greater than or equal to west for THREDDS bounding boxes. "
                "For western longitudes, a box from 82W to 80W should be passed as --west -82 --east -80."
            )
        params.extend(
            [
                ("north", str(request.north)),
                ("south", str(request.south)),
                ("east", str(request.east)),
                ("west", str(request.west)),
            ]
        )

    resolved_vert_coord = request.vert_coord
    if resolved_vert_coord is None and any(variable in {"u", "v"} for variable in variables):
        resolved_vert_coord = "0.0"

    if resolved_vert_coord is not None:
        params.append(("vertCoord", resolved_vert_coord))
    return params


def _default_output_name(dataset_path: str, request: ThreddsSubsetRequest) -> str:
    dataset_stem = dataset_path.strip("/").replace("/", "_")
    time_tag = request.time.replace(":", "-") if request.time else (request.day or "subset")
    variable_tag = "-".join(_resolve_variables(request))
    return f"{dataset_stem}_{time_tag}_{variable_tag}.nc"


def _split_request_by_variable_shape(
    request: ThreddsSubsetRequest,
    metadata: ThreddsDatasetMetadata,
) -> list[ThreddsSubsetRequest]:
    variables = _resolve_variables(request)
    vertical_variables = tuple(variable for variable in variables if variable in metadata.vertical_variables)
    surface_variables = tuple(variable for variable in variables if variable not in metadata.vertical_variables)

    grouped_requests: list[ThreddsSubsetRequest] = []
    if surface_variables:
        grouped_requests.append(replace(request, variables=surface_variables, vert_coord=None))
    if vertical_variables:
        grouped_requests.append(replace(request, variables=vertical_variables))
    return grouped_requests or [request]


def _build_download_spec(request: ThreddsSubsetRequest, metadata: ThreddsDatasetMetadata) -> DownloadSpec:
    url = f"{_join_url(metadata.ncss_base, metadata.dataset_path)}?{urlencode(_build_params(request), doseq=True)}"
    output_name = request.output_name or _default_output_name(metadata.dataset_path, request)
    return DownloadSpec(url=url, path=output_name)


def build_thredds_download_specs(request: ThreddsSubsetRequest) -> list[DownloadSpec]:
    metadata = _resolve_dataset_metadata(request)
    request_days = _resolve_requested_days(request, metadata)
    expanded_requests: list[ThreddsSubsetRequest] = []

    if request_days:
        for day in request_days:
            day_request = replace(request, day=day, start_date=None, end_date=None)
            expanded_requests.extend(_split_request_by_variable_shape(day_request, metadata))
    else:
        _validate_single_request_against_availability(request, metadata)
        expanded_requests.extend(_split_request_by_variable_shape(request, metadata))

    if request.output_name and len(expanded_requests) > 1:
        raise ValueError(
            "--output-name is only supported when a THREDDS pull produces exactly one file. "
            "Mixed-dimensional variables like ssh with surface currents produce multiple files."
        )

    return [
        _build_download_spec(
            replace(expanded_request, output_name=request.output_name if len(expanded_requests) == 1 else None),
            metadata,
        )
        for expanded_request in expanded_requests
    ]


def build_thredds_download_spec(request: ThreddsSubsetRequest) -> DownloadSpec:
    specs = build_thredds_download_specs(request)
    if len(specs) != 1:
        raise ValueError("Expected exactly one THREDDS download spec. Use build_thredds_download_specs for date ranges.")
    return specs[0]


def pull_thredds_catalog(request: ThreddsSubsetRequest) -> list[Path]:
    specs = build_thredds_download_specs(request)
    return pull_data(
        output_dir=request.output_dir,
        urls=(),
        manifest_path=None,
        overwrite=request.overwrite,
        specs=specs,
    )