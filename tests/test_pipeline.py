from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest
import xarray as xr
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from ode.cli import main
from ode.config import DataConfig, ModelConfig, OptimizationConfig, TrainingConfig
from ode.data.dataset import ForecastWindowDataset
from ode.data.hycom import prepare_hycom_dataset
from ode.data.visualization import animate_surface_dataset
from ode.inference import plot_point_timeseries, plot_prediction, predict_window
from ode.data.netcdf import convert_netcdf_to_zarr, open_training_dataset
from ode.data.pull import _download_to_path, pull_data
from ode.data.thredds import DEFAULT_THREDDS_VARIABLES, ThreddsSubsetRequest, build_thredds_download_spec, build_thredds_download_specs, resolve_thredds_request_window
from ode.models.pca_lstm import PCALSTMForecaster, compute_principal_components
from ode.training.engine import fit, save_checkpoint


def _write_sample_netcdf(path) -> None:
    time = np.arange(10)
    lat = np.linspace(-1.0, 1.0, 4)
    lon = np.linspace(120.0, 122.0, 5)
    base = np.arange(time.size * lat.size * lon.size, dtype=np.float32).reshape(time.size, lat.size, lon.size)
    dataset = xr.Dataset(
        data_vars={
            "ssh": (("time", "lat", "lon"), base),
            "u": (("time", "lat", "lon"), base * 0.1),
            "v": (("time", "lat", "lon"), base * -0.1),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    dataset.to_netcdf(path)


def _write_sample_hycom_split_files(base_dir, day: str, time_values) -> None:
    lat = np.linspace(20.0, 21.0, 3)
    lon = np.linspace(-82.0, -80.0, 4)
    depth = np.array([0.0], dtype=np.float32)
    ssh_base = np.arange(len(time_values) * lat.size * lon.size, dtype=np.float32).reshape(len(time_values), lat.size, lon.size)
    uv_base = np.arange(len(time_values) * depth.size * lat.size * lon.size, dtype=np.float32).reshape(len(time_values), depth.size, lat.size, lon.size)

    ssh_ds = xr.Dataset(
        data_vars={"ssh": (("MT", "Latitude", "Longitude"), ssh_base)},
        coords={"MT": np.array(time_values, dtype="datetime64[ns]"), "Latitude": lat, "Longitude": lon},
    )
    uv_ds = xr.Dataset(
        data_vars={
            "u": (("MT", "Depth", "Latitude", "Longitude"), uv_base),
            "v": (("MT", "Depth", "Latitude", "Longitude"), uv_base * -1.0),
        },
        coords={"MT": np.array(time_values, dtype="datetime64[ns]"), "Depth": depth, "Latitude": lat, "Longitude": lon},
    )

    ssh_ds.to_netcdf(base_dir / f"GOMl0.04_expt_32.5_{day}_ssh.nc")
    uv_ds.to_netcdf(base_dir / f"GOMl0.04_expt_32.5_{day}_u-v.nc")


def test_zarr_conversion_and_window_dataset(tmp_path) -> None:
    netcdf_path = tmp_path / "sample.nc"
    zarr_path = tmp_path / "sample.zarr"
    _write_sample_netcdf(netcdf_path)

    convert_netcdf_to_zarr([netcdf_path], zarr_path, chunks={"time": 2})
    dataset, store_path = open_training_dataset(paths=[netcdf_path], zarr_path=zarr_path, chunks={"time": 2})

    assert store_path == zarr_path
    window_dataset = ForecastWindowDataset(
        dataset,
        variables=("ssh", "u", "v"),
        input_steps=3,
        output_steps=2,
        time_dim="time",
        spatial_dims=("lat", "lon"),
    )
    inputs, targets = window_dataset[0]

    assert inputs.shape == (3, 3, 4, 5)
    assert targets.shape == (2, 3, 4, 5)
    assert len(window_dataset) == 6


def test_model_forward_and_training(tmp_path) -> None:
    netcdf_path = tmp_path / "train.nc"
    zarr_path = tmp_path / "train.zarr"
    _write_sample_netcdf(netcdf_path)

    config = TrainingConfig(
        data=DataConfig(
            paths=[str(netcdf_path)],
            zarr_path=str(zarr_path),
            variables=("ssh", "u", "v"),
            time_dim="time",
            spatial_dims=("lat", "lon"),
            input_steps=3,
            output_steps=1,
            batch_size=2,
            train_fraction=0.75,
            chunks={"time": 2},
        ),
        model=ModelConfig(pca_components=4, lstm_hidden_size=8, lstm_layers=1, lstm_dropout=0.0),
        optimization=OptimizationConfig(epochs=1, learning_rate=1e-3, device="cpu"),
    )

    result = fit(config)
    assert result.history["train_loss"]

    sample_fields = __import__("torch").randn(6, 3, 4, 5)
    pca_mean, pca_components, _ = compute_principal_components(sample_fields, num_components=4)
    model = PCALSTMForecaster(
        input_steps=3,
        in_channels=3,
        spatial_shape=(4, 5),
        output_steps=1,
        pca_mean=pca_mean,
        pca_components=pca_components,
        lstm_hidden_size=8,
        lstm_layers=1,
        lstm_dropout=0.0,
    )
    sample = np.random.randn(2, 3, 3, 4, 5).astype(np.float32)
    output = model.forward(__import__("torch").from_numpy(sample))

    assert output.shape == (2, 1, 3, 4, 5)


def test_predict_window_loads_checkpoint_and_saves_plot(tmp_path) -> None:
    netcdf_path = tmp_path / "predict.nc"
    zarr_path = tmp_path / "predict.zarr"
    checkpoint_path = tmp_path / "predict.pt"
    figure_path = tmp_path / "predict.png"
    _write_sample_netcdf(netcdf_path)

    config = TrainingConfig(
        data=DataConfig(
            paths=[str(netcdf_path)],
            zarr_path=str(zarr_path),
            variables=("ssh", "u", "v"),
            time_dim="time",
            spatial_dims=("lat", "lon"),
            input_steps=3,
            output_steps=1,
            batch_size=2,
            train_fraction=0.75,
            chunks={"time": 2},
        ),
        model=ModelConfig(pca_components=4, lstm_hidden_size=8, lstm_layers=1, lstm_dropout=0.0),
        optimization=OptimizationConfig(epochs=1, learning_rate=1e-3, device="cpu"),
    )

    result = fit(config)
    save_checkpoint(result, config, checkpoint_path)

    prediction = predict_window(checkpoint_path, split="val", sample_index=0, device="cpu")

    assert prediction.predictions.shape == (1, 3, 4, 5)
    assert prediction.targets.shape == (1, 3, 4, 5)
    assert prediction.inputs.shape == (3, 3, 4, 5)
    assert prediction.data_store == str(zarr_path)
    assert prediction.mse >= 0.0
    assert len(prediction.input_times) == 3
    assert len(prediction.target_times) == 1

    saved_path = plot_prediction(prediction, forecast_step=0, figure_path=figure_path)
    assert saved_path == figure_path
    assert figure_path.exists()


def test_predict_window_supports_autoregressive_lead_steps(tmp_path) -> None:
    netcdf_path = tmp_path / "lead.nc"
    zarr_path = tmp_path / "lead.zarr"
    checkpoint_path = tmp_path / "lead.pt"
    figure_path = tmp_path / "lead.png"
    _write_sample_netcdf(netcdf_path)

    config = TrainingConfig(
        data=DataConfig(
            paths=[str(netcdf_path)],
            zarr_path=str(zarr_path),
            variables=("ssh", "u", "v"),
            time_dim="time",
            spatial_dims=("lat", "lon"),
            input_steps=3,
            output_steps=1,
            batch_size=2,
            train_fraction=0.75,
            chunks={"time": 2},
        ),
        model=ModelConfig(pca_components=4, lstm_hidden_size=8, lstm_layers=1, lstm_dropout=0.0),
        optimization=OptimizationConfig(epochs=1, learning_rate=1e-3, device="cpu"),
    )

    result = fit(config)
    save_checkpoint(result, config, checkpoint_path)

    prediction = predict_window(checkpoint_path, split="all", sample_index=0, lead_steps=7, device="cpu")

    assert prediction.lead_steps == 7
    assert prediction.predictions.shape == (7, 3, 4, 5)
    assert prediction.targets.shape == (7, 3, 4, 5)
    assert len(prediction.target_times) == 7

    saved_path = plot_prediction(prediction, forecast_step=6, figure_path=figure_path)
    assert saved_path == figure_path
    assert figure_path.exists()


def test_plot_point_timeseries_uses_seeded_random_point(tmp_path) -> None:
    netcdf_path = tmp_path / "point.nc"
    zarr_path = tmp_path / "point.zarr"
    checkpoint_path = tmp_path / "point.pt"
    figure_path = tmp_path / "point_timeseries.png"
    _write_sample_netcdf(netcdf_path)

    config = TrainingConfig(
        data=DataConfig(
            paths=[str(netcdf_path)],
            zarr_path=str(zarr_path),
            variables=("ssh", "u", "v"),
            time_dim="time",
            spatial_dims=("lat", "lon"),
            input_steps=3,
            output_steps=1,
            batch_size=2,
            train_fraction=0.75,
            chunks={"time": 2},
        ),
        model=ModelConfig(pca_components=4, lstm_hidden_size=8, lstm_layers=1, lstm_dropout=0.0),
        optimization=OptimizationConfig(epochs=1, learning_rate=1e-3, device="cpu"),
    )

    result = fit(config)
    save_checkpoint(result, config, checkpoint_path)
    prediction = predict_window(checkpoint_path, split="val", sample_index=0, lead_steps=3, device="cpu")

    saved_path, selection = plot_point_timeseries(prediction, seed=123, figure_path=figure_path)
    rng = np.random.default_rng(123)
    expected_index = (int(rng.integers(4)), int(rng.integers(5)))
    lat_values = np.linspace(-1.0, 1.0, 4)
    lon_values = np.linspace(120.0, 122.0, 5)

    assert saved_path == figure_path
    assert figure_path.exists()
    assert selection.point_index == expected_index
    assert selection.point_coordinates == (lat_values[expected_index[0]], lon_values[expected_index[1]])


def test_animate_surface_dataset_saves_gif(tmp_path) -> None:
    raw_dir = tmp_path / "hycom"
    raw_dir.mkdir()
    _write_sample_hycom_split_files(raw_dir, "2019-03-01", ["2019-03-01", "2019-03-02"])
    _write_sample_hycom_split_files(raw_dir, "2019-03-02", ["2019-03-02", "2019-03-03"])

    prepared = prepare_hycom_dataset(raw_dir)
    animation_path = tmp_path / "raw.gif"

    saved_path = animate_surface_dataset(prepared.dataset, animation_path, fps=2, quiver_stride=1)

    assert saved_path == animation_path
    assert animation_path.exists()
    assert animation_path.stat().st_size > 0


def test_prepare_hycom_dataset_merges_and_deduplicates_times(tmp_path) -> None:
    raw_dir = tmp_path / "hycom"
    raw_dir.mkdir()
    _write_sample_hycom_split_files(raw_dir, "2019-03-01", ["2019-03-01", "2019-03-02"])
    _write_sample_hycom_split_files(raw_dir, "2019-03-02", ["2019-03-02", "2019-03-03"])

    prepared = prepare_hycom_dataset(raw_dir)

    assert list(prepared.dataset.data_vars) == ["ssh", "u", "v"]
    assert dict(prepared.dataset.sizes) == {"time": 3, "lat": 3, "lon": 4}
    assert "Depth" not in prepared.dataset.dims
    assert [str(value) for value in prepared.dataset["time"].values] == [
        "2019-03-01T00:00:00.000000000",
        "2019-03-02T00:00:00.000000000",
        "2019-03-03T00:00:00.000000000",
    ]


def test_prepare_hycom_dataset_honors_date_filters(tmp_path) -> None:
    raw_dir = tmp_path / "hycom"
    raw_dir.mkdir()
    _write_sample_hycom_split_files(raw_dir, "2018-12-01", ["2018-12-01", "2018-12-02"])
    _write_sample_hycom_split_files(raw_dir, "2019-03-01", ["2019-03-01", "2019-03-02"])
    _write_sample_hycom_split_files(raw_dir, "2019-03-02", ["2019-03-02", "2019-03-03"])

    prepared = prepare_hycom_dataset(raw_dir, start_date="2019-03-01", end_date="2019-03-02")

    assert [path.name for path in prepared.ssh_paths] == [
        "GOMl0.04_expt_32.5_2019-03-01_ssh.nc",
        "GOMl0.04_expt_32.5_2019-03-02_ssh.nc",
    ]
    assert [str(value) for value in prepared.dataset["time"].values] == [
        "2019-03-01T00:00:00.000000000",
        "2019-03-02T00:00:00.000000000",
        "2019-03-03T00:00:00.000000000",
    ]


def test_pull_data_with_manifest(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "ssh.nc"
    source_bytes = b"synthetic netcdf bytes"
    source_path.write_bytes(source_bytes)

    manifest_path = tmp_path / "manifest.json"
    checksum = hashlib.sha256(source_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "url": source_path.as_uri(),
                        "path": "raw/ssh.nc",
                        "sha256": checksum,
                    }
                ]
            }
        )
    )

    output_dir = tmp_path / "downloads"
    paths = pull_data(output_dir=output_dir, manifest_path=manifest_path)

    assert [result.path for result in paths] == [output_dir / "raw" / "ssh.nc"]
    assert paths[0].status == "downloaded"
    assert paths[0].path.read_bytes() == source_bytes


def test_cli_pull_command(tmp_path, monkeypatch, capsys) -> None:
    source_path = tmp_path / "currents.nc"
    source_path.write_bytes(b"currents data")
    output_dir = tmp_path / "cache"

    monkeypatch.setattr(
        "sys.argv",
        [
            "ode",
            "pull",
            "--url",
            source_path.as_uri(),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()
    stdout = capsys.readouterr().out

    downloaded_path = output_dir / "currents.nc"
    assert downloaded_path.exists()
    assert "Pulled" in stdout


def test_cli_prepare_hycom_command(tmp_path, monkeypatch, capsys) -> None:
    raw_dir = tmp_path / "hycom"
    raw_dir.mkdir()
    _write_sample_hycom_split_files(raw_dir, "2019-03-01", ["2019-03-01", "2019-03-02"])
    _write_sample_hycom_split_files(raw_dir, "2019-03-02", ["2019-03-02", "2019-03-03"])
    output_path = tmp_path / "prepared.zarr"

    monkeypatch.setattr(
        "sys.argv",
        [
            "ode",
            "prepare-hycom",
            "--input-dir",
            str(raw_dir),
            "--output",
            str(output_path),
        ],
    )

    main()
    stdout = capsys.readouterr().out

    assert output_path.exists()
    assert "Prepared HYCOM training store" in stdout
    assert "3 unique timestamps" in stdout


def test_pull_data_marks_existing_files_as_skipped(tmp_path) -> None:
    source_path = tmp_path / "ssh.nc"
    source_path.write_bytes(b"existing")
    output_dir = tmp_path / "downloads"

    first = pull_data(output_dir=output_dir, urls=[source_path.as_uri()])
    second = pull_data(output_dir=output_dir, urls=[source_path.as_uri()])

    assert first[0].status == "downloaded"
    assert second[0].status == "skipped"


def test_download_to_path_retries_transient_http_500(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "retry.nc"
    calls = {"count": 0}

    class DummyResponse:
        def __init__(self):
            self._served = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size=-1):
            if self._served:
                return b""
            self._served = True
            return b"payload"

    def fake_urlopen(_url):
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(_url, 500, "Internal Server Error", hdrs=None, fp=None)
        return DummyResponse()

    monkeypatch.setattr("ode.data.pull.urlopen", fake_urlopen)
    monkeypatch.setattr("ode.data.pull.time.sleep", lambda _seconds: None)

    result = _download_to_path(
        "https://example.org/retry.nc",
        destination,
        overwrite=True,
        max_retries=1,
        retry_delay_seconds=0,
    )

    assert result.status == "downloaded"
    assert destination.read_bytes() == b"payload"
    assert calls["count"] == 2


def test_download_to_path_raises_with_http_body_after_retries(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "failure.nc"

    class FailingBody:
        def read(self):
            return b"server exploded"

        def close(self):
            return None

    def fake_urlopen(_url):
        raise HTTPError(_url, 500, "Internal Server Error", hdrs=None, fp=FailingBody())

    monkeypatch.setattr("ode.data.pull.urlopen", fake_urlopen)
    monkeypatch.setattr("ode.data.pull.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="server exploded"):
        _download_to_path(
            "https://example.org/failure.nc",
            destination,
            overwrite=True,
            max_retries=1,
            retry_delay_seconds=0,
        )


def test_build_thredds_download_spec(monkeypatch) -> None:
    catalog_xml = b"""
    <catalog xmlns=\"http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0\">
      <service name=\"all\" serviceType=\"Compound\" base=\"\">
        <service name=\"ncss\" serviceType=\"NetcdfSubset\" base=\"//ncss.hycom.org/thredds/ncss/grid/\" />
      </service>
      <dataset name=\"HYCOM\" ID=\"GOMl0.04-expt_32.5\" urlPath=\"GOMl0.04/expt_32.5\" />
    </catalog>
    """
    dataset_xml = b"""
    <gridDataset>
      <TimeSpan>
        <begin>2014-04-02T00:00:00Z</begin>
        <end>2019-03-03T00:00:00Z</end>
      </TimeSpan>
            <grid name="ssh" shape="MT Latitude Longitude" />
            <grid name="u" shape="MT Depth Latitude Longitude" />
            <grid name="v" shape="MT Depth Latitude Longitude" />
    </gridDataset>
    """

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(url):
        if str(url).endswith("dataset.xml"):
            return DummyResponse(dataset_xml)
        return DummyResponse(catalog_xml)

    monkeypatch.setattr("ode.data.thredds.urlopen", fake_urlopen)

    spec = build_thredds_download_spec(
        ThreddsSubsetRequest(
            catalog_url="https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5",
            output_dir="data/raw",
            variables=("u", "v"),
            day="2019-03-03",
            north=31.9606,
            south=18.0916,
            west=-98.0,
            east=-76.4,
        )
    )

    parsed = urlparse(spec.url)
    params = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "ncss.hycom.org"
    assert parsed.path == "/thredds/ncss/grid/GOMl0.04/expt_32.5"
    assert params["var"] == ["u", "v"]
    assert params["time_start"] == ["2019-03-03T00:00:00Z"]
    assert params["time_end"] == ["2019-03-04T00:00:00Z"]
    assert params["vertCoord"] == ["0.0"]
    assert spec.path.endswith(".nc")


def test_build_thredds_download_specs_range_clamps_to_available_dates(monkeypatch) -> None:
    catalog_xml = b"""
    <catalog xmlns=\"http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0\">
      <service name=\"all\" serviceType=\"Compound\" base=\"\">
        <service name=\"ncss\" serviceType=\"NetcdfSubset\" base=\"//ncss.hycom.org/thredds/ncss/grid/\" />
      </service>
      <dataset name=\"HYCOM\" ID=\"GOMl0.04-expt_32.5\" urlPath=\"GOMl0.04/expt_32.5\" />
    </catalog>
    """
    dataset_xml = b"""
    <gridDataset>
      <TimeSpan>
        <begin>2014-04-02T00:00:00Z</begin>
        <end>2019-03-03T00:00:00Z</end>
      </TimeSpan>
            <grid name="ssh" shape="MT Latitude Longitude" />
            <grid name="u" shape="MT Depth Latitude Longitude" />
            <grid name="v" shape="MT Depth Latitude Longitude" />
    </gridDataset>
    """

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(url):
        if str(url).endswith("dataset.xml"):
            return DummyResponse(dataset_xml)
        return DummyResponse(catalog_xml)

    monkeypatch.setattr("ode.data.thredds.urlopen", fake_urlopen)

    specs = build_thredds_download_specs(
        ThreddsSubsetRequest(
            catalog_url="https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5",
            output_dir="data/raw",
            variables=("ssh",),
            start_date="2019-03-01",
            end_date="2019-03-05",
        )
    )

    assert len(specs) == 3
    parsed_urls = [urlparse(spec.url) for spec in specs]
    queries = [parse_qs(parsed.query) for parsed in parsed_urls]
    assert [query["time_start"][0] for query in queries] == [
        "2019-03-01T00:00:00Z",
        "2019-03-02T00:00:00Z",
        "2019-03-03T00:00:00Z",
    ]
    assert specs[-1].path.endswith("2019-03-03_ssh.nc")


def test_resolve_thredds_request_window_clamps_to_available_dates(monkeypatch) -> None:
    catalog_xml = b"""
    <catalog xmlns=\"http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0\">
      <service name=\"all\" serviceType=\"Compound\" base=\"\">
        <service name=\"ncss\" serviceType=\"NetcdfSubset\" base=\"//ncss.hycom.org/thredds/ncss/grid/\" />
      </service>
      <dataset name=\"HYCOM\" ID=\"GOMl0.04-expt_32.5\" urlPath=\"GOMl0.04/expt_32.5\" />
    </catalog>
    """
    dataset_xml = b"""
    <gridDataset>
      <TimeSpan>
        <begin>2014-04-02T00:00:00Z</begin>
        <end>2019-03-03T00:00:00Z</end>
      </TimeSpan>
      <grid name="ssh" shape="MT Latitude Longitude" />
      <grid name="u" shape="MT Depth Latitude Longitude" />
      <grid name="v" shape="MT Depth Latitude Longitude" />
    </gridDataset>
    """

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(url):
        if str(url).endswith("dataset.xml"):
            return DummyResponse(dataset_xml)
        return DummyResponse(catalog_xml)

    monkeypatch.setattr("ode.data.thredds.urlopen", fake_urlopen)

    window = resolve_thredds_request_window(
        ThreddsSubsetRequest(
            catalog_url="https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5",
            output_dir="data/raw",
            start_date="2019-03-01",
            end_date="2019-06-01",
        )
    )

    assert window.effective_start and window.effective_start.isoformat() == "2019-03-01"
    assert window.effective_end and window.effective_end.isoformat() == "2019-03-03"


def test_build_thredds_download_spec_defaults_to_ssh_u_v(monkeypatch) -> None:
    catalog_xml = b"""
    <catalog xmlns=\"http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0\">
      <service name=\"all\" serviceType=\"Compound\" base=\"\">
        <service name=\"ncss\" serviceType=\"NetcdfSubset\" base=\"//ncss.hycom.org/thredds/ncss/grid/\" />
      </service>
      <dataset name=\"HYCOM\" ID=\"GOMl0.04-expt_32.5\" urlPath=\"GOMl0.04/expt_32.5\" />
    </catalog>
    """
    dataset_xml = b"""
    <gridDataset>
      <TimeSpan>
        <begin>2014-04-02T00:00:00Z</begin>
        <end>2019-03-03T00:00:00Z</end>
      </TimeSpan>
            <grid name="ssh" shape="MT Latitude Longitude" />
            <grid name="u" shape="MT Depth Latitude Longitude" />
            <grid name="v" shape="MT Depth Latitude Longitude" />
    </gridDataset>
    """

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(url):
        if str(url).endswith("dataset.xml"):
            return DummyResponse(dataset_xml)
        return DummyResponse(catalog_xml)

    monkeypatch.setattr("ode.data.thredds.urlopen", fake_urlopen)

    specs = build_thredds_download_specs(
        ThreddsSubsetRequest(
            catalog_url="https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5",
            output_dir="data/raw",
            variables=(),
            day="2019-03-03",
        )
    )

    assert len(specs) == 2
    first_params = parse_qs(urlparse(specs[0].url).query)
    second_params = parse_qs(urlparse(specs[1].url).query)
    assert first_params["var"] == ["ssh"]
    assert "vertCoord" not in first_params
    assert second_params["var"] == ["u", "v"]
    assert second_params["vertCoord"] == ["0.0"]
    assert specs[0].path.endswith("2019-03-03_ssh.nc")
    assert specs[1].path.endswith("2019-03-03_u-v.nc")


def test_build_thredds_download_specs_range_defaults_split_ssh_and_surface_currents(monkeypatch) -> None:
    catalog_xml = b"""
    <catalog xmlns=\"http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0\">
      <service name=\"all\" serviceType=\"Compound\" base=\"\">
        <service name=\"ncss\" serviceType=\"NetcdfSubset\" base=\"//ncss.hycom.org/thredds/ncss/grid/\" />
      </service>
      <dataset name=\"HYCOM\" ID=\"GOMl0.04-expt_32.5\" urlPath=\"GOMl0.04/expt_32.5\" />
    </catalog>
    """
    dataset_xml = b"""
    <gridDataset>
      <TimeSpan>
        <begin>2014-04-02T00:00:00Z</begin>
        <end>2019-03-03T00:00:00Z</end>
      </TimeSpan>
      <grid name=\"ssh\" shape=\"MT Latitude Longitude\" />
      <grid name=\"u\" shape=\"MT Depth Latitude Longitude\" />
      <grid name=\"v\" shape=\"MT Depth Latitude Longitude\" />
    </gridDataset>
    """

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(url):
        if str(url).endswith("dataset.xml"):
            return DummyResponse(dataset_xml)
        return DummyResponse(catalog_xml)

    monkeypatch.setattr("ode.data.thredds.urlopen", fake_urlopen)

    specs = build_thredds_download_specs(
        ThreddsSubsetRequest(
            catalog_url="https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5",
            output_dir="data/raw",
            start_date="2019-03-02",
            end_date="2019-03-03",
        )
    )

    assert len(specs) == 4
    grouped = [parse_qs(urlparse(spec.url).query)["var"] for spec in specs]
    assert grouped == [["ssh"], ["u", "v"], ["ssh"], ["u", "v"]]


def test_build_thredds_download_spec_rejects_reversed_longitudes(monkeypatch) -> None:
    catalog_xml = b"""
    <catalog xmlns=\"http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0\">
      <service name=\"all\" serviceType=\"Compound\" base=\"\">
        <service name=\"ncss\" serviceType=\"NetcdfSubset\" base=\"//ncss.hycom.org/thredds/ncss/grid/\" />
      </service>
      <dataset name=\"HYCOM\" ID=\"GOMl0.04-expt_32.5\" urlPath=\"GOMl0.04/expt_32.5\" />
    </catalog>
    """
    dataset_xml = b"""
    <gridDataset>
      <TimeSpan>
        <begin>2014-04-02T00:00:00Z</begin>
        <end>2019-03-03T00:00:00Z</end>
      </TimeSpan>
    </gridDataset>
    """

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(url):
        if str(url).endswith("dataset.xml"):
            return DummyResponse(dataset_xml)
        return DummyResponse(catalog_xml)

    monkeypatch.setattr("ode.data.thredds.urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="east must be greater than or equal to west"):
        build_thredds_download_spec(
            ThreddsSubsetRequest(
                catalog_url="https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5",
                output_dir="data/raw",
                day="2019-03-03",
                north=21,
                south=20,
                west=-80,
                east=-82,
            )
        )


def test_cli_pull_thredds_command(tmp_path, monkeypatch, capsys) -> None:
    requested = {}

    def fake_pull_thredds_catalog(request):
        requested["request"] = request
        output_path = tmp_path / "subset.nc"
        output_path.write_bytes(b"subset")
        return [output_path]

    monkeypatch.setattr("ode.cli.pull_thredds_catalog", fake_pull_thredds_catalog)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ode",
            "pull-thredds",
            "--catalog-url",
            "https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5",
            "--day",
            "2019-03-03",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()
    stdout = capsys.readouterr().out

    assert requested["request"].catalog_url.startswith("https://tds.hycom.org/thredds/catalogs/")
    assert requested["request"].variables == DEFAULT_THREDDS_VARIABLES
    assert "Pulled" in stdout


def test_cli_pull_thredds_range_command(tmp_path, monkeypatch, capsys) -> None:
    requested = {}

    def fake_pull_thredds_catalog(request):
        requested["request"] = request
        output_a = tmp_path / "subset_a.nc"
        output_b = tmp_path / "subset_b.nc"
        output_a.write_bytes(b"a")
        output_b.write_bytes(b"b")
        return [output_a, output_b]

    monkeypatch.setattr("ode.cli.pull_thredds_catalog", fake_pull_thredds_catalog)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ode",
            "pull-thredds",
            "--catalog-url",
            "https://tds.hycom.org/thredds/catalogs/GOMl0.04/expt_32.5.html?dataset=GOMl0.04-expt_32.5",
            "--variable",
            "ssh",
            "--start-date",
            "2019-03-01",
            "--end-date",
            "2019-03-05",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()
    stdout = capsys.readouterr().out

    assert requested["request"].start_date == "2019-03-01"
    assert requested["request"].end_date == "2019-03-05"
    assert stdout.count("Pulled") == 2