"""Figures use the shared template and export to PNG only."""

import os
import pathlib
import typing

import numpy
import pytest

from lejepa_norm import paths
from lejepa_norm import plotting

# Read at import: other tests may load `.env` files that leak into os.environ.
_CHROME_ENV = os.environ.get(paths.KALEIDO_CHROME_VAR)


def _chrome() -> pathlib.Path | None:
    return pathlib.Path(_CHROME_ENV) if _CHROME_ENV else None


def test_chi_density_integrates_to_one_and_peaks_near_sqrt_d() -> None:
    grid = numpy.linspace(0.0, 40.0, 20001)
    pdf = plotting.chi_pdf(grid, 256)
    assert float(numpy.sum(pdf) * (grid[1] - grid[0])) == pytest.approx(1.0, 1e-4)
    assert grid[numpy.argmax(pdf)] == pytest.approx(numpy.sqrt(255), abs=0.01)


def test_export_rejects_non_png(tmp_path: pathlib.Path) -> None:
    fig = plotting.scatter([1, 2], [3, 4], "t", "x", "y")
    with pytest.raises(ValueError, match="PNG"):
        plotting.export_png(fig, tmp_path / "fig.html")


def test_dual_axis_plot_puts_right_series_on_second_axis() -> None:
    fig = plotting.dual_axis_line_plot(
        {"r2": plotting.Series([1, 2], [0.1, 0.9])},
        {"kappa": plotting.Series([1, 2], [5.0, 20.0], [4.0, 18.0], [6.0, 22.0])},
        "t",
        "epoch",
        "R^2",
        "kappa",
        right_reference=1.0,
    )
    spec = typing.cast("dict[str, typing.Any]", fig.to_dict())
    traces = spec["data"]
    assert [t["yaxis"] for t in traces] == ["y", "y2"]
    assert spec["layout"]["yaxis2"]["overlaying"] == "y"
    assert traces[0]["line"]["color"] != traces[1]["line"]["color"]


@pytest.mark.slow  # headless Chrome
def test_figures_render_to_png(tmp_path: pathlib.Path) -> None:
    chrome = _chrome()
    if not plotting.chrome_available(chrome):
        pytest.skip("Chrome for Kaleido is not available")
    r = numpy.linalg.norm(numpy.random.default_rng(0).normal(size=(500, 256)), axis=1)
    out = plotting.export_png(
        plotting.radius_histogram(r, 256, "radius"), tmp_path / "hist.png", chrome
    )
    assert out.read_bytes().startswith(b"\x89PNG")


def _xaxis(fig: plotting.go.Figure) -> dict[str, typing.Any]:
    layout = typing.cast(dict[str, typing.Any], fig.to_dict())["layout"]
    return typing.cast(dict[str, typing.Any], layout["xaxis"])


def test_line_plot_can_use_a_logarithmic_x_axis() -> None:
    series = {"a": plotting.Series(x=[50, 500, 5000], y=[1.0, 2.0, 3.0])}
    log = _xaxis(plotting.line_plot(series, "t", "x", "y", log_x=True))
    assert log["type"] == "log" and log["tickvals"] == [50.0, 500.0, 5000.0]
    assert "type" not in _xaxis(plotting.line_plot(series, "t", "x", "y"))
