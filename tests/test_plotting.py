"""Figures use the shared template and export to PNG only."""

import os
import pathlib

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
