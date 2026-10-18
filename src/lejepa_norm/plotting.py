"""Plotly figures exported to PNG via Kaleido (single shared template).

Every figure is built on the `lejepa_norm` template (fonts, sizes, colors), so
all figures in all notes look the same. Figures are only ever written as PNG
(`export_png`); no HTML is produced.

Kaleido >= 1.0 drives a headless Chrome. Pass the Chrome binary explicitly
(`paths.Paths.kaleido_chrome`, from `LEJEPA_NORM_KALEIDO_CHROME`); if None,
Kaleido searches the usual locations. MathJax is disabled so that export never
touches the network.
"""

# Plotly and Kaleido ship no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import collections.abc
import math
import os
import pathlib
import typing

import kaleido
import numpy
import numpy.typing
import plotly.graph_objects as go
import plotly.io

TEMPLATE_NAME = "lejepa_norm"
FONT_FAMILY = "Helvetica, Arial, sans-serif"
FONT_SIZE = 14
WIDTH = 800
HEIGHT = 500
SCALE = 2
# Categorical series colors, assigned in this fixed order (validated
# colorblind-safe on adjacent pairs); a 9th+ series reuses colors with a dash.
COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
DASHES = ("solid", "dash", "dot", "dashdot")
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE_COLOR = "#ffffff"
REFERENCE_COLOR = "#0b0b0b"

plotly.io.templates[TEMPLATE_NAME] = go.layout.Template(
    layout=go.Layout(
        font={"family": FONT_FAMILY, "size": FONT_SIZE, "color": TEXT_COLOR},
        title={
            "font": {"size": FONT_SIZE + 4},
            "x": 0.0,
            "xref": "paper",
            "xanchor": "left",
        },
        colorway=list(COLORS),
        paper_bgcolor=SURFACE_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        width=WIDTH,
        height=HEIGHT,
        margin={"l": 70, "r": 30, "t": 60, "b": 60},
        xaxis={
            "showgrid": False,
            "linecolor": MUTED_COLOR,
            "ticks": "outside",
            "tickcolor": MUTED_COLOR,
            "zeroline": False,
        },
        yaxis={
            "gridcolor": GRID_COLOR,
            "linecolor": MUTED_COLOR,
            "zeroline": False,
        },
        legend={"bgcolor": "rgba(255,255,255,0.8)", "font": {"size": FONT_SIZE - 2}},
    )
)


def _figure(title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=TEMPLATE_NAME,
        title={"text": title},
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def _style(index: int) -> tuple[str, str]:
    return COLORS[index % len(COLORS)], DASHES[(index // len(COLORS)) % len(DASHES)]


def chi_pdf(
    r: numpy.typing.ArrayLike, dim: int, scale: float = 1.0
) -> numpy.typing.NDArray[numpy.float64]:
    """Density of `scale * chi_dim` at `r` (computed in log space)."""
    x = numpy.asarray(r, dtype=numpy.float64) / scale
    log_pdf = (
        (dim - 1) * numpy.log(numpy.maximum(x, 1e-300))
        - x**2 / 2
        - (dim / 2 - 1) * math.log(2)
        - math.lgamma(dim / 2)
    )
    return numpy.where(x > 0, numpy.exp(log_pdf), 0.0) / scale


def radius_histogram(
    radius: numpy.typing.ArrayLike,
    dim: int,
    title: str,
    fit_scale: bool = False,
) -> go.Figure:
    """Histogram of radii against the `chi_d` density.

    Args:
        radius: Radii `||z||`.
        dim: Dimension `d` of the reference `chi_d`.
        title: Figure title.
        fit_scale: Overlay `sigma * chi_d` with `sigma^2 = E[r^2] / d` instead of
            the unit `chi_d` (useful for `h`, which is not normalized).

    Returns:
        The figure.
    """
    r = numpy.asarray(radius, dtype=numpy.float64).ravel()
    scale = float(numpy.sqrt(numpy.mean(r**2) / dim)) if fit_scale else 1.0
    fig = _figure(title, "radius r", "density")
    fig.add_trace(
        go.Histogram(
            x=r,
            histnorm="probability density",
            name="empirical",
            marker={"color": COLORS[0], "line": {"color": SURFACE_COLOR, "width": 1}},
            opacity=0.85,
        )
    )
    ref_lo = scale * max(math.sqrt(dim) - 6.0, 0.0)
    ref_hi = scale * (math.sqrt(dim) + 6.0)
    grid = numpy.linspace(min(float(r.min()), ref_lo), max(float(r.max()), ref_hi), 400)
    label = f"chi_{dim}" if not fit_scale else f"{scale:.3g} x chi_{dim}"
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=chi_pdf(grid, dim, scale),
            mode="lines",
            name=label,
            line={"color": REFERENCE_COLOR, "width": 2},
        )
    )
    fig.update_layout(bargap=0.0)
    return fig


class Series(typing.NamedTuple):
    """One line with optional error band: x, y and CI bounds (same length)."""

    x: collections.abc.Sequence[float]
    y: collections.abc.Sequence[float]
    low: collections.abc.Sequence[float] | None = None
    high: collections.abc.Sequence[float] | None = None


def line_plot(
    series: collections.abc.Mapping[str, Series],
    title: str,
    x_title: str,
    y_title: str,
    reference: float | None = None,
) -> go.Figure:
    """Lines with markers and optional CI error bars (e.g. metric vs severity).

    Args:
        series: Name to series; colors follow the mapping order.
        title: Figure title.
        x_title: X axis title.
        y_title: Y axis title.
        reference: Optional horizontal reference line (e.g. kappa = 1).

    Returns:
        The figure.
    """
    fig = _figure(title, x_title, y_title)
    for i, (name, s) in enumerate(series.items()):
        color, dash = _style(i)
        error_y = None
        if s.low is not None and s.high is not None:
            y = numpy.asarray(s.y, dtype=numpy.float64)
            error_y = {
                "type": "data",
                "symmetric": False,
                "array": numpy.asarray(s.high, dtype=numpy.float64) - y,
                "arrayminus": y - numpy.asarray(s.low, dtype=numpy.float64),
                "thickness": 1,
                "width": 3,
            }
        fig.add_trace(
            go.Scatter(
                x=list(s.x),
                y=list(s.y),
                mode="lines+markers",
                name=name,
                line={"color": color, "dash": dash, "width": 2},
                marker={"size": 8},
                error_y=error_y,
            )
        )
    if reference is not None:
        fig.add_hline(y=reference, line={"color": MUTED_COLOR, "dash": "dot"})
    return fig


def scatter(
    x: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    title: str,
    x_title: str,
    y_title: str,
) -> go.Figure:
    """Single-series scatter plot with semi-transparent markers."""
    fig = _figure(title, x_title, y_title)
    fig.add_trace(
        go.Scatter(
            x=numpy.asarray(x, dtype=numpy.float64).ravel(),
            y=numpy.asarray(y, dtype=numpy.float64).ravel(),
            mode="markers",
            marker={"color": COLORS[0], "size": 8, "opacity": 0.5},
            showlegend=False,
        )
    )
    return fig


def _kaleido_options(chrome: str | os.PathLike[str] | None) -> dict[str, typing.Any]:
    opts: dict[str, typing.Any] = {"mathjax": False}
    if chrome is not None:
        opts["path"] = str(chrome)
    return opts


def to_png_bytes(fig: go.Figure, chrome: str | os.PathLike[str] | None = None) -> bytes:
    """Renders `fig` to PNG bytes with Kaleido."""
    data = kaleido.calc_fig_sync(
        fig,
        opts={"format": "png", "width": WIDTH, "height": HEIGHT, "scale": SCALE},
        kopts=_kaleido_options(chrome),
    )
    return bytes(data)


def export_png(
    fig: go.Figure,
    path: str | os.PathLike[str],
    chrome: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Writes `fig` as a PNG file (the only export format allowed).

    Args:
        fig: The figure.
        path: Output path; must end in `.png`. Parent directories are created.
        chrome: Chrome binary for Kaleido (`paths.Paths.kaleido_chrome`).

    Returns:
        The written path.

    Raises:
        ValueError: `path` does not end in `.png`.
    """
    out = pathlib.Path(path)
    if out.suffix.lower() != ".png":
        raise ValueError(f"figures are exported as PNG only, got {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(to_png_bytes(fig, chrome))
    return out


def chrome_available(chrome: str | os.PathLike[str] | None = None) -> bool:
    """Whether Kaleido can render with `chrome` (tries a tiny figure)."""
    if chrome is not None and not pathlib.Path(chrome).is_file():
        return False
    try:
        to_png_bytes(go.Figure(), chrome)
    except Exception:  # noqa: BLE001 - any failure means "cannot render"
        return False
    return True
