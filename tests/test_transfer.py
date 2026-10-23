"""Radius transfer: a radius that is a function of direction transfers, a shift not."""

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm.eval import features
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards
from lejepa_norm.eval import transfer

N, K, D = 600, 4, 8
CFG = config.EvalConfig(num_bootstrap=200, max_kernel_images=N)


def _log_radius(u: numpy.ndarray) -> numpy.ndarray:
    """A smooth function of the direction `(N, D)`."""
    return 2.0 * u[:, 0] + numpy.sin(3.0 * u[:, 1]) + 3.0 * u[:, 2] * u[:, 3]


def _shard(
    base: numpy.ndarray,
    log_radius_offset: float,
    corruption: str,
    severity: int,
    rng: numpy.random.Generator,
) -> shards.EmbeddingShard:
    """Views jittered around `base`; radius = f(base) + offset + noise."""
    n = base.shape[0]
    unit = features.direction(base[:, None, :] + 0.05 * rng.normal(size=(n, K + 1, D)))
    rho = _log_radius(base) + log_radius_offset
    rho = rho[:, None] + 0.02 * rng.normal(size=(n, K + 1))
    z = (unit * numpy.exp(rho)[..., None]).astype(numpy.float32)
    return shards.EmbeddingShard(
        z=z,
        h=z.copy(),
        image_id=numpy.arange(n, dtype=numpy.int64),
        label=numpy.zeros(n, dtype=numpy.int64),
        corruption=numpy.full(n, corruption),
        severity=numpy.full(n, severity, dtype=numpy.int8),
    )


def _moved(base: numpy.ndarray, rng: numpy.random.Generator) -> numpy.ndarray:
    """Directions pushed systematically toward -e_0 (radius under f changes)."""
    push = numpy.zeros(D)
    push[0] = -0.8
    return features.direction(base + push + 0.1 * rng.normal(size=base.shape))


def _run(radius_offset: float) -> transfer.TransferResult:
    rng = numpy.random.default_rng(0)
    base = features.direction(rng.normal(size=(N, D)))
    test = _shard(base, 0.0, "", 0, rng)
    corrupt = _shard(_moved(base, rng), radius_offset, "noise", 3, rng)
    return transfer.run_transfer(test, corrupt, CFG, seed=1)


def test_radius_that_follows_the_direction_transfers() -> None:
    result = _run(radius_offset=0.0)
    (cell,) = result.cells
    # The direction moved, so the radius moved (raw shift is large) ...
    assert abs(cell.raw_shift.value) > 0.3
    assert cell.raw_shift.high < -0.3 or cell.raw_shift.low > 0.3
    # ... and the clean map explains all of it: residual ~ 0, CI covers 0.
    assert abs(cell.residual_held.value) < 0.05
    assert abs(cell.residual_held.value) < 0.05 * abs(cell.raw_shift.value)
    assert cell.predicted_shift == pytest.approx(cell.raw_shift.value, abs=0.05)
    assert result.clean_r2 > 0.9


def test_additive_radius_shift_does_not_transfer() -> None:
    result = _run(radius_offset=-0.4)
    (cell,) = result.cells
    # The extra shift is invisible in the direction: it stays in the residual.
    assert cell.residual_held.value == pytest.approx(-0.4, abs=0.05)
    assert cell.residual_held.high < -0.3
    assert cell.raw_shift.value - cell.predicted_shift == pytest.approx(-0.4, abs=0.05)


def test_held_out_twins_are_disjoint_from_the_fit_and_cover_every_image() -> None:
    result = _run(radius_offset=0.0)
    (cell,) = result.cells
    assert result.n_fit == N // 2 and result.n_held == N - N // 2
    assert cell.n_fit == result.n_fit and cell.n_held == result.n_held
    assert abs(result.clean_residual.value) < 0.05  # reference: no shift, no bias


def test_fitted_kernel_ridge_predicts_new_points_like_the_cv_value() -> None:
    rng = numpy.random.default_rng(3)
    u = features.direction(rng.normal(size=(500, D)))
    y = _log_radius(u)
    model = redundancy.FittedKernelRidge(u[:300], y[:300], rng)
    resid = y[300:] - model.predict(u[300:], chunk=64)  # chunked == unchunked
    assert 1.0 - resid.var() / y[300:].var() > 0.9
    assert model.predict(u[300:]) == pytest.approx(y[300:] - resid)


def test_aggregate_transfer_averages_seeds_and_reports_unexplained_fraction() -> None:
    def table(raw: float, res: float, r2: float) -> list[dict[str, str]]:
        return [
            {
                "corruption": "noise",
                "severity": "3",
                "raw shift": f"{raw} [{raw - 0.01}, {raw + 0.01}]",
                "residual (held-out twins)": f"{res} [{res - 0.01}, {res + 0.01}]",
                "R^2 (held-out twins)": str(r2),
            }
        ]

    rows = transfer.aggregate_transfer(
        {0.05: [table(-0.2, -0.02, 0.9), table(-0.1, 0.0, 0.8)]}
    )
    (row,) = rows
    assert row["seeds"] == 2
    assert row["raw shift"] == "-0.150 ± 0.071"
    assert row["unexplained fraction"] == pytest.approx(-0.01 / -0.15)
    with pytest.raises(ValueError):
        transfer.aggregate_transfer({0.05: [table(-0.2, 0.0, 0.9), []]})
