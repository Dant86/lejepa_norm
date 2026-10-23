"""Redundancy controls: planted dependence is found, independence is not."""

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm.eval import controls
from lejepa_norm.eval import features

N, K, D = 300, 4, 8
CFG = config.EvalConfig(cv_folds=5, num_permutations=20, max_kernel_images=N)


def _views(log_radius: numpy.ndarray, seed: int) -> numpy.ndarray:
    """`(N, K, D)` embeddings: per-image direction with view jitter, given radius."""
    rng = numpy.random.default_rng(seed)
    base = features.direction(rng.standard_normal((N, 1, D)))
    unit = features.direction(base + 0.1 * rng.standard_normal((N, K, D)))
    jitter = 0.05 * rng.standard_normal((N, K))
    return unit * numpy.exp(log_radius[:, None] + jitter)[..., None]


def _log_radius_from_direction(seed: int) -> numpy.ndarray:
    base = features.direction(numpy.random.default_rng(seed).standard_normal((N, 1, D)))
    return numpy.sin(3 * base[:, 0, 0]) + base[:, 0, 1] * base[:, 0, 2] * 4


def test_setups_cover_same_view_cross_view_and_single_view() -> None:
    x = _views(numpy.zeros(N), 0)
    names = [s.name for s in controls.view_setups(x)]
    assert len(names) == 5
    assert names[0].startswith("same views") and names[-1].startswith(
        "single view, other"
    )
    # With two views there are no half-to-half setups.
    assert len(controls.view_setups(x[:, :2])) == 3


def test_split_half_reliability_is_high_for_image_level_radius_and_low_for_noise() -> (
    None
):
    rho = numpy.random.default_rng(1).normal(size=N)
    assert controls.split_half_reliability(_views(rho, 2)) > 0.9
    noise_only = numpy.exp(numpy.random.default_rng(3).normal(size=(N, K, 1)))
    unit = features.direction(numpy.random.default_rng(4).standard_normal((N, K, D)))
    assert controls.split_half_reliability(unit * noise_only) < 0.1


def test_cross_view_controls_recover_planted_dependence() -> None:
    x = _views(_log_radius_from_direction(5), 5)
    rng = numpy.random.default_rng(6)
    rows = [controls.run_setup(s, CFG, rng) for s in controls.view_setups(x)]
    for row in rows:
        assert row["permutation p"] == pytest.approx(1 / (CFG.num_permutations + 1))
        assert abs(float(row["null mean"])) < 0.05
    halves = [r for r in rows if "disjoint" in str(r["setup"])]
    assert len(halves) == 2 and all(
        float(r["r2 (nested CV, kernel)"]) > 0.6 for r in halves
    )


def test_controls_reject_nothing_when_radius_is_independent_of_direction() -> None:
    rho = numpy.random.default_rng(7).normal(size=N)
    x = _views(rho, 8)
    rng = numpy.random.default_rng(9)
    for setup in controls.view_setups(x):
        row = controls.run_setup(setup, CFG, rng)
        assert float(row["r2 (nested CV, kernel)"]) < 0.1, setup.name
        assert float(row["permutation p"]) > 0.05, setup.name


def _table(r2: float, p: float = 0.002) -> list[dict[str, str]]:
    same = {"space": "z", "setup": "same views (all 8): rho_bar ~ u_hat"}
    reliability = {"space": "z", "setup": controls.RELIABILITY_SETUP}
    return [
        {**same, controls.R2: str(r2), "null max": "0.003", "permutation p": str(p)}
        | {"F": "50"},
        {**reliability, controls.R2: "0.9", "null max": "", "permutation p": ""}
        | {"F": ""},
    ]


def test_aggregate_controls_reports_mean_std_and_least_favorable_values() -> None:
    rows = controls.aggregate_controls(
        {0.05: [_table(0.90), _table(0.94, p=0.01)], 0.1: [_table(0.8)]}
    )
    assert [(r["lambda"], r["seeds"]) for r in rows[::2]] == [(0.05, 2), (0.1, 1)]
    first = rows[0]
    assert first[controls.R2] == "0.92 ± 0.028"
    assert first["worst permutation p"] == 0.01 and first["min F"] == 50.0
    assert "min F" not in rows[1]  # reliability row has only R^2
    assert rows[2][controls.R2] == "0.8"  # one seed: no std
    with pytest.raises(ValueError, match="different rows"):
        controls.aggregate_controls({0.05: [_table(0.9), _table(0.9)[:1]]})
