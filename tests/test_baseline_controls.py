"""E2 controls on planted data: dependence, smooth-function calibration, curve."""

import collections.abc
import pathlib

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm.eval import baseline_controls
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import shards

D, K = 6, 8
CFG = config.EvalConfig(
    cv_folds=3,
    num_permutations=8,
    max_kernel_images=240,
    num_bootstrap=60,
    smooth_draws=8,
    curve_holdout=150,
    curve_margin=0.01,
)


def _shard(n: int, seed: int, dependent: bool) -> shards.EmbeddingShard:
    """Views with a per-image direction; log-radius a function of it (or noise)."""
    rng = numpy.random.default_rng(seed)
    base = features.direction(rng.standard_normal((n, 1, D)))
    unit = features.direction(base + 0.1 * rng.standard_normal((n, K + 1, D)))
    signal = numpy.sin(2 * base[:, 0, 0]) + base[:, 0, 1] * base[:, 0, 2]
    if not dependent:
        signal = rng.standard_normal(n)
    rho = signal[:, None] + 0.05 * rng.standard_normal((n, K + 1))
    z = (unit * numpy.exp(rho)[..., None]).astype(numpy.float32)
    return shards.EmbeddingShard(
        z=z,
        h=z.copy(),
        image_id=numpy.arange(n, dtype=numpy.int64),
        label=numpy.arange(n, dtype=numpy.int64) % 2,
        corruption=numpy.array([""] * n, dtype=numpy.str_),
        severity=numpy.zeros(n, dtype=numpy.int8),
    )


def _run(name: str, seed: int, dependent: bool) -> baseline_controls.RunResult:
    return baseline_controls.run_unit(
        name,
        _shard(500, 100 + seed, dependent),
        _shard(500, 200 + seed, dependent),
        CFG,
        seed,
    )


@pytest.fixture(scope="module")
def results() -> collections.abc.Iterator[list[baseline_controls.RunResult]]:
    # A cheap curve: the reference size (where the bandwidth is chosen, and above
    # which it is reused) is 250, with two larger sizes and one draw each.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(baseline_controls, "REFERENCE_N", 250)
        patch.setattr(baseline_controls, "CURVE_SIZES", (250, 500))
        patch.setattr(baseline_controls, "CURVE_REPEATS", {250: 2, 500: 1})
        yield [
            _run("trained_s0", 0, True),
            _run("trained_s1", 1, True),
            _run("rand_s0", 0, False),
            _run("rand_s1", 1, False),
        ]


def test_split_run_name() -> None:
    assert baseline_controls.split_run_name("radius_marginal_lam0.05_s2") == (
        "radius_marginal_lam0.05",
        2,
    )
    with pytest.raises(ValueError):
        baseline_controls.split_run_name("no_seed")


def test_controls_separate_planted_dependence_from_independence(
    results: list[baseline_controls.RunResult],
) -> None:
    rows = baseline_controls.controls_rows(results, 40, numpy.random.default_rng(0))
    same = {r["group"]: r for r in rows if str(r["setup"]).startswith("same views")}
    assert float(same["trained"]["r2 (mean over seeds)"]) > 0.8  # pyright: ignore[reportArgumentType]
    assert float(same["rand"]["r2 (mean over seeds)"]) < 0.2  # pyright: ignore[reportArgumentType]
    contrast = baseline_controls.contrast_rows(
        results, "rand", 40, numpy.random.default_rng(0)
    )
    assert {r["group"] for r in contrast} == {"trained"}
    top = next(r for r in contrast if str(r["setup"]).startswith("same views"))
    assert float(top["difference in r2"]) > 0.5  # pyright: ignore[reportArgumentType]


def test_smooth_functions_calibrate_what_r2_means(
    results: list[baseline_controls.RunResult],
) -> None:
    info = results[0].info["smooth"]
    assert info["tau_hat"] > 0 and 0 <= info["noise_fraction"] < 0.2
    r2 = {f: results[0].arrays[f"smooth|{f}"] for f in info["families"]}
    assert all(v.shape == (CFG.smooth_draws,) for v in r2.values())
    # Smooth noiseless functions are recovered; noise caps R^2; rough ones are hard.
    assert r2["GP tau=1x, noiseless"].mean() > 0.9
    assert r2["GP tau=1x, matched noise"].mean() < r2["GP tau=1x, noiseless"].mean()
    assert r2["random linear, noiseless"].mean() > 0.9
    assert r2["GP tau=0.25x, noiseless"].mean() < r2["GP tau=4x, noiseless"].mean()
    rows = baseline_controls.smooth_rows(results, 40, numpy.random.default_rng(0))
    assert {r["group"] for r in rows} == {"trained", "rand"}
    assert len(rows) == 2 * len(info["families"])


def test_learning_curve_rises_and_saturation_has_a_verdict(
    results: list[baseline_controls.RunResult],
) -> None:
    sizes = results[0].info["curve"]["sizes"]
    # Pool of 1000 images minus the 150 held out: the last size is the whole pool.
    assert sizes == [250, 500, 850]
    rows = baseline_controls.curve_rows(results, 40, numpy.random.default_rng(0))
    trained = [
        r for r in rows if r["group"] == "trained" and r["setup"] == "same views"
    ]
    values = [float(r["r2 (mean over seeds)"]) for r in trained]  # pyright: ignore[reportArgumentType]
    assert values[-1] > values[0]
    assert values[-1] > 0.8
    sat = baseline_controls.saturation_rows(
        results, CFG.curve_margin, 40, numpy.random.default_rng(0)
    )
    assert sat
    assert {r["verdict"] for r in sat} <= {"equivalent", "different", "inconclusive"}


def test_shard_roundtrip_and_report(
    results: list[baseline_controls.RunResult], tmp_path: pathlib.Path
) -> None:
    baseline_controls.save_run(tmp_path / "a.npz", results[0])
    back = baseline_controls.load_run(tmp_path / "a.npz")
    assert back.name == results[0].name and back.seed == 0 and back.group == "trained"
    assert back.info["smooth"]["families"] == results[0].info["smooth"]["families"]
    for key, value in results[0].arrays.items():
        numpy.testing.assert_allclose(back.arrays[key], value)

    notes = baseline_controls.write_notes(
        results, CFG, tmp_path / "controls.md", "rand", 0, None, figures=False
    )
    assert [n.name for n in notes] == [
        "controls_redundancy.md",
        "controls_smooth.md",
        "controls_curve.md",
    ]
    text = (tmp_path / "controls_curve.md").read_text()
    assert "baseline.saturation" in text and "verdict" in text
    assert "baseline.contrast" in (tmp_path / "controls_redundancy.md").read_text()


def test_list_runs_requires_both_dumps_and_a_seed(tmp_path: pathlib.Path) -> None:
    for name, sets in {
        "radius_marginal_lam0.1_s1": ("stl10_train", "stl10_test"),
        "radius_marginal_lam0.1_s2": ("stl10_test",),
        "radius_marginal_dynamics": ("stl10_train", "stl10_test"),
        "other_s0": ("stl10_train", "stl10_test"),
    }.items():
        for s in sets:
            d = dumps.set_dir(tmp_path, name, s)
            d.mkdir(parents=True)
            (d / "part-00000.npz").write_bytes(b"")
    assert baseline_controls.list_runs(tmp_path, ["radius_marginal"]) == [
        "radius_marginal_lam0.1_s1"
    ]
