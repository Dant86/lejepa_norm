"""Frozen-probe radius dependence: exact decomposition, invariances, aggregation."""

import importlib.util
import pathlib
import types

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import downstream
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import frozen_linear as fl
from lejepa_norm.eval import randomization
from lejepa_norm.eval import shards

REPO = pathlib.Path(__file__).parent.parent
N, D, CLASSES = 240, 8, 3
BASE = randomization.BASELINE
PERMUTE = randomization.PERMUTE


def _embeddings(
    seed: int, label_in_radius: bool = False, signal: float = 3.0
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Class in the direction (or only in the radius); radius independent otherwise."""
    rng = numpy.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, N)
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    direction = (
        0.4 * rng.normal(size=(N, D))
        + (0.0 if label_in_radius else signal) * (centers[labels])
    )
    direction[:, 0] += 1.0
    log_r = 0.3 * rng.normal(size=N) + (1.5 * labels if label_in_radius else 0.0)
    return features.direction(direction) * numpy.exp(log_r)[:, None], labels


def _analyze(
    label_in_radius: bool = False, name: str = "toy", signal: float = 3.0
) -> fl.RunAnalysis:
    bank, by = _embeddings(10, label_in_radius, signal)
    query, qy = _embeddings(11, label_in_radius, signal)
    return fl.analyze_run(
        name, 0.05, 0, bank, by, query, qy, num_repeats=3, cv_folds=3, seed=0
    )


def _change(result: fl.RunAnalysis, probe: str, condition: str) -> float:
    return float(
        result.arrays[f"correct|{probe}|{condition}"].mean()
        - result.arrays[f"correct|{probe}|{BASE}"].mean()
    )


def test_affine_form_reproduces_the_probe_exactly() -> None:
    x, y = _embeddings(1)
    probe = downstream.LinearProbe(x, y, CLASSES)
    weights, bias = probe.affine()
    logits = x @ weights.T + bias
    log_p = numpy.log(probe.predict_proba(x))
    # Softmax logits equal log-probabilities up to a per-row constant.
    numpy.testing.assert_allclose(
        logits - logits.mean(axis=1, keepdims=True),
        log_p - log_p.mean(axis=1, keepdims=True),
        atol=1e-6,
    )
    numpy.testing.assert_array_equal(logits.argmax(axis=1), probe.predict(x))
    # Without centering and intercept the probe is homogeneous: scale invariant.
    homog = downstream.LinearProbe(x, y, CLASSES, center=False, fit_intercept=False)
    numpy.testing.assert_array_equal(homog.predict(x), homog.predict(7.0 * x))
    w, b = homog.affine()
    numpy.testing.assert_allclose(b, 0.0, atol=1e-12)
    assert w.shape == (CLASSES, D)


def test_radius_free_probes_are_exactly_invariant_and_bias_creates_the_rest() -> None:
    result = _analyze(signal=0.8)  # overlapping classes: small margins exist
    for probe in (fl.HOMOG, fl.NOBIAS, fl.UNIT, fl.MLP_UNIT):
        for condition in randomization.CONDITIONS[1:]:
            assert _change(result, probe, condition) == 0.0, (probe, condition)
    # A flip to the baseline runner-up requires the pairwise margin to change sign,
    # i.e. the crossing radius lies between the original and the permuted radius.
    to_runner = result.arrays["to_runner_up|linear"] > 0
    distance = numpy.broadcast_to(result.arrays["cross_distance"], to_runner.shape)
    ratio = numpy.abs(result.arrays["log_ratio"])
    assert to_runner.any()
    assert (distance[to_runner] <= ratio[to_runner] + 1e-9).all()
    flipped = result.arrays["flip|linear"] > 0
    assert flipped[to_runner].all()
    # Decomposition scalars exist and are positive.
    assert result.scalars["bias spread (sd over classes)"] > 0
    assert 0 < result.scalars["share of images with a crossing"] <= 1


def test_constant_radius_curve_is_flat_for_radius_free_probes() -> None:
    result = _analyze()
    for probe in (fl.HOMOG, fl.UNIT, fl.NOBIAS, fl.MLP_UNIT):
        curve = result.arrays[f"curve|{probe}"]
        assert numpy.ptp(curve) == 0.0
    # The intercept probe collapses when the radius goes to 0 (only the bias is left).
    raw = result.arrays[f"curve|{fl.RAW}"]
    assert raw[-1] > 0.9 and raw[0] < raw[-1]


def test_radius_only_labels_break_the_raw_probe_but_not_by_a_bias_artifact() -> None:
    result = _analyze(label_in_radius=True)
    assert result.arrays[f"correct|{fl.RAW}|{BASE}"].mean() > 0.85
    assert _change(result, fl.RAW, PERMUTE) < -0.3
    # Radius-free probes cannot read a label that lives only in the radius.
    assert result.arrays[f"correct|{fl.UNIT}|{BASE}"].mean() < 0.6


def test_radius_permuted_training_removes_the_frozen_dependence() -> None:
    result = _analyze()
    assert abs(_change(result, fl.AUG, PERMUTE)) < 0.03
    assert result.arrays[f"correct|{fl.AUG}|{BASE}"].mean() > 0.85


def test_run_analysis_roundtrips_through_a_shard(tmp_path: pathlib.Path) -> None:
    result = _analyze(name="toy_lam0.05_s0")
    fl.save_run(tmp_path / "x" / "toy.npz", result)
    loaded = fl.load_run(tmp_path / "x" / "toy.npz")
    assert (loaded.name, loaded.lam, loaded.seed) == (result.name, 0.05, 0)
    assert loaded.scalars == result.scalars
    assert loaded.arrays.keys() == result.arrays.keys()
    numpy.testing.assert_array_equal(
        loaded.arrays["flip|linear"], result.arrays["flip|linear"]
    )
    assert not list((tmp_path / "x").glob("*.partial"))


def test_spearman_is_a_rank_correlation_on_resamples() -> None:
    x = numpy.arange(50.0)
    idx = numpy.random.default_rng(0).integers(0, 50, size=(30, 50))
    numpy.testing.assert_allclose(fl.spearman(x, numpy.exp(x), idx), 1.0)
    numpy.testing.assert_allclose(fl.spearman(x, -(x**3), idx), -1.0)


def _two_lambdas() -> list[fl.RunAnalysis]:
    out: list[fl.RunAnalysis] = []
    for lam, seed in ((0.05, 0), (0.05, 1), (0.1, 0)):
        bank, by = _embeddings(20 + seed)
        query, qy = _embeddings(30 + seed)
        out.append(
            fl.analyze_run(
                f"toy_{lam}_{seed}",
                lam,
                seed,
                bank,
                by,
                query,
                qy,
                num_repeats=2,
                cv_folds=3,
                seed=0,
            )
        )
    return out


def test_aggregation_tables_cover_every_lambda_probe_and_reader() -> None:
    results = _two_lambdas()
    rng = numpy.random.default_rng(0)
    changes = fl.change_rows(results, 0.005, 50, rng)
    assert len(changes) == 2 * len(fl.PROBES) * 3
    by_key = {(r["lambda"], r["probe"], r["condition"]): r for r in changes}
    exact = by_key[0.05, fl.HOMOG, PERMUTE]
    assert exact["verdict"] == randomization.EQUIVALENT
    assert exact["95% CI (points)"] == "[+0.00, +0.00]"
    assert {r["verdict"] for r in changes} <= {
        randomization.EQUIVALENT,
        randomization.DIFFERENT,
        randomization.INCONCLUSIVE,
    }
    assert len(fl.cost_rows(results, 0.005, 50, rng)) == 2 * (len(fl.PROBES) - 2)
    assert len(fl.flip_rows(results, 50, rng)) == 2 * len(fl.READERS)
    correlations = fl.correlation_rows(results, 50, rng)
    assert len(correlations) == 2 * (5 + 4)
    assert len(fl.grid_rows(results)) == 2 * 2 * 5
    assert fl.decomposition_rows(results)
    points = fl.curve_points(results)
    assert len(points) == 2 * len(fl.PROBES) * len(fl.CURVE_EXPONENTS)
    assert fl.curve_rows(points, (fl.RAW,))


def _dump(labels: numpy.ndarray, seed: int) -> shards.EmbeddingShard:
    x, _ = _embeddings(seed)
    z = numpy.repeat(x[:, None, :], 2, axis=1).astype(numpy.float32)
    return shards.EmbeddingShard(
        z=z,
        h=z.copy(),
        image_id=numpy.arange(N, dtype=numpy.int64),
        label=labels.astype(numpy.int64),
        corruption=numpy.array([""] * N),
        severity=numpy.zeros(N, dtype=numpy.int8),
    )


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_frozen_linear_mode_runs_per_run_units_then_aggregates(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, embed, notes = tmp_path / "runs", tmp_path / "embed", tmp_path / "notes"
    names = {
        "toy_lam0.05_s0": (0.05, 0),
        "toy_lam0.05_s1": (0.05, 1),
        "toy_lam0.1_s0": (0.1, 0),
    }
    for name, (lam, seed) in names.items():
        cfg = config.ExperimentConfig(seed=seed)
        cfg.loss.lambda_ = lam
        config.save_resolved(cfg, runs / name)
        for eval_set, offset in ((dumps.STL10_TRAIN, 0), (dumps.STL10_TEST, 100)):
            if name == "toy_lam0.1_s0" and eval_set == dumps.STL10_TEST:
                continue  # a run without both dumps is not part of the study
            out = dumps.set_dir(embed, name, eval_set)
            out.mkdir(parents=True)
            labels = _embeddings(seed + offset)[1]
            shards.save_shard(out / "part-00000.npz", _dump(labels, seed + offset))
    for var, value in (
        (paths.DATA_DIR_VAR, tmp_path / "data"),
        (paths.RUNS_DIR_VAR, runs),
        (paths.EMBED_DIR_VAR, embed),
        (paths.NOTES_DIR_VAR, notes),
        (paths.KALEIDO_CHROME_VAR, ""),
    ):
        monkeypatch.setenv(var, str(value))
    monkeypatch.chdir(tmp_path)
    common = [
        "--config",
        str(REPO / "configs" / "base.yaml"),
        "--frozen-linear",
        "toy",
        "--set",
        "eval.num_randomizations=2",
        "--set",
        "eval.num_bootstrap=20",
        "--set",
        "eval.cv_folds=3",
    ]
    app = _load_app()
    assert app.main([*common, "--frozen-list-runs"]) == 0
    listed = capsys.readouterr().out.split()
    assert listed == ["toy_lam0.05_s0", "toy_lam0.05_s1"]
    with pytest.raises(FileNotFoundError):  # aggregating before the units ran
        app.main([*common, "--note", "idea/frozen.md", "--no-figures"])
    for name in listed:
        assert app.main([*common, "--frozen-run", name]) == 0
    shard_dir = runs / "frozen_linear" / "default"
    assert sorted(p.name for p in shard_dir.glob("*.npz")) == [
        f"{n}.npz" for n in listed
    ]
    assert app.main([*common, "--note", "idea/frozen.md", "--no-figures"]) == 0
    text = (notes / "idea" / "frozen.md").read_text()
    for section in (
        "change",
        "cost",
        "flips",
        "correlations",
        "grid",
        "decomposition",
        "curves",
    ):
        assert f"BEGIN GENERATED: analyze.frozen_{section}" in text
    assert "| 0.1 |" not in text
    assert app.main([*common, "--note", "idea/frozen.md", "--no-figures"]) == 0
    assert (notes / "idea" / "frozen.md").read_text() == text  # idempotent
