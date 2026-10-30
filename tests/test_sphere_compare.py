"""E3 comparison of a radius-free model with the radius model, end to end."""

import importlib.util
import json
import math
import pathlib
import types

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import ood_radius
from lejepa_norm.eval import randomization
from lejepa_norm.eval import retrieval
from lejepa_norm.eval import shards
from lejepa_norm.eval import shift
from lejepa_norm.eval import sphere_compare

REPO = pathlib.Path(__file__).parent.parent
D, CLASSES, VIEWS = 8, 3, 3  # VIEWS = center + 2 augmented
KNN = randomization.KNN_EUCLIDEAN
SEEDS = (0, 1)


def _embeddings(
    labels: numpy.ndarray, seed: int, noise: float, radius_scale: float
) -> numpy.ndarray:
    """`(n, VIEWS, D)`: class in the direction, per-image radius, jittered views."""
    rng = numpy.random.default_rng(seed)
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    base = noise * rng.normal(size=(labels.size, 1, D)) + centers[labels][:, None]
    u = features.direction(base + 0.1 * rng.normal(size=(labels.size, VIEWS, D)))
    log_r = 0.4 * rng.normal(size=(labels.size, 1)) + numpy.log(radius_scale)
    return u * numpy.exp(log_r)[..., None]


def _shard(
    z: numpy.ndarray,
    labels: numpy.ndarray,
    image_id: numpy.ndarray | None = None,
    corruption: str = "",
    severity: int = 0,
) -> shards.EmbeddingShard:
    n = labels.size
    z32 = z.astype(numpy.float32)
    return shards.EmbeddingShard(
        z=z32,
        h=z32.copy(),
        image_id=numpy.arange(n, dtype=numpy.int64) if image_id is None else image_id,
        label=labels.astype(numpy.int64),
        corruption=numpy.array([corruption] * n),
        severity=numpy.full(n, severity, dtype=numpy.int8),
    )


def _write_dumps(
    embed: pathlib.Path, name: str, seed: int, radius_scale: float
) -> None:
    rng = numpy.random.default_rng(seed + 50)
    labels = {
        dumps.STL10_TRAIN: rng.integers(0, CLASSES, 90),
        dumps.STL10_TEST: rng.integers(0, CLASSES, 60),
    }
    sets = {
        dumps.STL10_TRAIN: _shard(
            _embeddings(labels[dumps.STL10_TRAIN], seed, 0.6, radius_scale),
            labels[dumps.STL10_TRAIN],
        ),
        dumps.STL10_TEST: _shard(
            _embeddings(labels[dumps.STL10_TEST], seed + 10, 0.6, radius_scale),
            labels[dumps.STL10_TEST],
        ),
    }
    corrupted = [
        _shard(
            _embeddings(labels[dumps.STL10_TEST], seed + 20 + s, 0.9, radius_scale),
            labels[dumps.STL10_TEST],
            numpy.arange(60, dtype=numpy.int64),
            corruption,
            s,
        )
        for corruption in ("blur", "noise")
        for s in (1, 2)
    ]
    sets[dumps.STL10_C] = dumps.concatenate(corrupted)
    for ood, offset in ((dumps.DTD_TEST, 30), (dumps.SVHN_TEST, 40)):
        ood_labels = rng.integers(0, CLASSES, 50)
        sets[ood] = _shard(
            _embeddings(ood_labels, seed + offset, 2.0, radius_scale), ood_labels
        )
    for eval_set, shard in sets.items():
        out = dumps.set_dir(embed, name, eval_set)
        out.mkdir(parents=True)
        shards.save_shard(out / "part-00000.npz", shard)


def _write_run(runs: pathlib.Path, name: str, seed: int, radius_free: bool) -> None:
    cfg = config.ExperimentConfig(seed=seed)
    cfg.loss.lambda_ = 0.05
    cfg.loss.radius_free = radius_free
    cfg.train.epochs = 2
    config.save_resolved(cfg, runs / name)
    lines = [
        {
            "kind": "epoch",
            "epoch": epoch,
            "step": epoch,
            "loss_inv_epoch_mean": 1.0 / epoch + 0.01 * seed,
            "loss_dist_epoch_mean": 0.5,
            "kappa_z_epoch_mean": 0.0 if radius_free else 20.0,
            "knn_acc_h": 0.5 + 0.1 * epoch,
            "knn_acc_z": 0.4 + 0.1 * epoch,
        }
        for epoch in (1, 2)
    ]
    (runs / name / "metrics.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_to_sphere_gives_constant_norm_and_keeps_the_direction() -> None:
    x = numpy.random.default_rng(0).normal(size=(5, D)) * numpy.arange(1, 6)[:, None]
    s = sphere_compare.to_sphere(x)
    numpy.testing.assert_allclose(features.radius(s), math.sqrt(D))
    numpy.testing.assert_allclose(features.direction(s), features.direction(x))


def test_clean_bank_baseline_equals_the_shift_unit_baseline() -> None:
    rng = numpy.random.default_rng(0)
    by, ty = rng.integers(0, CLASSES, 90), rng.integers(0, CLASSES, 60)
    bank = _embeddings(by, 1, 0.6, 1.0)[:, 0]
    test_z = _embeddings(ty, 2, 0.6, 1.0)[:, 0]
    clean = shards.CenterShard(
        z=test_z.astype(numpy.float32),
        image_id=numpy.arange(60, dtype=numpy.int64),
        label=ty.astype(numpy.int64),
        corruption=numpy.array([""] * 60),
        severity=numpy.zeros(60, dtype=numpy.int8),
    )
    corrupted = shards.CenterShard(
        z=_embeddings(ty, 3, 0.9, 1.0)[:, 0].astype(numpy.float32),
        image_id=numpy.arange(60, dtype=numpy.int64),
        label=ty.astype(numpy.int64),
        corruption=numpy.array(["blur"] * 60),
        severity=numpy.full(60, 1, dtype=numpy.int8),
    )
    readers = (KNN, randomization.LINEAR)
    mine, ids = sphere_compare.evaluate_clean_bank_cells(
        bank, by, clean, corrupted, k=5, seed=0, readers=readers
    )
    reference, ref_ids, _, _ = shift.evaluate_cells(
        bank, by, clean, corrupted, 5, 1, 3, 0, readers=readers
    )
    for key, value in mine.items():
        numpy.testing.assert_array_equal(value, reference[key])
    for cell, value in ids.items():
        numpy.testing.assert_array_equal(value, ref_ids[cell])


def test_accuracy_comparison_certifies_equality_and_detects_a_gap() -> None:
    rng = numpy.random.default_rng(0)
    cells = [(shift.CLEAN, 0), ("blur", 1), ("noise", 1)]
    ids = {c: numpy.arange(2000, dtype=numpy.int64) for c in cells}

    def unit(name: str, seed: int, hit_rate: float) -> shift.ShiftUnit:
        correct: dict[shift.Key, numpy.ndarray] = {}
        for reader in sphere_compare.READERS:
            for protocol, condition in (
                (randomization.FROZEN, randomization.BASELINE),
                (randomization.RETRAIN, randomization.BASELINE),
                (randomization.RETRAIN, randomization.CONSTANT),
            ):
                for corruption, severity in cells:
                    key = (reader, protocol, condition, corruption, severity)
                    correct[key] = (rng.random(2000) < hit_rate).astype(numpy.float64)
        return shift.ShiftUnit(name, 0.05, seed, correct, ids, {}, {})

    same = sphere_compare.pair_by_seed(
        [unit("a0", 0, 0.7), unit("a1", 1, 0.7)],
        [unit("b0", 0, 0.7), unit("b1", 1, 0.7)],
    )
    worse = sphere_compare.pair_by_seed(
        [unit("a0", 0, 0.6), unit("a1", 1, 0.6)],
        [unit("b0", 0, 0.7), unit("b1", 1, 0.7)],
    )
    # Independent samples of 2000 images x 2 seeds: sd of the difference ~1 point.
    wide = sphere_compare.compare_accuracy(same, 0.03, 200, numpy.random.default_rng(1))
    gap = sphere_compare.compare_accuracy(
        worse, 0.005, 200, numpy.random.default_rng(1)
    )
    assert len(wide) == len(gap) == 4 * len(sphere_compare.READERS) * 2
    assert all(c.verdict == randomization.EQUIVALENT for c in wide)  # margin 3 points
    clean = [c for c in gap if c.level == sphere_compare.CLEAN_LEVEL]
    assert all(c.difference < -0.05 for c in clean)
    assert all(c.verdict == randomization.DIFFERENT for c in clean)
    pooled = next(
        c
        for c in gap
        if c.level == sphere_compare.OVERALL_LEVEL
        and c.reader == KNN
        and c.reference == sphere_compare.AS_TRAINED
    )
    assert pooled.difference == pytest.approx(-0.1, abs=0.04)
    with pytest.raises(ValueError):
        sphere_compare.pair_by_seed([unit("a0", 0, 0.7)], [unit("b1", 1, 0.7)])


def test_non_inferiority_and_svhn_reading() -> None:
    assert sphere_compare.non_inferiority((-0.002, 0.01), (-0.004, 0.012), 0.005) == (
        sphere_compare.NON_INFERIOR
    )
    assert sphere_compare.non_inferiority((-0.03, -0.01), (-0.04, -0.008), 0.005) == (
        sphere_compare.INFERIOR
    )
    assert sphere_compare.non_inferiority((-0.02, 0.0), (-0.03, 0.01), 0.005) == (
        sphere_compare.UNDECIDED
    )
    clearly_lower = (-0.1 + 0.001 * numpy.arange(101)).astype(numpy.float64)
    assert sphere_compare.svhn_prediction(clearly_lower) == sphere_compare.LOWER
    noisy = numpy.concatenate([[-0.01], numpy.linspace(-0.05, 0.05, 100)])
    assert sphere_compare.svhn_prediction(noisy) == (
        sphere_compare.DIRECTION_ONLY_LOWER
    )
    assert (
        sphere_compare.svhn_prediction(numpy.asarray(-noisy, dtype=numpy.float64))
        == sphere_compare.NOT_LOWER
    )


def _ood_unit(name: str, seed: int, informative_radius: bool) -> ood_radius.OodUnit:
    rng = numpy.random.default_rng(seed)

    def make(n: int, shifted: bool) -> numpy.ndarray:
        centers = numpy.random.default_rng(5).normal(size=(CLASSES, D))
        labels = rng.integers(0, CLASSES, n)
        u = features.direction(0.4 * rng.normal(size=(n, D)) + centers[labels])
        if shifted:
            u = features.direction(u + 0.5 * rng.normal(size=(n, D)))
        shift_r = 0.8 if (shifted and informative_radius) else 0.0
        return u * numpy.exp(0.15 * rng.normal(size=n) + shift_r)[:, None]

    scores = {
        "svhn_test": ood_radius.compute_scores(
            make(200, False), make(150, False), make(120, True), k=5, seed=seed
        )
    }
    return ood_radius.OodUnit(name, 0.05, seed, 150, scores)


def test_ood_comparison_pairs_resamples_and_reads_the_prediction() -> None:
    radius_units = [_ood_unit(f"r{s}", s, True) for s in SEEDS]
    # Identical units under identical resamples: every paired difference is zero.
    pairs = sphere_compare.pair_by_seed(
        [_ood_unit(f"r{s}", s, True) for s in SEEDS], radius_units
    )
    same = sphere_compare.compare_ood(pairs, 30, 0)[0]
    zero = same.sphere - same.radius[ood_radius.KNN_DIRECTION]
    numpy.testing.assert_array_equal(zero, 0.0)
    # The radius carries the OOD signal for the radius model: its best radius-using
    # score beats the radius-free direction-only score, and the reading says so.
    pairs = sphere_compare.pair_by_seed(
        [_ood_unit(f"s{s}", s, False) for s in SEEDS], radius_units
    )
    result = sphere_compare.compare_ood(pairs, 60, 0)[0]
    best = result.radius[result.best_radius_score]
    assert best[0] > result.sphere[0] + 0.02
    assert sphere_compare.svhn_prediction(result.sphere - best) == sphere_compare.LOWER
    rows = sphere_compare.ood_difference_rows([result])
    assert len(rows) == len(sphere_compare.OOD_REFERENCES)
    assert sum(1 for r in rows if r["prediction 3 reading"]) == 1


def test_app_computes_units_and_writes_the_comparison_note(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, embed, notes = tmp_path / "runs", tmp_path / "embed", tmp_path / "notes"
    for seed in SEEDS:
        _write_run(runs, f"toyradius_lam0.05_s{seed}", seed, False)
        _write_dumps(embed, f"toyradius_lam0.05_s{seed}", seed, 1.0)
        _write_run(runs, f"toysphere_lam0.05_s{seed}", seed, True)
        _write_dumps(embed, f"toysphere_lam0.05_s{seed}", seed + 5, 3.0)
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
        "--set",
        "eval.num_randomizations=1",
        "--set",
        "eval.num_bootstrap=20",
        "--set",
        "eval.cv_folds=3",
        "--set",
        "eval.knn_k=5",
    ]
    app = _load_app()
    for seed in SEEDS:
        radius = f"toyradius_lam0.05_s{seed}"
        assert app.main([*common, "--run-name", radius, "--shift-unit"]) == 0
        argv = [*common, "--retrieval", "toyradius", "--retrieval-run", radius]
        assert app.main(argv) == 0
        sphere = f"toysphere_lam0.05_s{seed}"
        argv = [
            *common,
            "--compare-prefix",
            "toysphere",
            "--baseline-prefix",
            "toyradius",
            "--compare-unit",
            sphere,
        ]
        assert app.main(argv) == 0
        assert (runs / sphere / sphere_compare.CELLS_FILE).exists()
    # A run that was not trained radius-free has no sphere unit.
    with pytest.raises(ValueError, match="radius_free"):
        app.main(
            [
                *common,
                "--compare-prefix",
                "toysphere",
                "--baseline-prefix",
                "toyradius",
                "--compare-unit",
                "toyradius_lam0.05_s0",
            ]
        )
    argv = [
        *common,
        "--compare-prefix",
        "toysphere",
        "--baseline-prefix",
        "toyradius",
        "--note",
        "idea/sphere.md",
        "--no-figures",
    ]
    assert app.main(argv) == 0
    text = (notes / "idea" / "sphere.md").read_text()
    for name in ("inputs", "readings", "accuracy", "retrieval", "ood", "health"):
        assert f"BEGIN GENERATED: analyze.sphere_{name}" in text
    assert "prediction 1" in text and "prediction 2" in text
    assert "prediction 3" in text and "Training healthy" in text
    accuracy_rows = [
        ln for ln in text.splitlines() if ln.startswith("| 0.05 | clean STL-10 test")
    ]
    # 3 readers x 2 references at the clean level.
    assert len(accuracy_rows) == 3 * 2
    assert app.main(argv) == 0
    assert (notes / "idea" / "sphere.md").read_text() == text  # idempotent
    # Same retrieval comparison through the library: both conditions coincide.
    result = retrieval.load_result(
        runs
        / "toysphere_lam0.05_s0"
        / sphere_compare.RETRIEVAL_DIR
        / "toysphere_lam0.05_s0.npz"
    )
    for (task, metric, condition), values in result.values.items():
        if condition == retrieval.CONSTANT:
            numpy.testing.assert_array_equal(
                values, result.values[task, metric, retrieval.BASELINE]
            )
