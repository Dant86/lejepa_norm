"""Tests of the radius-vs-confidence study (`confidence`, `usage_confidence`)."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

import importlib.util
import pathlib
import types

import numpy
import pytest
import sklearn.metrics

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import confidence
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import shards

REPO = pathlib.Path(__file__).parent.parent
D, CLASSES = 8, 3


def _query(
    n: int, radius_tracks_errors: bool, seed: int = 0
) -> tuple[confidence.Query, numpy.ndarray]:
    """Readers with an uninformative `p1`; the log-radius is lower on errors iff
    `radius_tracks_errors`."""
    rng = numpy.random.default_rng(seed)
    label = rng.integers(0, CLASSES, n)
    error = rng.random(n) < 0.3
    pred = numpy.where(error, (label + 1) % CLASSES, label)
    rho = rng.normal(size=n) - (1.2 * error if radius_tracks_errors else 0.0)
    readers = {
        name: confidence.ReaderOutput(
            p1=rng.uniform(0.4, 1.0, n), margin=rng.uniform(0.0, 0.5, n), pred=pred
        )
        for name in confidence.READERS
    }
    return confidence.Query(rho, rho * 0.0, label, readers), error


def test_booter_auroc_and_coverage_match_references() -> None:
    rng = numpy.random.default_rng(1)
    score, positive = rng.normal(size=300), rng.random(300) < 0.4
    booter = confidence.Booter(300, 10, seed=0)
    auroc = booter.auroc(score, positive)
    assert auroc.shape == (11,)
    assert auroc[0] == pytest.approx(sklearn.metrics.roc_auc_score(positive, score))
    correct = rng.random(300) < 0.7
    keep = booter.coverage_accuracy(score, correct, 0.5)
    top = numpy.argsort(-score)[:150]
    assert keep[0] == pytest.approx(correct[top].mean(), abs=0.02)
    mean = booter.mean(score)
    assert mean[0] == pytest.approx(score.mean())


def test_radius_that_tracks_errors_is_detected_and_adds_beyond_confidence() -> None:
    query, _ = _query(2000, radius_tracks_errors=True)
    booter = confidence.Booter(2000, 100, seed=0)
    stats = confidence.set_statistics(query, numpy.zeros(CLASSES), booter, seed=0)
    unit = confidence.Unit(
        "toy", "lam0.05", 0, "z", {(confidence.CLEAN, m): v for m, v in stats.items()}
    )
    reader = confidence.PROBE_U
    rho = confidence.aggregate([unit], confidence.CLEAN, f"{reader}|auroc[-rho]")
    assert rho.value > 0.7
    assert confidence.signal_verdict(rho) == confidence.INFORMATIVE
    gain = confidence.aggregate([unit], confidence.CLEAN, f"{reader}|d_auroc[M1-M0]")
    assert gain.value > 0.1
    assert (
        confidence.difference_verdict(gain, confidence.AUROC_MARGIN) == confidence.ADDS
    )
    p1 = confidence.aggregate([unit], confidence.CLEAN, f"{reader}|auroc[-p1]")
    assert confidence.signal_verdict(p1) != confidence.INFORMATIVE  # p1 is noise


def test_radius_independent_of_errors_is_not_informative_and_adds_nothing() -> None:
    query, _ = _query(2000, radius_tracks_errors=False, seed=3)
    booter = confidence.Booter(2000, 100, seed=1)
    stats = confidence.set_statistics(query, numpy.zeros(CLASSES), booter, seed=0)
    unit = confidence.Unit(
        "toy", "lam0.05", 0, "z", {(confidence.CLEAN, m): v for m, v in stats.items()}
    )
    reader = confidence.KNN_U
    rho = confidence.aggregate([unit], confidence.CLEAN, f"{reader}|auroc[-rho]")
    assert abs(rho.value - 0.5) < 0.05
    assert confidence.signal_verdict(rho) != confidence.INFORMATIVE
    gain = confidence.aggregate([unit], confidence.CLEAN, f"{reader}|d_auroc[M1-M0]")
    assert abs(gain.value) < 0.03
    assert (
        confidence.difference_verdict(gain, confidence.AUROC_MARGIN) != confidence.ADDS
    )


def test_ood_statistics_find_a_smaller_radius_and_a_detector_gain() -> None:
    seen, _ = _query(800, radius_tracks_errors=False, seed=4)
    other, _ = _query(1200, radius_tracks_errors=False, seed=5)
    other.rho = other.rho - 1.0  # OOD images have the smaller radius
    stats = confidence.ood_statistics(
        seen,
        other,
        confidence.Booter(800, 50, seed=0),
        confidence.Booter(1200, 50, seed=1),
        seed=0,
    )
    reader = confidence.PROBE_U
    assert stats[f"{reader}|auroc_ood[-rho]"][0] > 0.7
    assert stats[f"{reader}|d_auroc_ood[M1-M0]"][0] > 0.1
    assert stats["shift_rho"][0] == pytest.approx(
        other.rho.mean() - seen.rho.mean(), abs=1e-9
    )


def test_unit_files_round_trip_and_point_only_metrics_have_no_interval(
    tmp_path: pathlib.Path,
) -> None:
    unit = confidence.Unit(
        "toy",
        "lam0.1",
        2,
        "h",
        {
            (confidence.CLEAN, "a"): numpy.array([0.6, 0.55, 0.65]),
            ("corr:x", "b"): numpy.array([0.1]),
        },
    )
    confidence.save_unit(tmp_path / "u.npz", unit)
    loaded = confidence.load_unit(tmp_path / "u.npz")
    assert (loaded.name, loaded.group, loaded.seed, loaded.space) == (
        "toy",
        "lam0.1",
        2,
        "h",
    )
    assert numpy.allclose(loaded.stats[confidence.CLEAN, "a"], [0.6, 0.55, 0.65])
    assert confidence.aggregate([loaded], "corr:x", "b").ci95 is None
    assert confidence.aggregate([loaded], confidence.CLEAN, "a").ci95 is not None
    assert numpy.isnan(confidence.aggregate([loaded], "nope", "b").value)


def test_out_of_fold_scores_are_nan_for_a_tiny_class() -> None:
    design = numpy.random.default_rng(0).normal(size=(50, 2))
    target = numpy.zeros(50, dtype=bool)
    target[:2] = True
    assert numpy.isnan(confidence.out_of_fold_scores(design, target, 0)).all()


def _embed(labels: numpy.ndarray, seed: int, drop: float = 0.0) -> numpy.ndarray:
    """Class in the direction; the radius is low where the point is far from its
    class centre (a confidence-like radius), optionally shifted by `drop`."""
    rng = numpy.random.default_rng(seed)
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    raw = 0.9 * rng.normal(size=(labels.size, D)) + 2.0 * centers[labels]
    u = features.direction(raw)
    closeness = (u * features.direction(centers[labels])).sum(axis=1)
    return u * numpy.exp(1.0 + closeness - drop)[:, None]


def _center(
    z: numpy.ndarray, labels: numpy.ndarray, corruption: str, severity: int
) -> shards.CenterShard:
    n = labels.size
    return shards.CenterShard(
        z=z.astype(numpy.float32),
        image_id=numpy.arange(n, dtype=numpy.int64),
        label=labels.astype(numpy.int64),
        corruption=numpy.array([corruption] * n),
        severity=numpy.full(n, severity, dtype=numpy.int8),
    )


def _write_dump(
    embed: pathlib.Path, name: str, eval_set: str, center: shards.CenterShard
) -> None:
    z = numpy.repeat(center.z[:, None, :], 3, axis=1)
    out = dumps.set_dir(embed, name, eval_set)
    out.mkdir(parents=True, exist_ok=True)
    shards.save_shard(
        out / "part-00000.npz",
        shards.EmbeddingShard(
            z=z,
            h=z * 2.0,  # h differs from z so the space option is observable
            image_id=center.image_id,
            label=center.label,
            corruption=center.corruption,
            severity=center.severity,
        ),
    )


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_confidence_modes_write_units_and_notes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, embed, notes = tmp_path / "runs", tmp_path / "embed", tmp_path / "notes"
    rng = numpy.random.default_rng(0)
    by, ty = rng.integers(0, CLASSES, 160), rng.integers(0, CLASSES, 200)
    for name, lam, seed in (
        ("toy_lam0.05_s0", 0.05, 0),
        ("toy_lam0.05_s1", 0.05, 1),
        ("toy_lam0.1_s0", 0.1, 0),
    ):
        cfg = config.ExperimentConfig(seed=seed)
        cfg.loss.lambda_ = lam
        config.save_resolved(cfg, runs / name)
        _write_dump(
            embed, name, dumps.STL10_TRAIN, _center(_embed(by, seed), by, "", 0)
        )
        _write_dump(
            embed, name, dumps.STL10_TEST, _center(_embed(ty, 5 + seed), ty, "", 0)
        )
        cells = [_center(_embed(ty, 20 + 3 * s, 0.4 * s), ty, "dim", s) for s in (1, 2)]
        _write_dump(
            embed,
            name,
            dumps.STL10_C,
            shards.CenterShard(
                z=numpy.concatenate([c.z for c in cells]),
                image_id=numpy.concatenate([c.image_id for c in cells]),
                label=numpy.concatenate([c.label for c in cells]),
                corruption=numpy.concatenate([c.corruption for c in cells]),
                severity=numpy.concatenate([c.severity for c in cells]),
            ),
        )
        _write_dump(embed, name, dumps.DTD_TEST, _center(_embed(ty, 7, 0.8), ty, "", 0))
    for var, value in (
        (paths.DATA_DIR_VAR, tmp_path / "data"),
        (paths.RUNS_DIR_VAR, runs),
        (paths.EMBED_DIR_VAR, embed),
        (paths.NOTES_DIR_VAR, notes),
        (paths.KALEIDO_CHROME_VAR, ""),
    ):
        monkeypatch.setenv(var, str(value))
    monkeypatch.chdir(tmp_path)
    base = [
        "--config",
        str(REPO / "configs" / "base.yaml"),
        "--set",
        "eval.knn_k=5",
        "--set",
        "eval.num_bootstrap=20",
    ]
    app = _load_app()
    for name in ("toy_lam0.05_s0", "toy_lam0.05_s1", "toy_lam0.1_s0"):
        assert app.main([*base, "--run-name", name, "--confidence-unit"]) == 0
        assert (runs / name / "confidence_z.npz").is_file()
    assert (
        app.main(
            [
                *base,
                "--run-name",
                "toy_lam0.1_s0",
                "--confidence-unit",
                "--confidence-space",
                "h",
            ]
        )
        == 0
    )
    argv = [
        *base,
        "--confidence",
        "toy",
        "--note",
        "idea/confidence.md",
        "--no-figures",
    ]
    assert app.main(argv) == 0
    text = (notes / "idea" / "confidence_z_lam0.05.md").read_text()
    for section in (
        "confidence_inputs",
        "confidence_clean",
        "confidence_nested",
        "confidence_corrupted",
        "confidence_per_corruption",
        "confidence_shift",
        "confidence_cells",
        "confidence_ood",
    ):
        assert f"BEGIN GENERATED: analyze.{section}" in text, section
    assert "ood:dtd" in text and "ood:svhn" not in text
    summary = (notes / "idea" / "confidence.md").read_text()
    assert "analyze.confidence_summary_z" in summary
    assert app.main(argv) == 0
    assert (notes / "idea" / "confidence_z_lam0.05.md").read_text() == text
    # The dimmed corrupted images have a smaller radius than their clean twins.
    units = confidence.load_unit(runs / "toy_lam0.05_s0" / "confidence_z.npz")
    shift = confidence.aggregate([units], "corr:dim", "shift_rho")
    assert shift.value < 0
    # `--confidence-part ood` keeps the other groups and rewrites only the OOD ones.
    before = confidence.load_unit(runs / "toy_lam0.05_s0" / "confidence_z.npz")
    assert (
        app.main(
            [
                *base,
                "--run-name",
                "toy_lam0.05_s0",
                "--confidence-unit",
                "--confidence-part",
                "ood",
            ]
        )
        == 0
    )
    after = confidence.load_unit(runs / "toy_lam0.05_s0" / "confidence_z.npz")
    assert set(after.stats) == set(before.stats)
    key = (confidence.CLEAN, f"{confidence.PROBE_U}|auroc[-rho]")
    assert numpy.array_equal(after.stats[key], before.stats[key], equal_nan=True)
    assert (confidence.CLEAN, f"{confidence.PROBE_U}|d_auroc[M3-M0]") in after.stats
    h_unit = confidence.load_unit(runs / "toy_lam0.1_s0" / "confidence_h.npz")
    assert h_unit.space == "h"
