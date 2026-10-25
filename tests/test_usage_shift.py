"""Radius use under shift: corrupted-test readers and OOD scores, end to end."""

import importlib.util
import pathlib
import types

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import metrics
from lejepa_norm.eval import ood_radius
from lejepa_norm.eval import randomization
from lejepa_norm.eval import shards
from lejepa_norm.eval import shift

REPO = pathlib.Path(__file__).parent.parent
D, CLASSES = 8, 3
KNN = randomization.KNN_EUCLIDEAN
READERS = (KNN, randomization.LINEAR)


def _embed(
    labels: numpy.ndarray, seed: int, radius_scale: float = 1.0, noise: float = 0.8
) -> numpy.ndarray:
    """Class carried by the direction; radius ~ 1 times `radius_scale`."""
    rng = numpy.random.default_rng(seed)
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    u = features.direction(noise * rng.normal(size=(labels.size, D)) + centers[labels])
    log_r = 0.5 * rng.normal(size=labels.size) + numpy.log(radius_scale)
    return u * numpy.exp(log_r)[:, None]


def _center_shard(
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


def _corrupted(labels: numpy.ndarray) -> shards.CenterShard:
    """`rescale` multiplies the radius by 6 (a pure radius shift), `jitter` keeps
    the radius distribution and adds direction noise; two severities each."""
    parts = [
        _center_shard(_embed(labels, 30 + s, 6.0), labels, "rescale", s) for s in (1, 2)
    ] + [
        _center_shard(_embed(labels, 40 + s, 1.0, 0.8 + 0.2 * s), labels, "jitter", s)
        for s in (1, 2)
    ]
    return shards.CenterShard(
        z=numpy.concatenate([p.z for p in parts]),
        image_id=numpy.concatenate([p.image_id for p in parts]),
        label=numpy.concatenate([p.label for p in parts]),
        corruption=numpy.concatenate([p.corruption for p in parts]),
        severity=numpy.concatenate([p.severity for p in parts]),
    )


def _units() -> list[shift.ShiftUnit]:
    rng = numpy.random.default_rng(0)
    by = rng.integers(0, CLASSES, 150)
    ty = rng.integers(0, CLASSES, 90)
    bank = _embed(by, 10)
    clean = _center_shard(_embed(ty, 11), ty, "", 0)
    units: list[shift.ShiftUnit] = []
    for seed in (0, 1):
        correct, ids, radius_shift, residual = shift.evaluate_cells(
            bank,
            by,
            clean,
            _corrupted(ty),
            k=5,
            num_repeats=2,
            cv_folds=3,
            seed=seed,
            readers=READERS,
        )
        units.append(
            shift.ShiftUnit("toy", 0.05, seed, correct, ids, radius_shift, residual)
        )
    return units


def test_center_shard_loading_matches_the_full_shard(tmp_path: pathlib.Path) -> None:
    rng = numpy.random.default_rng(0)
    full = shards.EmbeddingShard(
        z=rng.normal(size=(6, 3, 4)).astype(numpy.float32),
        h=rng.normal(size=(6, 3, 5)).astype(numpy.float32),
        image_id=numpy.arange(6, dtype=numpy.int64),
        label=numpy.arange(6, dtype=numpy.int64) % 2,
        corruption=numpy.array(["a"] * 6),
        severity=numpy.arange(6, dtype=numpy.int8),
    )
    out = dumps.set_dir(tmp_path, "run", dumps.STL10_C)
    out.mkdir(parents=True)
    shards.save_shard(out / "part-00000.npz", full)
    shards.save_shard(out / "part-00001.npz", full)
    center = dumps.load_center_set(out)
    assert center is not None
    numpy.testing.assert_array_equal(
        center.z, numpy.concatenate([full.z[:, 0], full.z[:, 0]])
    )
    assert center.corruption.size == 12 and center.severity.dtype == numpy.int8
    assert dumps.load_center_set(tmp_path / "nothing") is None


def test_radius_shift_hurts_frozen_readers_and_normalizing_recovers() -> None:
    units = _units()
    cells = [("rescale", 1), ("rescale", 2), ("jitter", 1), ("jitter", 2)]
    assert units[0].cells() == [(shift.CLEAN, 0), *sorted(cells)]
    # The rescale corruption shifts the log-radius by log 6; jitter does not.
    assert units[0].radius_shift["rescale", 1] == pytest.approx(numpy.log(6), abs=0.3)
    assert abs(units[0].radius_shift["jitter", 1]) < 0.3

    def acc(unit: shift.ShiftUnit, cond: str, cell: tuple[str, int]) -> float:
        key = (KNN, randomization.FROZEN, cond, *cell)
        return float(unit.correct[key].mean())

    base, const = (
        acc(units[0], "baseline", ("rescale", 1)),
        acc(units[0], randomization.CONSTANT, ("rescale", 1)),
    )
    assert const > base  # the shifted radius hurts; normalizing recovers
    scaled = acc(units[0], shift.SCALED_CONSTANT, ("rescale", 1))
    assert scaled > base
    # Retrain baseline is the frozen baseline (same reader, same bank).
    key = (KNN, randomization.RETRAIN, "baseline", "rescale", 1)
    numpy.testing.assert_array_equal(
        units[0].correct[key],
        units[0].correct[(*key[:1], randomization.FROZEN, *key[2:])],
    )

    comparisons = shift.compare(
        units, shift.EQUIVALENCE_MARGIN, 100, numpy.random.default_rng(0)
    )
    index = {
        (c.reader, c.protocol, c.condition, c.corruption, c.severity): c
        for c in comparisons
    }
    # readers x protocols x 4 conditions x (cells + corruptions + severities + 1)
    assert len(comparisons) == 2 * 2 * 4 * 9
    gain = index[KNN, randomization.FROZEN, randomization.CONSTANT, "rescale", 0]
    assert gain.difference > 0 and gain.verdict in {
        randomization.DIFFERENT,
        randomization.INCONCLUSIVE,
    }
    overall = index[KNN, randomization.FROZEN, randomization.CONSTANT, shift.OVERALL, 0]
    pooled = numpy.mean(
        [
            index[KNN, randomization.FROZEN, randomization.CONSTANT, c, 0].difference
            for c in ("rescale", "jitter")
        ]
    )
    assert overall.difference == pytest.approx(pooled)
    for c in comparisons:
        assert c.ci95[0] <= c.difference <= c.ci95[1] or c.ci95[0] == c.ci95[1]
    assert len(shift.cell_rows(comparisons, 0.05)) == 2 * 4 * 4
    assert shift.corruption_rows(comparisons, 0.05)
    context = shift.context_rows(units, comparisons)
    assert context[0]["corruption"] == "clean test (reference)"
    assert len(shift.shift_correlations(units)) == 2 * 4
    assert len(shift.figures(comparisons, 0.05)) == 2 * len(READERS)


def test_unit_files_round_trip(tmp_path: pathlib.Path) -> None:
    unit = _units()[0]
    shift.save_unit(tmp_path / "u.npz", unit)
    back = shift.load_unit(tmp_path / "u.npz")
    assert back.name == "toy" and back.lam == 0.05 and back.seed == 0
    assert back.correct.keys() == unit.correct.keys()
    key = (KNN, randomization.FROZEN, "baseline", "rescale", 1)
    numpy.testing.assert_allclose(back.correct[key], unit.correct[key])
    assert back.radius_shift == pytest.approx(unit.radius_shift)


def test_roc_statistics_match_the_reference_auroc_and_fpr() -> None:
    rng = numpy.random.default_rng(0)
    scores = (
        rng.normal(size=(300, 2))
        + numpy.array([0.8, 0.0]) * (numpy.arange(300) >= 200)[:, None]
    )
    is_ood = numpy.arange(300) >= 200
    auroc, fpr = ood_radius.roc_statistics(scores, is_ood, numpy.ones((1, 300)))
    for c in range(2):
        assert auroc[0, c] == pytest.approx(metrics.auroc(scores[:, c], is_ood))
        # The threshold keeps exactly 95% of the 200 ID images.
        threshold = numpy.sort(scores[~is_ood, c])[189]
        assert fpr[0, c] == pytest.approx((scores[is_ood, c] <= threshold).mean())
    assert auroc[0, 0] > 0.65 and abs(auroc[0, 1] - 0.5) < 0.1


def _ood_units(
    radius_informative: bool, num_seeds: int = 2
) -> list[ood_radius.OodUnit]:
    units: list[ood_radius.OodUnit] = []
    for seed in range(num_seeds):
        rng = numpy.random.default_rng(seed)

        def make(
            n: int, shifted: bool, rng: numpy.random.Generator = rng
        ) -> numpy.ndarray:
            centers = numpy.random.default_rng(5).normal(size=(CLASSES, D))
            labels = rng.integers(0, CLASSES, n)
            u = features.direction(0.4 * rng.normal(size=(n, D)) + centers[labels])
            if shifted:  # OOD directions: same family, more noise
                u = features.direction(u + 0.5 * rng.normal(size=(n, D)))
            shift_r = 0.8 if (shifted and radius_informative) else 0.0
            return u * numpy.exp(0.15 * rng.normal(size=n) + shift_r)[:, None]

        bank, test, ood = make(300, False), make(200, False), make(150, True)
        scores = {"toy_ood": ood_radius.compute_scores(bank, test, ood, k=5, seed=seed)}
        units.append(ood_radius.OodUnit(f"toy{seed}", 0.05, seed, 200, scores))
    return units


def test_radius_that_carries_ood_information_is_detected(
    tmp_path: pathlib.Path,
) -> None:
    informative = ood_radius.evaluate(
        _ood_units(True), 60, numpy.random.default_rng(0), chunk=25
    )[0]
    assert informative.auroc[ood_radius.RADIUS][0] > 0.85
    assert ood_radius.radius_verdict(informative.auroc[ood_radius.RADIUS]) == (
        ood_radius.INFORMATIVE
    )
    gain = (
        informative.auroc[ood_radius.COMBINED]
        - informative.auroc[ood_radius.KNN_DIRECTION]
    )
    assert ood_radius.addition_verdict(gain) == ood_radius.ADDS
    # Permuting the radius removes what it carried: raw k-NN drops, radius -> 0.5.
    assert informative.auroc[ood_radius.KNN_RAW][0] > (
        informative.auroc[ood_radius.KNN_RAW_PERMUTED][0] + 0.03
    )
    assert abs(informative.auroc[ood_radius.DEVIATION_PERMUTED][0] - 0.5) < 0.1
    assert informative.oracle_weights  # one per seed
    rows = ood_radius.score_rows([informative]) + ood_radius.difference_rows(
        [informative]
    )
    assert len(rows) == len(ood_radius.SCORES) + len(ood_radius.DIFFERENCES)

    # Round trip of a unit file.
    unit = _ood_units(True, 1)[0]
    ood_radius.save_unit(tmp_path / "o.npz", unit)
    back = ood_radius.load_unit(tmp_path / "o.npz")
    assert back.num_id == 200 and back.scores.keys() == unit.scores.keys()


def test_radius_that_is_a_nuisance_does_not_add_to_the_direction_score() -> None:
    result = ood_radius.evaluate(
        _ood_units(False, 3), 100, numpy.random.default_rng(0), chunk=50
    )[0]
    assert ood_radius.radius_verdict(result.auroc[ood_radius.DEVIATION]) in {
        ood_radius.NOT_INFORMATIVE,
        ood_radius.INCONCLUSIVE,
    }
    gain = result.auroc[ood_radius.COMBINED] - result.auroc[ood_radius.KNN_DIRECTION]
    assert ood_radius.addition_verdict(gain) != ood_radius.ADDS
    assert ood_radius.addition_verdict(numpy.zeros(101)) == ood_radius.NOT_ADD


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            h=z.copy(),
            image_id=center.image_id,
            label=center.label,
            corruption=center.corruption,
            severity=center.severity,
        ),
    )


def test_analyze_shift_modes_write_unit_files_and_per_lambda_notes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, embed, notes = tmp_path / "runs", tmp_path / "embed", tmp_path / "notes"
    rng = numpy.random.default_rng(0)
    by, ty = rng.integers(0, CLASSES, 90), rng.integers(0, CLASSES, 60)
    for name, lam, seed in (
        ("toy_lam0.05_s0", 0.05, 0),
        ("toy_lam0.05_s1", 0.05, 1),
        ("toy_lam0.1_s0", 0.1, 0),
    ):
        cfg = config.ExperimentConfig(seed=seed)
        cfg.loss.lambda_ = lam
        config.save_resolved(cfg, runs / name)
        _write_dump(
            embed, name, dumps.STL10_TRAIN, _center_shard(_embed(by, seed), by, "", 0)
        )
        _write_dump(
            embed,
            name,
            dumps.STL10_TEST,
            _center_shard(_embed(ty, 5 + seed), ty, "", 0),
        )
        _write_dump(embed, name, dumps.STL10_C, _corrupted(ty))
        _write_dump(
            embed,
            name,
            dumps.DTD_TEST,
            _center_shard(_embed(ty, 7, 2.0, 1.0), ty, "", 0),
        )
        _write_dump(
            embed,
            name,
            dumps.SVHN_TEST,
            _center_shard(_embed(ty, 8, 1.0, 1.5), ty, "", 0),
        )
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
        "eval.num_randomizations=2",
        "--set",
        "eval.cv_folds=3",
        "--set",
        "eval.num_bootstrap=20",
    ]
    app = _load_app()
    for name in ("toy_lam0.05_s0", "toy_lam0.05_s1", "toy_lam0.1_s0"):
        assert app.main([*base, "--run-name", name, "--shift-unit"]) == 0
        assert (runs / name / "shift_cells.npz").is_file()
        assert (runs / name / "shift_ood.npz").is_file()
    (runs / "toy_lam0.1_s0" / "shift_ood.npz").unlink()  # reported, not dropped
    argv = [*base, "--shift", "toy", "--note", "idea/shift.md", "--no-figures"]
    assert app.main(argv) == 0
    text = (notes / "idea" / "shift_lam0.05.md").read_text()
    for name in (
        "shift_inputs",
        "shift_context",
        "shift_corruption",
        "shift_correlation",
        "shift_cells",
        "ood_scores",
        "ood_differences",
        "ood_oracle",
    ):
        assert f"BEGIN GENERATED: analyze.{name}" in text, name
    other = (notes / "idea" / "shift_lam0.1.md").read_text()
    assert (
        "shift_ood.npz" in other and "BEGIN GENERATED: analyze.ood_scores" not in other
    )
    assert app.main(argv) == 0
    assert (notes / "idea" / "shift_lam0.05.md").read_text() == text  # idempotent
