"""Radius view-signal study: geometry, oracle radius, severity, aggregation."""

import pathlib

import numpy
import pytest

from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import randomization
from lejepa_norm.eval import shards
from lejepa_norm.eval import view_signal
from lejepa_norm.eval import view_signal_stats as stats

D, CLASSES, VIEWS = 8, 3, 4
KNN = randomization.KNN_EUCLIDEAN
N_BANK, N_TEST = 150, 90


def _directions(labels: numpy.ndarray, rng: numpy.random.Generator) -> numpy.ndarray:
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    return features.direction(0.7 * rng.normal(size=(labels.size, D)) + centers[labels])


def _views(
    u: numpy.ndarray, log_r: numpy.ndarray, rng: numpy.random.Generator
) -> numpy.ndarray:
    """`(N, 1 + VIEWS, D)`: the center view and noisy augmented views."""
    out = [u * numpy.exp(log_r)[:, None]]
    for _ in range(VIEWS):
        v = features.direction(u + 0.3 * rng.normal(size=u.shape))
        out.append(v * numpy.exp(log_r + 0.1 * rng.normal(size=log_r.size))[:, None])
    return numpy.stack(out, axis=1).astype(numpy.float32)


def _shard(
    z: numpy.ndarray, labels: numpy.ndarray, corruption: str, severity: int
) -> shards.ViewShard:
    n = labels.size
    return shards.ViewShard(
        z=z,
        image_id=numpy.arange(n, dtype=numpy.int64),
        label=labels.astype(numpy.int64),
        corruption=numpy.array([corruption] * n),
        severity=numpy.full(n, severity, dtype=numpy.int8),
    )


def _world() -> tuple[numpy.ndarray, numpy.ndarray, shards.ViewShard, shards.ViewShard]:
    """`rescale` multiplies the radius by 0.05 (purely radial displacement), `jitter`
    adds direction noise of growing size and keeps the radius."""
    rng = numpy.random.default_rng(0)
    by = rng.integers(0, CLASSES, N_BANK)
    ty = rng.integers(0, CLASSES, N_TEST)
    bank = _directions(by, rng) * numpy.exp(0.3 * rng.normal(size=N_BANK))[:, None]
    u_test = _directions(ty, rng)
    log_r = 0.3 * rng.normal(size=N_TEST)
    clean = _shard(_views(u_test, log_r, rng), ty, "", 0)
    parts: list[shards.ViewShard] = []
    for severity in (1, 2):
        parts.append(
            _shard(
                _views(u_test, log_r + numpy.log(0.05), rng), ty, "rescale", severity
            )
        )
        noisy = features.direction(
            u_test + 0.5 * severity * rng.normal(size=u_test.shape)
        )
        parts.append(_shard(_views(noisy, log_r, rng), ty, "jitter", severity))
    corrupted = shards.ViewShard(
        z=numpy.concatenate([p.z for p in parts]),
        image_id=numpy.concatenate([p.image_id for p in parts]),
        label=numpy.concatenate([p.label for p in parts]),
        corruption=numpy.concatenate([p.corruption for p in parts]),
        severity=numpy.concatenate([p.severity for p in parts]),
    )
    return bank, by, clean, corrupted


def _units() -> list[view_signal.ViewSignalUnit]:
    bank, by, clean, corrupted = _world()
    units: list[view_signal.ViewSignalUnit] = []
    for seed in (0, 1):
        fields = view_signal.evaluate_unit(
            bank, by, clean, corrupted, k=5, cv_folds=3, seed=seed
        )
        units.append(view_signal.ViewSignalUnit("toy", 0.05, seed, *fields))
    return units


@pytest.fixture(scope="module")
def units() -> list[view_signal.ViewSignalUnit]:
    return _units()


def test_displacement_geometry_separates_radial_from_tangential(
    units: list[view_signal.ViewSignalUnit],
) -> None:
    rows = {
        (r["level"], r["lambda"]): r
        for r in stats.geometry_rows(units, 50, numpy.random.default_rng(0))
    }
    rescale = float(rows["rescale", 0.05]["radial share F_rad"])
    jitter = float(rows["jitter", 0.05]["radial share F_rad"])
    null = float(
        rows["null: clean image -> other clean image", 0.05]["radial share F_rad"]
    )
    assert rescale == pytest.approx(1.0, abs=1e-6)  # x = c / 20: purely radial
    assert jitter < 0.5 and 0.0 < null < rescale
    assert float(rows["rescale", 0.05]["median r_x / r_c"]) == pytest.approx(0.05)


def test_oracle_radius_recovers_a_radius_shift_but_not_direction_noise(
    units: list[view_signal.ViewSignalUnit],
) -> None:
    comparisons = stats.compare_accuracy(
        units,
        stats.ORACLE_CONTRASTS,
        False,
        0.005,
        50,
        numpy.random.default_rng(0),
    )
    index = {(c.reader, c.level, c.contrast): c for c in comparisons}
    # readers x (2 corruptions + 2 severities + overall) x 4 contrasts
    assert len(comparisons) == len(view_signal.READERS) * 5 * 4
    gain = index[KNN, "rescale", "oracle radius - baseline"]
    assert gain.difference > 0.05 and gain.verdict == randomization.DIFFERENT
    # Here the shifted radius is the whole damage: the clean direction with the
    # corrupted radius is as bad as the corrupted image, and with the clean radius
    # (jitter keeps it) the clean direction is the clean twin.
    assert index[KNN, "rescale", "oracle direction - clean twin"].difference < -0.2
    assert index[KNN, "jitter", "oracle direction - clean twin"].difference == 0.0
    # Restoring the radius cannot undo direction noise.
    noise = index[KNN, "jitter", "oracle radius - baseline"]
    assert abs(noise.difference) < 0.05
    recovery = {
        (r["level"], r["reader"]): float(str(r["recovered by oracle radius"]))
        for r in stats.recovery_rows(units, 50, numpy.random.default_rng(0))
    }
    assert recovery[("rescale", KNN)] > 0.8
    assert set(stats.figures(stats.geometry_cell_rows(units), comparisons, 0.05)) == {
        "radial_share",
        "oracle_radius",
    }


def test_aggregation_schemes_and_comparisons(
    units: list[view_signal.ViewSignalUnit],
) -> None:
    r = numpy.array([[1.0, 2.0, 4.0], [3.0, 1.0, 1.5]])
    weights = view_signal.aggregation_weights(r, numpy.zeros_like(r), 1.0)
    numpy.testing.assert_array_equal(weights[view_signal.AGG_TOP].sum(axis=1), [1, 1])
    assert weights[view_signal.AGG_TOP][0].argmax() == 2
    assert weights[view_signal.AGG_BOTTOM][1].argmax() == 1
    numpy.testing.assert_allclose(
        view_signal.effective_number(weights[view_signal.AGG_MEAN]), [3.0, 3.0]
    )
    assert units[0].effective_views[view_signal.AGG_TOP] == 1.0
    assert 1.0 < units[0].effective_views[view_signal.AGG_RADIUS] <= VIEWS
    # Weights scaled by a constant leave the aggregated direction unchanged.
    u = features.direction(numpy.random.default_rng(1).normal(size=(2, 3, D)))
    numpy.testing.assert_allclose(
        view_signal.aggregate(u, r), view_signal.aggregate(u, 5 * r)
    )

    comparisons = stats.compare_accuracy(
        units, stats.AGGREGATION_CONTRASTS, True, 0.005, 50, numpy.random.default_rng(0)
    )
    levels = {c.level for c in comparisons}
    assert stats.CLEAN_TEST in levels and stats.OVERALL in levels
    assert len(comparisons) == len(view_signal.READERS) * 6 * len(
        stats.AGGREGATION_CONTRASTS
    )
    for c in comparisons:
        assert c.ci95[0] <= c.difference <= c.ci95[1] or c.ci95[0] == c.ci95[1]
    assert stats.effective_view_rows(units) and stats.accuracy_summary_rows(units)


def test_severity_models_and_unit_round_trip(
    units: list[view_signal.ViewSignalUnit], tmp_path: pathlib.Path
) -> None:
    rows = stats.severity_rows(units, 30, numpy.random.default_rng(0))
    by_corruption = {r["corruption"]: r for r in rows}
    assert set(by_corruption) == {"jitter", "rescale", "mean over corruptions"}
    # A pure radius shift is read off the radius residual; the direction cannot.
    res = str(by_corruption["rescale"]["Spearman(res, severity), signed"])
    assert abs(float(res.split()[0])) > 0.5  # negative: the radius shrinks
    radius = float(
        str(by_corruption["rescale"]["held-out Spearman: radius"]).split()[0]
    )
    direction = float(
        str(by_corruption["rescale"]["held-out Spearman: direction"]).split()[0]
    )
    assert radius > direction + 0.2
    assert by_corruption["rescale"]["verdict"] in {
        stats.ADDS,
        stats.DOES_NOT_ADD,
        stats.WORSE,
        stats.INCONCLUSIVE,
    }
    assert stats.severity_verdict((-0.005, 0.005), (-0.006, 0.006), 0.01) == (
        stats.DOES_NOT_ADD
    )
    assert stats.severity_verdict((0.02, 0.05), (0.015, 0.06), 0.01) == stats.ADDS

    view_signal.save_unit(tmp_path / "u.npz", units[0])
    back = view_signal.load_unit(tmp_path / "u.npz")
    assert back.name == "toy" and back.lam == 0.05 and back.seed == 0
    assert back.correct.keys() == units[0].correct.keys()
    key = (KNN, view_signal.ORACLE_RADIUS, "rescale", 1)
    numpy.testing.assert_allclose(back.correct[key], units[0].correct[key])
    numpy.testing.assert_allclose(
        back.severity_rows["rescale"].res,
        units[0].severity_rows["rescale"].res,
        atol=1e-5,
    )
    assert back.residual_sd == pytest.approx(units[0].residual_sd)


def test_view_shard_loading_drops_h_and_keeps_every_view(
    tmp_path: pathlib.Path,
) -> None:
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
    loaded = dumps.load_view_set(out)
    assert loaded is not None and loaded.z.shape == (12, 3, 4)
    numpy.testing.assert_array_equal(loaded.z[:6], full.z)
    assert dumps.load_view_set(tmp_path / "nothing") is None
