"""Radius randomization: transformations are what they claim, the study runs."""

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
from lejepa_norm.eval import randomization
from lejepa_norm.eval import shards

REPO = pathlib.Path(__file__).parent.parent
N, D, CLASSES = 240, 8, 3


def _embeddings(
    seed: int, label_in_direction: bool, label_in_radius: bool
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """`(N, D)` embeddings and labels with the class in the direction, the radius, or
    neither."""
    rng = numpy.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, N)
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    direction = 0.3 * rng.normal(size=(N, D))
    direction[:, 0] += 1.0  # shared by all classes unless the label is added below
    if label_in_direction:
        direction = direction + 3.0 * centers[labels]
    log_r = 0.1 * rng.normal(size=N)
    if label_in_radius:
        log_r = log_r + 1.5 * labels
    return features.direction(direction) * numpy.exp(log_r)[:, None], labels


def test_permutation_keeps_directions_and_the_multiset_of_radii() -> None:
    x, _ = _embeddings(0, True, True)
    out = randomization.permute_radius(x, numpy.random.default_rng(1))
    numpy.testing.assert_allclose(features.direction(out), features.direction(x))
    numpy.testing.assert_allclose(
        numpy.sort(features.radius(out)), numpy.sort(features.radius(x))
    )
    assert not numpy.allclose(features.radius(out), features.radius(x))


def test_residual_shuffle_keeps_the_predicted_part_and_permutes_the_residual() -> None:
    x, _ = _embeddings(0, True, True)
    predicted = 0.5 * features.log_radius(x) + 0.1  # any held-out prediction
    out = randomization.residual_shuffle(x, predicted, numpy.random.default_rng(2))
    numpy.testing.assert_allclose(features.direction(out), features.direction(x))
    before = features.log_radius(x) - predicted
    after = features.log_radius(out) - predicted
    numpy.testing.assert_allclose(numpy.sort(after), numpy.sort(before), atol=1e-9)
    assert not numpy.allclose(after, before)


def test_constant_radius_is_constant_and_keeps_directions() -> None:
    x, _ = _embeddings(0, True, True)
    out = randomization.constant_radius(x, 2.5)
    numpy.testing.assert_allclose(features.radius(out), 2.5)
    numpy.testing.assert_allclose(features.direction(out), features.direction(x))


def test_euclidean_knn_matches_brute_force_and_cosine_on_unit_vectors() -> None:
    x, y = _embeddings(3, True, True)
    bank, query, by = x[:180], x[180:], y[:180]
    votes = randomization.euclidean_knn_votes(bank, by, query, 5, CLASSES, chunk=17)
    dist = numpy.linalg.norm(query[:, None] - bank[None], axis=-1)
    nearest = numpy.argsort(dist, axis=1)[:, :5]
    expected = numpy.stack([numpy.bincount(by[r], minlength=CLASSES) for r in nearest])
    numpy.testing.assert_allclose(votes, expected / 5)
    # Constant radius == cosine k-NN of the unmodified embeddings.
    unit_bank, unit_query = (randomization.constant_radius(a) for a in (bank, query))
    numpy.testing.assert_allclose(
        randomization.euclidean_knn_votes(unit_bank, by, unit_query, 5, CLASSES),
        downstream.knn_votes(bank, by, query, 5, CLASSES),
    )


def _cells(
    label_in_direction: bool, label_in_radius: bool, readers: tuple[str, ...]
) -> randomization.Evaluation:
    bank, by = _embeddings(10, label_in_direction, label_in_radius)
    query, qy = _embeddings(11, label_in_direction, label_in_radius)
    return randomization.evaluate(
        bank, by, query, qy, k=5, num_repeats=3, cv_folds=3, seed=0, readers=readers
    )


def _acc(ev: randomization.Evaluation, reader: str, protocol: str, cond: str) -> float:
    return float(ev.cells[reader, protocol, cond].correct.mean())


FAST = (randomization.KNN_EUCLIDEAN, randomization.KNN_COSINE, randomization.LINEAR)


def test_radius_only_labels_are_destroyed_by_every_randomization() -> None:
    ev = _cells(False, True, FAST)
    chance = 1 / CLASSES
    for reader in (randomization.KNN_EUCLIDEAN, randomization.LINEAR):
        for protocol in randomization.PROTOCOLS:
            assert _acc(ev, reader, protocol, randomization.BASELINE) > 0.9
            for cond in randomization.CONDITIONS[1:]:
                assert _acc(ev, reader, protocol, cond) < chance + 0.15, (
                    reader,
                    protocol,
                    cond,
                )
    # Repeats: 3 for randomized conditions, 1 for deterministic ones.
    cell = ev.cells[randomization.LINEAR, randomization.RETRAIN, randomization.PERMUTE]
    assert len(cell.repeat_accuracy) == 3
    assert (
        len(
            ev.cells[
                randomization.LINEAR, randomization.FROZEN, "baseline"
            ].repeat_accuracy
        )
        == 1
    )


def test_direction_only_labels_survive_and_cosine_is_radius_invariant() -> None:
    ev = _cells(True, False, FAST)
    for reader in FAST:
        for protocol in randomization.PROTOCOLS:
            assert _acc(ev, reader, protocol, randomization.CONSTANT) > 0.9
    cosine = randomization.KNN_COSINE
    for protocol in randomization.PROTOCOLS:
        base = ev.cells[cosine, protocol, randomization.BASELINE].correct
        for cond in randomization.CONDITIONS[1:]:
            numpy.testing.assert_array_equal(
                ev.cells[cosine, protocol, cond].correct, base
            )
    # Euclidean k-NN on unit vectors is the cosine k-NN.
    for protocol in randomization.PROTOCOLS:
        numpy.testing.assert_array_equal(
            ev.cells[
                randomization.KNN_EUCLIDEAN, protocol, randomization.CONSTANT
            ].correct,
            ev.cells[cosine, protocol, randomization.BASELINE].correct,
        )


def test_mlp_reader_runs_and_selects_a_weight_decay() -> None:
    ev = _cells(True, False, (randomization.MLP,))
    assert ev.weight_decay in randomization.MLP_WEIGHT_DECAYS
    assert _acc(ev, randomization.MLP, randomization.RETRAIN, "baseline") > 0.9


def test_radius_predictable_from_direction_survives_the_residual_shuffle() -> None:
    """Class carried by the direction, radius a smooth function of the direction:
    the held-out regression finds it and keeping the predicted part keeps accuracy."""

    def make(seed: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        r = numpy.random.default_rng(seed)
        labels = r.integers(0, CLASSES, N)
        centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
        u = features.direction(0.3 * r.normal(size=(N, D)) + 3.0 * centers[labels])
        log_r = 1.5 * u[:, 0] + 0.02 * r.normal(size=N)
        return u * numpy.exp(log_r)[:, None], labels

    (bank, by), (query, qy) = make(20), make(21)
    ev = randomization.evaluate(
        bank,
        by,
        query,
        qy,
        k=5,
        num_repeats=3,
        cv_folds=3,
        seed=0,
        readers=(randomization.KNN_EUCLIDEAN,),
    )
    assert float(ev.model[0]["bank (out-of-fold)"]) > 0.8
    assert float(ev.model[0]["test (held out)"]) > 0.8
    knn = randomization.KNN_EUCLIDEAN
    residual = _acc(ev, knn, randomization.RETRAIN, randomization.RESIDUAL)
    permuted = _acc(ev, knn, randomization.RETRAIN, randomization.PERMUTE)
    assert residual >= permuted - 0.02 and residual > 0.8


def test_verdict_follows_the_equivalence_margin() -> None:
    v = randomization.verdict
    assert v((-0.002, 0.003), (-0.003, 0.004), 0.005) == randomization.EQUIVALENT
    assert v((-0.04, -0.02), (-0.05, -0.01), 0.005) == randomization.DIFFERENT
    assert v((-0.02, 0.02), (-0.03, 0.03), 0.005) == randomization.INCONCLUSIVE


def _fake_result(
    name: str, lam: float, seed: int, drop: float
) -> randomization.RunResult:
    rng = numpy.random.default_rng(seed + int(100 * lam))
    n = 400
    base = rng.random(n) < 0.8
    cells: dict[randomization.Key, randomization.Cell] = {}
    for reader in randomization.READERS:
        for protocol in randomization.PROTOCOLS:
            for cond in randomization.CONDITIONS:
                hit = base.copy()
                if cond == randomization.PERMUTE and drop > 0:
                    hit &= rng.random(n) > drop
                cells[reader, protocol, cond] = randomization.Cell(
                    hit.astype(numpy.float64), (float(hit.mean()),)
                )
    return randomization.RunResult(
        name, lam, seed, randomization.Evaluation(cells, [], 0.01)
    )


def test_compare_detects_a_real_drop_and_certifies_no_change() -> None:
    results = [_fake_result(f"a{s}", 0.05, s, 0.2) for s in range(3)]
    results += [_fake_result(f"b{s}", 0.1, s, 0.0) for s in range(3)]
    comparisons = randomization.compare(
        results, 0.005, 200, numpy.random.default_rng(0)
    )
    assert len(comparisons) == 2 * 4 * 2 * 3
    by_key = {(c.lam, c.reader, c.protocol, c.condition): c for c in comparisons}
    knn = randomization.KNN_EUCLIDEAN
    drop = by_key[0.05, knn, randomization.RETRAIN, randomization.PERMUTE]
    assert drop.verdict == randomization.DIFFERENT and drop.difference < -0.1
    none = by_key[0.1, knn, randomization.RETRAIN, randomization.PERMUTE]
    assert none.verdict == randomization.EQUIVALENT and none.ci95 == (0.0, 0.0)
    rows = randomization.comparison_rows(comparisons)
    assert rows[0]["verdict"] in {
        randomization.EQUIVALENT,
        randomization.DIFFERENT,
        randomization.INCONCLUSIVE,
    }
    assert len(randomization.accuracy_rows(results)) == 2 * 4 * 2 * 4


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shard(labels: numpy.ndarray, seed: int) -> shards.EmbeddingShard:
    """`(n, 5, D)` views: class-clustered direction, per-view jitter, image radius."""
    rng = numpy.random.default_rng(seed)
    n = labels.size
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    base = 0.5 * rng.normal(size=(n, 1, D)) + 2.0 * centers[labels][:, None]
    views = features.direction(base + 0.1 * rng.normal(size=(n, 5, D)))
    log_r = 0.2 * rng.normal(size=(n, 1)) + 0.02 * rng.normal(size=(n, 5))
    z = (views * numpy.exp(log_r)[..., None]).astype(numpy.float32)
    return shards.EmbeddingShard(
        z=z,
        h=z.copy(),
        image_id=numpy.arange(n, dtype=numpy.int64),
        label=labels.astype(numpy.int64),
        corruption=numpy.array([""] * n),
        severity=numpy.zeros(n, dtype=numpy.int8),
    )


def test_analyze_randomize_radius_mode_writes_tables_and_lists_skipped_runs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, embed, notes = tmp_path / "runs", tmp_path / "embed", tmp_path / "notes"
    labels = numpy.random.default_rng(0).integers(0, CLASSES, 150)
    for name, lam, seed in (
        ("toy_lam0.05_s0", 0.05, 0),
        ("toy_lam0.05_s1", 0.05, 1),
        ("toy_lam0.1_s0", 0.1, 0),
    ):
        cfg = config.ExperimentConfig(seed=seed)
        cfg.loss.lambda_ = lam
        config.save_resolved(cfg, runs / name)
        for eval_set, offset in ((dumps.STL10_TRAIN, 0), (dumps.STL10_TEST, 100)):
            if name == "toy_lam0.1_s0" and eval_set == dumps.STL10_TRAIN:
                continue  # the dump this run lacks must be reported, not dropped
            out = dumps.set_dir(embed, name, eval_set)
            out.mkdir(parents=True)
            shards.save_shard(out / "part-00000.npz", _shard(labels, seed + offset))
    for var, value in (
        (paths.DATA_DIR_VAR, tmp_path / "data"),
        (paths.RUNS_DIR_VAR, runs),
        (paths.EMBED_DIR_VAR, embed),
        (paths.NOTES_DIR_VAR, notes),
        (paths.KALEIDO_CHROME_VAR, ""),
    ):
        monkeypatch.setenv(var, str(value))
    monkeypatch.chdir(tmp_path)
    argv = [
        "--config",
        str(REPO / "configs" / "base.yaml"),
        "--randomize-radius",
        "toy",
        "--note",
        "idea/knn.md",
        "--no-figures",
        "--set",
        "eval.knn_k=5",
        "--set",
        "eval.num_randomizations=2",
        "--set",
        "eval.num_bootstrap=20",
    ]
    app = _load_app()
    assert app.main(argv) == 0
    text = (notes / "idea" / "knn.md").read_text()
    for name in ("comparison", "accuracy", "runs", "model", "skipped"):
        assert f"BEGIN GENERATED: analyze.randomization_{name}" in text
    lines = text.splitlines()
    verdicts = [ln for ln in lines if ln.startswith("| 0.05 |") and "±" not in ln]
    assert len(verdicts) == 4 * 2 * 3  # readers x protocols x conditions
    assert all(
        ln.rstrip().endswith(("equivalent |", "different |", "inconclusive |"))
        for ln in verdicts
    )
    assert not any(ln.startswith("| 0.1 |") for ln in lines)
    assert "| toy_lam0.1_s0 | stl10_train |" in text
    assert app.main(argv) == 0
    assert (notes / "idea" / "knn.md").read_text() == text  # idempotent
