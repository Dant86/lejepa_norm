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


def _evaluate(label_in_direction: bool, label_in_radius: bool) -> dict[str, float]:
    bank, by = _embeddings(10, label_in_direction, label_in_radius)
    query, qy = _embeddings(11, label_in_direction, label_in_radius)
    evaluation = randomization.evaluate(
        bank, by, query, qy, k=5, num_repeats=3, cv_folds=3, seed=0
    )
    assert [r["condition"] for r in evaluation.rows] == list(randomization.CONDITIONS)
    assert [r["repeats"] for r in evaluation.rows] == [1, 3, 3, 1, 1]
    return {str(r["condition"]): float(r["accuracy"]) for r in evaluation.rows}


def test_radius_only_labels_are_destroyed_by_every_randomization() -> None:
    acc = _evaluate(label_in_direction=False, label_in_radius=True)
    assert acc[randomization.BASELINE] > 0.9
    chance = 1 / CLASSES
    for condition in (randomization.PERMUTE, randomization.CONSTANT):
        assert acc[condition] < chance + 0.15, condition
    # The radius is not predictable from direction here, so the residual carries
    # the whole class signal and shuffling it also destroys it.
    assert acc[randomization.RESIDUAL] < chance + 0.15


def test_direction_only_labels_survive_constant_radius_and_match_cosine() -> None:
    acc = _evaluate(label_in_direction=True, label_in_radius=False)
    assert acc[randomization.BASELINE] > 0.9
    assert acc[randomization.CONSTANT] > 0.9
    assert acc[randomization.CONSTANT] == pytest.approx(acc[randomization.COSINE])


def test_radius_predictable_from_direction_survives_the_residual_shuffle() -> None:
    """Radius = class signal fully carried by the direction: keeping the predicted
    part keeps the class signal, a full permutation of radii does not hurt more
    than the metric noise it adds."""

    def make(seed: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        r = numpy.random.default_rng(seed)
        labels = r.integers(0, CLASSES, N)
        centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
        u = features.direction(0.3 * r.normal(size=(N, D)) + 3.0 * centers[labels])
        # The radius is a smooth function of the direction plus little noise.
        log_r = 1.5 * u[:, 0] + 0.02 * r.normal(size=N)
        return u * numpy.exp(log_r)[:, None], labels

    (bank, by), (query, qy) = make(20), make(21)
    evaluation = randomization.evaluate(
        bank, by, query, qy, k=5, num_repeats=3, cv_folds=3, seed=0
    )
    assert float(evaluation.model[0]["bank (out-of-fold)"]) > 0.8
    assert float(evaluation.model[0]["queries (held out)"]) > 0.8
    acc = {str(r["condition"]): float(r["accuracy"]) for r in evaluation.rows}
    assert acc[randomization.RESIDUAL] >= acc[randomization.PERMUTE] - 0.02
    assert acc[randomization.RESIDUAL] > 0.8


def test_aggregate_and_series_group_by_lambda_and_seed() -> None:
    def rows(base: float, other: float) -> list[randomization.Row]:
        return [
            {
                "condition": c,
                "accuracy": base if c == randomization.BASELINE else other,
                "sd over repeats": 0.01,
            }
            for c in randomization.CONDITIONS
        ]

    results = [
        randomization.RunResult("a", 0.05, 0, randomization.CENTER, rows(0.8, 0.7)),
        randomization.RunResult("b", 0.05, 1, randomization.CENTER, rows(0.8, 0.6)),
        randomization.RunResult("c", 0.1, 0, randomization.CENTER, rows(0.9, 0.9)),
    ]
    table = randomization.aggregate(results)
    permute = [r for r in table if r["condition"] == randomization.PERMUTE]
    assert [r["lambda"] for r in permute] == [0.05, 0.1]
    assert permute[0]["accuracy (mean ± std over seeds)"] == "0.6500 ± 0.0707"
    assert permute[0]["change vs baseline"] == "-0.1500 ± 0.0707"
    lams, means, _ = randomization.accuracy_by_lambda(results, randomization.CENTER)[
        randomization.PERMUTE
    ]
    assert lams == [0.05, 0.1] and means == pytest.approx([0.65, 0.9])


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
    ]
    app = _load_app()
    assert app.main(argv) == 0
    text = (notes / "idea" / "knn.md").read_text()
    for name in ("summary", "runs", "model", "skipped"):
        assert f"BEGIN GENERATED: analyze.randomization_{name}" in text
    lines = text.splitlines()
    summary = [ln for ln in lines if ln.startswith("| center view | 0.05 |")]
    assert len(summary) == len(randomization.CONDITIONS)
    assert "| 2 |" in next(ln for ln in summary if "baseline" in ln)  # two seeds
    assert not any(ln.startswith("| center view | 0.1 |") for ln in lines)
    assert "| toy_lam0.1_s0 | stl10_train |" in text
    assert app.main(argv) == 0
    assert (notes / "idea" / "knn.md").read_text() == text  # idempotent
