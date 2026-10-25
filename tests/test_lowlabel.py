"""Low-label E1: subsets, hyperparameter selection, the workflow and the app mode."""

import functools
import importlib.util
import pathlib
import types

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import lowlabel
from lejepa_norm.eval import randomization
from lejepa_norm.eval import shards

REPO = pathlib.Path(__file__).parent.parent
CLASSES, D, PER_CLASS = 3, 8, 30


def _data(seed: int, in_radius: bool) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Balanced `(3 * 60, D)` embeddings; the class sits in the direction, or (with
    `in_radius`) only in the radius."""
    rng = numpy.random.default_rng(seed)
    labels = numpy.tile(numpy.arange(CLASSES), PER_CLASS)
    rng.shuffle(labels)
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    direction = 0.3 * rng.normal(size=(labels.size, D))
    direction[:, 0] += 1.0
    log_r = 0.1 * rng.normal(size=labels.size)
    if in_radius:
        log_r = log_r + 1.5 * labels
    else:
        direction = direction + 3.0 * centers[labels]
    return features.direction(direction) * numpy.exp(log_r)[:, None], labels


def test_subsets_are_balanced_deterministic_and_distinct() -> None:
    _, y = _data(0, False)
    a = lowlabel.subset_indices(y, 5, 4)
    b = lowlabel.subset_indices(y, 5, 4)
    assert len(a) == 4
    for first, second in zip(a, b, strict=True):
        numpy.testing.assert_array_equal(first, second)
        assert numpy.bincount(y[first]).tolist() == [5] * CLASSES
    assert len({tuple(s) for s in a}) == 4
    # A budget that takes the whole split has one (identical) subset.
    (whole,) = lowlabel.subset_indices(y, PER_CLASS, 4)
    assert whole.size == y.size


def test_cross_validation_picks_the_better_candidate_and_is_stratified() -> None:
    x, y = _data(1, False)
    folds = lowlabel.stratified_folds(y, 5, numpy.random.default_rng(0))
    for f in range(5):
        assert numpy.bincount(y[folds == f]).tolist() == [6] * CLASSES

    def predict(
        value: float, xf: numpy.ndarray, yf: numpy.ndarray, xv: numpy.ndarray
    ) -> numpy.ndarray:
        if value == 0.0:  # a useless reader
            return numpy.zeros(xv.shape[0], dtype=numpy.int64)
        means = numpy.stack([xf[yf == c].mean(axis=0) for c in range(CLASSES)])
        dist = numpy.linalg.norm(xv[:, None] - means[None], axis=-1)
        return dist.argmin(axis=1).astype(numpy.int64)

    assert lowlabel.select_hyperparameter(x, y, (0.0, 1.0), predict, 3, 0) == 1.0
    assert lowlabel.select_hyperparameter(x, y, (1.0, 0.0), predict, 3, 0) == 1.0
    # Ties go to the first candidate.
    assert lowlabel.select_hyperparameter(x, y, (0.0, 0.0), predict, 3, 0) == 0.0


@functools.cache
def _evaluate(in_radius: bool) -> list[lowlabel.BudgetResult]:
    bank, by = _data(10, in_radius)
    query, qy = _data(11, in_radius)
    return lowlabel.evaluate_run(
        bank,
        by,
        query,
        qy,
        k=5,
        num_repeats=2,
        cv_folds=3,
        seed=0,
        per_class_budgets=(5, 10, PER_CLASS),
        num_subsets=3,
        readers=(randomization.KNN_EUCLIDEAN, randomization.LINEAR),
    )


def _acc(result: lowlabel.BudgetResult, reader: str, protocol: str, cond: str) -> float:
    return float(result.cells[reader, protocol, cond].correct.mean())


def test_run_shapes_and_label_dependence_of_accuracy() -> None:
    results = _evaluate(False)
    assert [r.num_subsets for r in results] == [3, 3, 1]
    assert [r.num_labels for r in results] == [15, 30, 90]
    knn = randomization.KNN_EUCLIDEAN
    for result in results:
        cell = result.cells[knn, randomization.RETRAIN, randomization.BASELINE]
        assert len(cell.repeat_accuracy) == result.num_subsets
        # The per-image correctness is the mean of the per-subset accuracies.
        assert float(cell.correct.mean()) == pytest.approx(
            sum(cell.repeat_accuracy) / len(cell.repeat_accuracy)
        )
    # The linear probe's C is selected for each (subset, reader) it ran for.
    chosen = [h for h in results[0].hyper if h["reader"] == randomization.LINEAR]
    assert len(chosen) == 3
    assert all(h["hyperparameter"] in lowlabel.LINEAR_CS for h in chosen)
    assert _acc(results[-1], knn, randomization.RETRAIN, "baseline") > 0.9


def test_radius_only_labels_are_destroyed_at_every_budget() -> None:
    for result in _evaluate(True):
        for reader in (randomization.KNN_EUCLIDEAN, randomization.LINEAR):
            base = _acc(result, reader, randomization.RETRAIN, "baseline")
            permuted = _acc(
                result, reader, randomization.RETRAIN, randomization.PERMUTE
            )
            assert base > 0.6 and permuted < base - 0.15, (result.per_class, reader)


def test_shard_round_trip_and_aggregation(tmp_path: pathlib.Path) -> None:
    results = _evaluate(True)
    lowlabel.save_run(tmp_path / "a" / "run.npz", results)
    loaded = lowlabel.load_run(tmp_path / "a" / "run.npz")
    assert [r.per_class for r in loaded] == [5, 10, PER_CLASS]
    for before, after in zip(results, loaded, strict=True):
        assert before.hyper == after.hyper and before.cells.keys() == after.cells.keys()
        for key, cell in before.cells.items():
            numpy.testing.assert_array_equal(cell.correct, after.cells[key].correct)
            assert cell.repeat_accuracy == after.cells[key].repeat_accuracy
    runs = [
        lowlabel.RunBudgets(f"r{s}", 0.05, s, loaded) for s in range(2)
    ]  # two "seeds" with identical results
    comparisons = lowlabel.compare(
        runs,
        0.005,
        50,
        numpy.random.default_rng(0),
        readers=(randomization.KNN_EUCLIDEAN, randomization.LINEAR),
    )
    # budgets x readers x protocols x non-baseline conditions
    assert len(comparisons) == 3 * 2 * 2 * 3
    spread = {c.per_class: c.subset_se for c in comparisons}
    assert spread[PER_CLASS] == 0.0  # one subset: no subset spread
    assert any(c.subset_se > 0 for c in comparisons if c.num_subsets > 1)
    # The subset SE can only make a verdict more cautious, never less.
    for c in comparisons:
        if c.verdict_with_subsets == randomization.EQUIVALENT:
            assert c.comparison.verdict == randomization.EQUIVALENT
    assert lowlabel.comparison_rows(comparisons)[0]["subsets"] == 3


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app_lowlabel", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shard(seed: int) -> shards.EmbeddingShard:
    z, labels = _data(seed, False)
    views = numpy.repeat(z[:, None, :], 2, axis=1).astype(numpy.float32)
    n = labels.size
    return shards.EmbeddingShard(
        z=views,
        h=views.copy(),
        image_id=numpy.arange(n, dtype=numpy.int64),
        label=labels.astype(numpy.int64),
        corruption=numpy.array([""] * n),
        severity=numpy.zeros(n, dtype=numpy.int8),
    )


def test_analyze_lowlabel_mode_computes_shards_then_aggregates_per_lambda(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, embed, notes = tmp_path / "runs", tmp_path / "embed", tmp_path / "notes"
    names = (("toy_lam0.05_s0", 0.05, 0), ("toy_lam0.1_s0", 0.1, 0))
    for name, lam, seed in names:
        cfg = config.ExperimentConfig(seed=seed)
        cfg.loss.lambda_ = lam
        config.save_resolved(cfg, runs / name)
        for eval_set, data_seed in ((dumps.STL10_TRAIN, 3), (dumps.STL10_TEST, 4)):
            out = dumps.set_dir(embed, name, eval_set)
            out.mkdir(parents=True)
            shards.save_shard(out / "part-00000.npz", _shard(data_seed + seed))
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
        "--lowlabel",
        "toy",
        "--set",
        "eval.knn_k=5",
        "--set",
        "eval.num_randomizations=2",
        "--set",
        "eval.num_bootstrap=20",
        "--set",
        "eval.cv_folds=3",
        "--set",
        "eval.lowlabel_subsets=2",
    ]
    app = _load_app()
    for name, _, _ in names:
        assert app.main([*base, "--lowlabel-run", name]) == 0
        assert (runs / "lowlabel" / "default" / f"{name}.npz").exists()
    note = ["--note", "idea/lowlabel_lam0.05.md", "--no-figures"]
    assert app.main([*base, *note, "--lowlabel-lambda", "0.05"]) == 0
    text = (notes / "idea" / "lowlabel_lam0.05.md").read_text()
    for name in ("comparison", "accuracy", "hyper"):
        assert f"BEGIN GENERATED: analyze.lowlabel_{name}" in text
    lines = text.splitlines()
    verdicts = [
        ln
        for ln in lines
        if ln.startswith("| 0.05 |")
        and ln.rstrip().endswith(("equivalent |", "different |", "inconclusive |"))
    ]
    assert len(verdicts) == 4 * 3 * 2 * 3  # budgets x readers x protocols x conditions
    assert not any(ln.startswith("| 0.1 |") for ln in lines)
    assert app.main([*base, *note, "--lowlabel-lambda", "0.05"]) == 0
    assert (notes / "idea" / "lowlabel_lam0.05.md").read_text() == text
