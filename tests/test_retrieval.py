"""Retrieval under radius conditions: metrics are right, radius cases behave."""

import importlib.util
import pathlib
import types

import numpy
import pytest
import torch

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import randomization
from lejepa_norm.eval import retrieval
from lejepa_norm.eval import shards

REPO = pathlib.Path(__file__).parent.parent
D, CLASSES, VIEWS = 8, 3, 5  # VIEWS = center + 4 augmented


def _views(
    seed: int,
    n: int,
    class_in_direction: bool,
    class_in_radius: bool,
    noise: float,
    jitter: float = 0.05,
    view_noise: float = 0.002,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """`(n, VIEWS, D)` embeddings and labels.

    Each image has a latent direction (class-clustered or not) and an image-level
    radius (class-dependent, plus log-normal noise of sd `noise`); views jitter
    both slightly.
    """
    rng = numpy.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, n)
    centers = numpy.random.default_rng(99).normal(size=(CLASSES, D))
    base = rng.normal(size=(n, 1, D))
    if class_in_direction:
        base = 0.4 * base + 3.0 * centers[labels][:, None]
    u = features.direction(base + jitter * rng.normal(size=(n, VIEWS, D)))
    log_r = noise * rng.normal(size=(n, 1)) + view_noise * rng.normal(size=(n, VIEWS))
    if class_in_radius:
        log_r = log_r + 3.0 * labels[:, None]
    return u * numpy.exp(log_r)[..., None], labels


def _mean(
    values: dict[retrieval.Key, numpy.ndarray], task: str, metric: str, cond: str
) -> float:
    return float(values[task, metric, cond].mean())


def test_class_metrics_match_brute_force() -> None:
    x, y = _views(0, 60, True, False, 0.0)
    q = x[:, 0]
    out = retrieval.class_retrieval(q, y, q, y, exclude_self=True, chunk=7)
    for i in (0, 13, 59):
        dist: list[float] = numpy.square(q - q[i]).sum(axis=1).tolist()
        order = [j for j in sorted(range(len(dist)), key=dist.__getitem__) if j != i]
        rel = [float(y[j] == y[i]) for j in order]
        hits = 0.0
        precisions: list[float] = []
        for rank, r in enumerate(rel, start=1):
            hits += r
            if r:
                precisions.append(hits / rank)
        assert out["mAP"][i] == pytest.approx(sum(precisions) / len(precisions))
        for k in retrieval.KS:
            assert out[f"P@{k}"][i] == pytest.approx(sum(rel[:k]) / k)
    # A separate gallery (no self exclusion) works the same way.
    other = retrieval.class_retrieval(q[:20], y[:20], q[20:], y[20:], False, chunk=3)
    assert other["mAP"].shape == (20,)


def test_instance_recall_finds_the_other_view_by_rank() -> None:
    rng = numpy.random.default_rng(1)
    a = rng.normal(size=(50, D))
    b = a + 0.01 * rng.normal(size=a.shape)
    out = retrieval.instance_retrieval(a, b, chunk=9)
    assert out["R@1"].mean() == 1.0 and out["R@5"].mean() == 1.0
    # A gallery that swaps two images puts those targets off rank 1.
    swapped = b.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    out = retrieval.instance_retrieval(a, swapped)
    assert out["R@1"][:2].sum() == 0 and out["R@1"][2:].all()


def test_transforms_keep_directions_and_permute_whole_images() -> None:
    x, _ = _views(2, 40, True, True, 0.3)
    pred = 0.5 * features.log_radius(x)
    rng = numpy.random.default_rng(3)
    for cond in retrieval.INSTANCE_CONDITIONS:
        out = retrieval.transform_views(x, cond, pred, rng)
        numpy.testing.assert_allclose(
            features.direction(out), features.direction(x), atol=1e-9
        )
    out = retrieval.transform_views(x, retrieval.PERMUTE, pred, rng)
    assert not numpy.allclose(features.radius(out), features.radius(x))
    # Whole images move: the multiset of per-image log-radius rows is unchanged.
    rows_out = numpy.sort(features.log_radius(out), axis=0)
    rows_in = numpy.sort(features.log_radius(x), axis=0)
    numpy.testing.assert_allclose(rows_out, rows_in)
    unit = retrieval.transform_views(x, retrieval.CONSTANT, pred, rng)
    numpy.testing.assert_allclose(features.radius(unit), 1.0)


def _run(
    class_in_direction: bool,
    class_in_radius: bool,
    noise: float,
    jitter: float = 0.05,
    view_noise: float = 0.002,
) -> dict[retrieval.Key, numpy.ndarray]:
    args = (class_in_direction, class_in_radius, noise, jitter, view_noise)
    train, ytr = _views(10, 150, *args)
    test, yte = _views(11, 150, *args)
    values, _ = retrieval.evaluate(
        train, ytr, test, yte, num_repeats=2, cv_folds=3, seed=0, chunk=40
    )
    return values


def test_radius_noise_hurts_euclidean_and_constant_radius_removes_it() -> None:
    """Class in the direction only; heavy image-level radius noise hurts class
    retrieval and unit norm removes the damage. For instance retrieval a *stable*
    image-level radius is a key (constant radius loses it), while radius noise
    that differs between views is a nuisance (constant radius helps)."""
    v = _run(True, False, 1.0, jitter=0.5)
    for task in (retrieval.CLASS_TEST, retrieval.CLASS_TRAIN):
        base = _mean(v, task, "mAP", retrieval.BASELINE)
        assert _mean(v, task, "mAP", retrieval.CONSTANT) > base + 0.05, task
    stable_base = _mean(v, retrieval.INSTANCE, "R@1", retrieval.BASELINE)
    stable_const = _mean(v, retrieval.INSTANCE, "R@1", retrieval.CONSTANT)
    assert stable_base > stable_const + 0.05
    # Permuting whole images keeps the shared radius a key; permuting per view
    # destroys it, so the per-view permutation is worse for instance retrieval.
    keep = _mean(v, retrieval.INSTANCE, "R@1", retrieval.PERMUTE)
    per_view = _mean(v, retrieval.INSTANCE, "R@1", retrieval.PERMUTE_VIEW)
    assert keep > per_view + 0.05
    unstable = _run(True, False, 0.0, jitter=0.5, view_noise=1.0)
    base = _mean(unstable, retrieval.INSTANCE, "R@1", retrieval.BASELINE)
    const = _mean(unstable, retrieval.INSTANCE, "R@1", retrieval.CONSTANT)
    assert const > base + 0.05


def test_constant_radius_changes_nothing_when_radius_is_constant() -> None:
    v = _run(True, False, 0.0)  # radius constant up to 0.2% view jitter
    for (task, metric, cond), values in v.items():
        if cond == retrieval.CONSTANT:
            base = v[task, metric, retrieval.BASELINE]
            assert abs(values.mean() - base.mean()) < 0.02, (task, metric)
            # Permuting a (near-)constant radius is a no-op as well.
            perm = v[task, metric, retrieval.PERMUTE]
            assert abs(perm.mean() - base.mean()) < 0.02, (task, metric)


def test_radius_only_class_signal_is_destroyed_by_every_condition() -> None:
    v = _run(False, True, 0.0)
    chance = 1 / CLASSES
    assert _mean(v, retrieval.CLASS_TEST, "P@1", retrieval.BASELINE) > 0.95
    for cond in retrieval.CLASS_CONDITIONS[1:]:
        for task in (retrieval.CLASS_TEST, retrieval.CLASS_TRAIN):
            assert _mean(v, task, "P@1", cond) < chance + 0.15, (task, cond)


def test_compare_certifies_equivalence_and_detects_a_difference() -> None:
    def fake(seed: int, lam: float, drop: float) -> retrieval.RunResult:
        rng = numpy.random.default_rng(seed + int(100 * lam))
        n = 500
        values: dict[retrieval.Key, numpy.ndarray] = {}
        for task in retrieval.TASKS:
            base = rng.random(n)
            for metric in retrieval.task_metrics(task):
                for cond in retrieval.task_conditions(task):
                    shift = drop if cond == retrieval.PERMUTE else 0.0
                    values[task, metric, cond] = base - shift
        return retrieval.RunResult(f"r{seed}", lam, seed, values, 0.5)

    results = [fake(s, 0.05, 0.1) for s in range(3)] + [
        fake(s, 0.1, 0.0) for s in range(3)
    ]
    comps = retrieval.compare(results, 0.005, 100, numpy.random.default_rng(0))
    key = {(c.lam, c.task, c.metric, c.condition): c for c in comps}
    hit = key[0.05, retrieval.INSTANCE, "R@1", retrieval.PERMUTE]
    assert hit.verdict == randomization.DIFFERENT
    assert hit.difference == pytest.approx(-0.1)
    none = key[0.1, retrieval.CLASS_TEST, "mAP", retrieval.PERMUTE]
    assert none.verdict == randomization.EQUIVALENT
    assert len(retrieval.comparison_rows(comps)) == len(comps)
    assert len(retrieval.value_rows(results, retrieval.INSTANCE)) == 2 * 5


def test_result_roundtrip(tmp_path: pathlib.Path) -> None:
    values = {(retrieval.INSTANCE, "R@1", retrieval.BASELINE): numpy.arange(4.0)}
    result = retrieval.RunResult("run_a", 0.05, 2, values, 0.25)
    loaded = retrieval.load_result(retrieval.save_result(tmp_path, result))
    assert (loaded.name, loaded.lam, loaded.seed, loaded.r2_test) == (
        "run_a",
        0.05,
        2,
        0.25,
    )
    numpy.testing.assert_array_equal(
        loaded.values[retrieval.INSTANCE, "R@1", retrieval.BASELINE], numpy.arange(4.0)
    )


def test_gpu_and_cpu_agree_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    x, y = _views(0, 60, True, False, 0.3)
    cpu = retrieval.class_retrieval(x[:, 0], y, x[:, 0], y, True)
    gpu = retrieval.class_retrieval(
        x[:, 0], y, x[:, 0], y, True, device=torch.device("cuda")
    )
    numpy.testing.assert_allclose(cpu["mAP"], gpu["mAP"])


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shard(labels: numpy.ndarray, seed: int) -> shards.EmbeddingShard:
    z, _ = _views(seed, labels.size, True, False, 0.3)
    z32 = z.astype(numpy.float32)
    n = labels.size
    return shards.EmbeddingShard(
        z=z32,
        h=z32.copy(),
        image_id=numpy.arange(n, dtype=numpy.int64),
        label=labels.astype(numpy.int64),
        corruption=numpy.array([""] * n),
        severity=numpy.zeros(n, dtype=numpy.int8),
    )


def test_app_runs_one_job_per_run_then_aggregates(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, embed, notes = tmp_path / "runs", tmp_path / "embed", tmp_path / "notes"
    labels = numpy.random.default_rng(0).integers(0, CLASSES, 90)
    names = (("toy_lam0.05_s0", 0.05, 0), ("toy_lam0.05_s1", 0.05, 1))
    for name, lam, seed in names:
        cfg = config.ExperimentConfig(seed=seed)
        cfg.loss.lambda_ = lam
        config.save_resolved(cfg, runs / name)
        for eval_set, offset in ((dumps.STL10_TRAIN, 0), (dumps.STL10_TEST, 100)):
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
    base = [
        "--config",
        str(REPO / "configs" / "base.yaml"),
        "--retrieval",
        "toy",
        "--set",
        "eval.num_randomizations=2",
        "--set",
        "eval.num_bootstrap=20",
        "--set",
        "eval.cv_folds=3",
    ]
    app = _load_app()
    for name, _, _ in names:
        assert app.main([*base, "--retrieval-run", name]) == 0
        assert (runs / "radius_usage_retrieval" / f"{name}.npz").exists()
    argv = [*base, "--note", "idea/retrieval.md", "--no-figures"]
    assert app.main(argv) == 0
    text = (notes / "idea" / "retrieval.md").read_text()
    for name in ("comparison", "values", "r2"):
        assert f"BEGIN GENERATED: analyze.retrieval_{name}" in text
    rows = [ln for ln in text.splitlines() if ln.startswith("| 0.05 |")]
    verdict_rows = [r for r in rows if "±" not in r]
    # 2 class tasks x 4 metrics x 3 conditions + instance 2 metrics x 4 conditions.
    assert len(verdict_rows) == 2 * 4 * 3 + 2 * 4
    assert app.main(argv) == 0
    assert (notes / "idea" / "retrieval.md").read_text() == text  # idempotent
