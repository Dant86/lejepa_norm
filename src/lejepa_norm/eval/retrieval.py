"""Is the embedding radius *used* by retrieval? Euclidean retrieval in z-space.

Companion of `randomization` (same radius conditions, same paired-bootstrap/TOST
statistics) for retrieval instead of classification. Only `z` is analyzed
(LeJEPA's product is z) and only Euclidean distance is used (cosine distance is
radius-invariant by construction; the `constant` condition *is* the normalized
embedding, so `constant` vs `baseline` is cosine vs Euclidean retrieval).

Views in the dumps: the center view (view 0) and K = 8 *global* augmentations
(`embedding.py` samples `data.view_spec.global_spec` only; there are no local
views in the dumps). Class-level retrieval uses the center views; instance-level
retrieval uses the augmented (global) views.

Tasks (`TASKS`):

* `class_test`: queries = test center views, gallery = the other test center views
  (leave-query-out), relevance = same class label. Metrics: mAP (average
  precision over the full ranking, averaged over queries) and precision@{1, 5, 10}
  (the fraction of the top k that is relevant; precision@1 is the recall@1 hit
  rate of finding a same-class item).
* `class_train`: same queries, gallery = the train center views.
* `instance`: view-to-view. For a view-index pair `(a, b)` with `a != b`, the
  query is view `a` of every test image, the gallery view `b` of every test image,
  and the target of query `i` is gallery item `i`. Metrics: recall@1 and recall@5
  (the target is among the 1 / 5 nearest gallery items; rank counts strictly
  closer items). Disjoint pairs `(0, 1), (2, 3), ...` of the augmented views are
  averaged per image, so every image's score is the fraction of pairs it was found
  in.

Conditions (the E1 conditions of `randomization`, applied to the query and gallery
embeddings): `baseline`, `permute` (radius of a random *image*, with the view index
kept, so a view pair keeps a shared, now direction-unrelated, radius), `residual`
(direction-predicted held-out log-radius kept, residual permuted across images, the
predictor is a kernel ridge fitted on the train center views) and `constant`
(unit norm). Instance retrieval adds `PERMUTE_VIEW`: every embedding takes the
radius of a random embedding of the pool, which also destroys the image-level
stability of the radius. For `class_test` and the queries of `class_train` the
same transformed test embeddings serve as query and gallery; the train gallery of
`class_train` is transformed independently.

Statistics: the per-query value of every metric under every condition is compared
with the baseline's on the same queries (paired bootstrap over queries within each
training seed, averaged over seeds, `randomization.bootstrap_seed_mean`),
with the TOST-style verdict of `randomization.verdict`. The margin is
`eval.retrieval_margin` (0.5 points of mAP / precision / recall), fixed before any
retrieval result was computed: the same size as the E1 accuracy margin and well
above the per-metric noise of 8000 queries (a per-query metric with sd <= 0.5
has a standard error <= 0.6 points), so equivalence is certifiable. Shuffle
repeats are averaged per query and are not part of the interval.
"""

import collections.abc
import dataclasses
import pathlib
import statistics

import numpy
import numpy.typing
import torch

from lejepa_norm import config
from lejepa_norm.eval import features
from lejepa_norm.eval import randomization
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
Row = dict[str, str | int | float]

BASELINE = randomization.BASELINE
PERMUTE = randomization.PERMUTE
RESIDUAL = randomization.RESIDUAL
CONSTANT = randomization.CONSTANT
PERMUTE_VIEW = "permute radius per view (instance task only)"
CLASS_CONDITIONS = (BASELINE, PERMUTE, RESIDUAL, CONSTANT)
INSTANCE_CONDITIONS = (*CLASS_CONDITIONS, PERMUTE_VIEW)

CLASS_TEST = "class_test"
CLASS_TRAIN = "class_train"
INSTANCE = "instance"
TASKS = (CLASS_TEST, CLASS_TRAIN, INSTANCE)
TASK_TITLES = {
    CLASS_TEST: "class-level, gallery = test (leave-query-out)",
    CLASS_TRAIN: "class-level, gallery = train",
    INSTANCE: "instance-level, view to view",
}
KS = (1, 5, 10)
AP = "mAP"
CLASS_METRICS = (AP, *(f"P@{k}" for k in KS))
INSTANCE_KS = (1, 5)
INSTANCE_METRICS = tuple(f"R@{k}" for k in INSTANCE_KS)
DEFAULT_CHUNK = 512

Key = tuple[str, str, str]  # (task, metric, condition)


def task_metrics(task: str) -> tuple[str, ...]:
    """Metric names of a task."""
    return INSTANCE_METRICS if task == INSTANCE else CLASS_METRICS


def task_conditions(task: str) -> tuple[str, ...]:
    """Condition names of a task."""
    return INSTANCE_CONDITIONS if task == INSTANCE else CLASS_CONDITIONS


def _sorted_relevance_metrics(
    dist: torch.Tensor, relevant: torch.Tensor
) -> dict[str, torch.Tensor]:
    """AP and precision@k of each row of `dist` `(B, G)` given `relevant` `(B, G)`."""
    order = dist.argsort(dim=1)
    rel = relevant.gather(1, order)
    hits = rel.to(torch.float64)
    ranks = torch.arange(1, rel.shape[1] + 1, device=dist.device, dtype=torch.float64)
    precision = hits.cumsum(dim=1) / ranks
    num_relevant = hits.sum(dim=1).clamp(min=1.0)
    out = {AP: (precision * hits).sum(dim=1) / num_relevant}
    for k in KS:
        out[f"P@{k}"] = hits[:, :k].mean(dim=1)
    return out


def class_retrieval(
    query: FloatArray,
    query_labels: IntArray,
    gallery: FloatArray,
    gallery_labels: IntArray,
    exclude_self: bool,
    chunk: int = DEFAULT_CHUNK,
    device: torch.device | None = None,
) -> dict[str, FloatArray]:
    """Per-query mAP and precision@k of Euclidean class-level retrieval.

    Memory is `O(chunk * G)`: queries are processed in blocks.

    Args:
        query: `(N, d)` query embeddings.
        query_labels: `(N,)` labels (relevance: equal label).
        gallery: `(G, d)` gallery embeddings.
        gallery_labels: `(G,)` labels.
        exclude_self: Query `i` is gallery item `i` (same array); it is removed
            from its own ranking.
        chunk: Queries per block.
        device: Torch device (default CPU).

    Returns:
        `{metric: (N,)}` for `CLASS_METRICS`, in 0-1 units.
    """
    dev = device or torch.device("cpu")
    q = torch.as_tensor(query, dtype=torch.float64, device=dev)
    g = torch.as_tensor(gallery, dtype=torch.float64, device=dev)
    ql = torch.as_tensor(query_labels, device=dev)
    gl = torch.as_tensor(gallery_labels, device=dev)
    g_sq = g.square().sum(dim=1)
    out: dict[str, list[FloatArray]] = {m: [] for m in CLASS_METRICS}
    for start in range(0, q.shape[0], chunk):
        block = q[start : start + chunk]
        dist = g_sq[None, :] - 2.0 * (block @ g.T) + block.square().sum(dim=1)[:, None]
        relevant = ql[start : start + chunk, None] == gl[None, :]
        if exclude_self:
            rows = torch.arange(block.shape[0], device=dev)
            dist[rows, start + rows] = float("inf")
            relevant[rows, start + rows] = False
        for name, values in _sorted_relevance_metrics(dist, relevant).items():
            out[name].append(values.cpu().numpy())
    return {m: numpy.concatenate(parts) for m, parts in out.items()}


def instance_retrieval(
    query: FloatArray,
    gallery: FloatArray,
    chunk: int = DEFAULT_CHUNK,
    device: torch.device | None = None,
) -> dict[str, FloatArray]:
    """Per-query recall@k of finding gallery item `i` for query `i` (Euclidean).

    Args:
        query: `(N, d)` query embeddings (one view per image).
        gallery: `(N, d)` gallery embeddings (another view per image).
        chunk: Queries per block.
        device: Torch device (default CPU).

    Returns:
        `{"R@1": (N,), "R@5": (N,)}` of 0/1 hits. The rank of the target is the
        number of strictly closer gallery items.
    """
    dev = device or torch.device("cpu")
    q = torch.as_tensor(query, dtype=torch.float64, device=dev)
    g = torch.as_tensor(gallery, dtype=torch.float64, device=dev)
    g_sq = g.square().sum(dim=1)
    ranks: list[IntArray] = []
    for start in range(0, q.shape[0], chunk):
        block = q[start : start + chunk]
        dist = g_sq[None, :] - 2.0 * (block @ g.T) + block.square().sum(dim=1)[:, None]
        rows = torch.arange(block.shape[0], device=dev)
        target = dist[rows, start + rows]
        ranks.append((dist < target[:, None]).sum(dim=1).cpu().numpy())
    rank = numpy.concatenate(ranks)
    return {f"R@{k}": (rank < k).astype(numpy.float64) for k in INSTANCE_KS}


def transform_views(
    x: FloatArray,
    condition: str,
    predicted: FloatArray,
    rng: numpy.random.Generator,
) -> FloatArray:
    """Applies a radius condition to `(N, V, d)` embeddings (V views per image).

    Args:
        x: Embeddings.
        condition: One of `INSTANCE_CONDITIONS`.
        predicted: `(N, V)` held-out direction-predicted log-radius (used by
            `RESIDUAL` only).
        rng: Generator of the permutation.

    Returns:
        Embeddings with the same directions and modified radii. `PERMUTE` and
        `RESIDUAL` permute *images* (every view of image `i` takes the value of
        view `v` of image `perm[i]`); `PERMUTE_VIEW` permutes all `N * V`
        embeddings independently.
    """
    n, v, d = x.shape
    flat = x.reshape(n * v, d)
    if condition == BASELINE:
        return x
    if condition == CONSTANT:
        return randomization.constant_radius(flat).reshape(n, v, d)
    log_r = features.log_radius(flat).reshape(n, v)
    if condition == PERMUTE:
        new = log_r[rng.permutation(n)]
    elif condition == RESIDUAL:
        new = predicted + (log_r - predicted)[rng.permutation(n)]
    elif condition == PERMUTE_VIEW:
        new = log_r.reshape(-1)[rng.permutation(n * v)].reshape(n, v)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return randomization.with_radius(flat, numpy.exp(new).reshape(-1)).reshape(n, v, d)


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Per-query values of one run.

    Attributes:
        name: Run name.
        lam: SIGReg weight.
        seed: Training seed.
        values: `(task, metric, condition) -> (num_queries,)` per-query metric
            (averaged over the shuffle repeats of randomized conditions).
        r2_test: Held-out `R^2` of log-radius from direction (test center views).
    """

    name: str
    lam: float
    seed: int
    values: dict[Key, FloatArray]
    r2_test: float


def view_pairs(num_views: int) -> tuple[tuple[int, int], ...]:
    """Disjoint augmented-view pairs `(0, 1), (2, 3), ...` (indices into views 1..K)."""
    return tuple((2 * i, 2 * i + 1) for i in range(num_views // 2))


def evaluate(
    train_z: FloatArray,
    train_labels: IntArray,
    test_z: FloatArray,
    test_labels: IntArray,
    num_repeats: int,
    cv_folds: int,
    seed: int,
    chunk: int = DEFAULT_CHUNK,
    device: torch.device | None = None,
) -> tuple[dict[Key, FloatArray], float]:
    """All tasks, metrics and conditions for one run.

    Args:
        train_z: `(M, K + 1, d)` train embeddings (center view first).
        train_labels: `(M,)` labels.
        test_z: `(N, K + 1, d)` test embeddings.
        test_labels: `(N,)` labels.
        num_repeats: Shuffles per randomized condition.
        cv_folds: Outer folds of the direction -> radius regression.
        seed: Seed of every random draw.
        chunk: Queries per block of the distance computation.
        device: Torch device.

    Returns:
        `(values, r2_test)`: per-query metrics keyed by `(task, metric,
        condition)` and the held-out `R^2` of the radius regression.
    """
    ytr = numpy.asarray(train_labels, dtype=numpy.int64).ravel()
    yte = numpy.asarray(test_labels, dtype=numpy.int64).ravel()
    c_train = features.select_views(train_z, "center")
    c_test = features.select_views(test_z, "center")
    aug = features.select_views(test_z, "augmented")
    num_aug = aug.shape[1]
    model = randomization.fit_radius_model(
        c_train[:, 0], c_test[:, 0], cv_folds, numpy.random.default_rng(seed)
    )
    # Held-out predictions at every test augmented view (fitted on train centers).
    u_train = features.direction(c_train[:, 0])
    rho_train = features.log_radius(c_train[:, 0])
    pred_aug = numpy.stack(
        [
            redundancy.fit_predict(
                u_train,
                rho_train,
                features.direction(aug[:, v]),
                rng=numpy.random.default_rng(seed),
            )
            for v in range(num_aug)
        ],
        axis=1,
    )
    pairs = view_pairs(num_aug)
    values: dict[Key, FloatArray] = {}

    def run_condition(condition: str, repeats: int) -> None:
        sums: dict[Key, FloatArray] = {}
        for repeat in range(repeats):
            rng_train = numpy.random.default_rng([seed, repeat, 0])
            rng_test = numpy.random.default_rng([seed, repeat, 1])
            rng_aug = numpy.random.default_rng([seed, repeat, 2])
            measured: dict[Key, FloatArray] = {}
            if condition in CLASS_CONDITIONS:
                test_t = transform_views(
                    c_test, condition, model.query[:, None], rng_test
                )[:, 0]
                train_t = transform_views(
                    c_train, condition, model.bank[:, None], rng_train
                )[:, 0]
                for task, gallery, gallery_labels, exclude in (
                    (CLASS_TEST, test_t, yte, True),
                    (CLASS_TRAIN, train_t, ytr, False),
                ):
                    result = class_retrieval(
                        test_t, yte, gallery, gallery_labels, exclude, chunk, device
                    )
                    measured.update(
                        {(task, m, condition): r for m, r in result.items()}
                    )
            aug_t = transform_views(aug, condition, pred_aug, rng_aug)
            hits: dict[str, list[FloatArray]] = {m: [] for m in INSTANCE_METRICS}
            for a, b in pairs:
                result = instance_retrieval(aug_t[:, a], aug_t[:, b], chunk, device)
                for m, r in result.items():
                    hits[m].append(r)
            measured.update(
                {
                    (INSTANCE, m, condition): numpy.mean(parts, axis=0)
                    for m, parts in hits.items()
                }
            )
            for key, value in measured.items():
                sums[key] = sums.get(key, 0.0) + value
        values.update({key: total / repeats for key, total in sums.items()})

    for condition in INSTANCE_CONDITIONS:
        random_condition = condition in (PERMUTE, RESIDUAL, PERMUTE_VIEW)
        run_condition(condition, num_repeats if random_condition else 1)
    return values, model.r2_query


def evaluate_shards(
    name: str,
    lam: float,
    seed: int,
    train: shards.EmbeddingShard,
    test: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    device: torch.device | None = None,
) -> RunResult:
    """`evaluate` on the z views of the `stl10_train` / `stl10_test` dumps."""
    values, r2 = evaluate(
        numpy.asarray(train.z, dtype=numpy.float64),
        train.label,
        numpy.asarray(test.z, dtype=numpy.float64),
        test.label,
        num_repeats=cfg.num_randomizations,
        cv_folds=cfg.cv_folds,
        seed=seed,
        device=device,
    )
    return RunResult(name, lam, seed, values, r2)


_SEP = "::"


def save_result(directory: pathlib.Path, result: RunResult) -> pathlib.Path:
    """Writes a run's per-query values as `<directory>/<name>.npz` (atomically)."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {_SEP.join(key): value for key, value in result.values.items()}
    arrays["meta"] = numpy.array([result.lam, result.seed, result.r2_test])
    dest = directory / f"{result.name}.npz"
    partial = dest.with_name(dest.name + ".partial")
    with partial.open("wb") as f:
        numpy.savez(f, **arrays)  # pyright: ignore[reportArgumentType]
    partial.replace(dest)
    return dest


def load_result(path: pathlib.Path) -> RunResult:
    """Reads a file written by `save_result`."""
    values: dict[Key, FloatArray] = {}
    with numpy.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            if name == "meta":
                continue
            task, metric, condition = name.split(_SEP)
            values[task, metric, condition] = arrays[name]
        lam, seed, r2 = (float(v) for v in arrays["meta"])
    return RunResult(path.stem, lam, int(seed), values, r2)


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Paired difference of one condition against the baseline, per lambda.

    Attributes:
        lam: SIGReg weight.
        task: Task id (`TASKS`).
        metric: Metric name.
        condition: The compared (non-baseline) condition.
        seeds: Training seeds averaged.
        baseline: Mean baseline value over seeds (0-1 units).
        value: Mean value of the condition over seeds.
        difference: `value - baseline`.
        ci90: 90% bootstrap interval of the difference.
        ci95: 95% bootstrap interval of the difference.
        verdict: `randomization.EQUIVALENT`, `DIFFERENT` or `INCONCLUSIVE`.
    """

    lam: float
    task: str
    metric: str
    condition: str
    seeds: int
    baseline: float
    value: float
    difference: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    verdict: str


def compare(
    results: collections.abc.Sequence[RunResult],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Comparison]:
    """Paired bootstrap differences vs. the baseline, per lambda (see module doc).

    Queries are resampled within each seed (one resample per seed shared by all
    metrics and conditions of that seed); per-seed mean differences are averaged.
    """
    out: list[Comparison] = []
    for lam in sorted({r.lam for r in results}):
        group = sorted((r for r in results if r.lam == lam), key=lambda r: r.seed)
        draws: dict[str, list[numpy.typing.NDArray[numpy.int64]]] = {}
        for task in TASKS:
            size = group[0].values[task, task_metrics(task)[0], BASELINE].size
            draws[task] = randomization.bootstrap_draws(
                size, len(group), num_bootstrap, rng
            )
        for task in TASKS:
            for metric in task_metrics(task):
                base = [r.values[task, metric, BASELINE] for r in group]
                for condition in task_conditions(task)[1:]:
                    cond = [r.values[task, metric, condition] for r in group]
                    diffs = [c - b for c, b in zip(cond, base, strict=True)]
                    ci90, ci95 = randomization.intervals(
                        randomization.bootstrap_seed_mean(
                            draws[task], lambda s, idx, d=diffs: d[s][idx].mean(axis=1)
                        )
                    )
                    b_mean = statistics.fmean(float(b.mean()) for b in base)
                    c_mean = statistics.fmean(float(c.mean()) for c in cond)
                    out.append(
                        Comparison(
                            lam=lam,
                            task=task,
                            metric=metric,
                            condition=condition,
                            seeds=len(group),
                            baseline=b_mean,
                            value=c_mean,
                            difference=c_mean - b_mean,
                            ci90=ci90,
                            ci95=ci95,
                            verdict=randomization.verdict(ci90, ci95, margin),
                        )
                    )
    return out


def comparison_rows(comparisons: list[Comparison]) -> list[Row]:
    """Comparisons as table rows, values and differences in points (x100)."""

    def interval(ci: tuple[float, float]) -> str:
        return f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"

    return [
        {
            "lambda": c.lam,
            "task": TASK_TITLES[c.task],
            "metric": c.metric,
            "condition": c.condition,
            "baseline (%)": f"{100 * c.baseline:.2f}",
            "condition (%)": f"{100 * c.value:.2f}",
            "difference (points)": f"{100 * c.difference:+.2f}",
            "90% CI (points)": interval(c.ci90),
            "95% CI (points)": interval(c.ci95),
            "verdict": c.verdict,
        }
        for c in comparisons
    ]


def value_rows(results: collections.abc.Sequence[RunResult], task: str) -> list[Row]:
    """Metrics (%), mean ± std over training seeds, per lambda and condition."""
    rows: list[Row] = []
    for lam in sorted({r.lam for r in results}):
        group = [r for r in results if r.lam == lam]
        for condition in task_conditions(task):
            row: Row = {"lambda": lam, "condition": condition, "seeds": len(group)}
            for metric in task_metrics(task):
                per_seed = [
                    100 * float(r.values[task, metric, condition].mean()) for r in group
                ]
                sd = statistics.stdev(per_seed) if len(per_seed) > 1 else float("nan")
                row[metric] = f"{statistics.fmean(per_seed):.2f} ± {sd:.2f}"
            rows.append(row)
    return rows
