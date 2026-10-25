"""Is the embedding radius *used*? Classification with the radius randomized.

The redundancy study showed that the radius is largely predictable from the
direction. That does not say whether the radius matters. Here classifiers on the
z-space embeddings of STL-10 (bank/training set: `train`, test set: `test`,
labels = classes) are run after modifying the radius, and the paired accuracy
change against the unmodified embeddings is the measurement. Only `z` is
analyzed (LeJEPA's product is z) and only the un-augmented center view of each
image is used (the convention of the online k-NN monitor).

Conditions, each applied to the bank and to the test set *independently* with the
same transformation type:

* `baseline`: unmodified embeddings.
* `permute`: every embedding keeps its direction and takes the radius of a
  uniformly random image of its own set (the multiset of radii is unchanged).
* `residual`: `log r = pred + resid`, where `pred` is the part of the log-radius
  predicted from the direction by kernel ridge (`redundancy`), `resid = log r -
  pred`; only `resid` is permuted, `pred` is kept. `pred` is always *held out*:
  for the bank the out-of-fold prediction of a nested-CV fit within the bank, for
  the test set the prediction of a model fitted on the whole bank. In-sample
  predictions would let the fit absorb part of the radius and hide
  direction-predictable radius in `resid`. Consequence: `resid` also contains the
  regression's estimation error, so it slightly over-states the radius
  information that is *not* predictable from direction. The shuffled
  log-radius has approximately (not exactly) the original marginal.
* `constant`: all embeddings rescaled to unit norm. Euclidean k-NN on unit vectors
  ranks neighbours exactly like cosine k-NN, whatever the common radius is.

Readers (`READERS`): Euclidean k-NN, cosine k-NN (reference: invariant to the
radius by construction, so its paired differences must be exactly 0), a linear
(logistic) probe and a one-hidden-layer MLP with a fixed epoch budget whose weight
decay is chosen on a validation split of the clean bank (once per run, then
reused for all conditions). Two protocols (`PROTOCOLS`): `retrain` (the reader is
refitted on the transformed bank and tested on the transformed test set) and
`frozen` (fitted on the clean bank, applied to the transformed test set).

Statistics: the per-image correctness of every condition (averaged over its
shuffle repeats) is compared with the baseline's on the *same* test images. The
shuffle repeats only quantify shuffle noise; they are not a sample of images or
of trainings. `compare` resamples test images within each training seed (paired
bootstrap) and averages over seeds, giving a CI of the mean paired difference per
lambda, and a TOST-style verdict: *equivalent* if the 90% CI (two one-sided 5%
tests) lies inside `(-margin, +margin)`; otherwise *different* if the 95% CI
excludes 0; otherwise *inconclusive*. The margin is fixed in the config
(`eval.equivalence_margin`) before the results are looked at. Image resampling
does not cover training-seed variability beyond averaging the three seeds.
"""

import collections.abc
import dataclasses
import statistics
import typing

import numpy
import numpy.typing

from lejepa_norm import config
from lejepa_norm.eval import downstream
from lejepa_norm.eval import features
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
Row = dict[str, str | int | float]

BASELINE = "baseline"
PERMUTE = "permute radius across images"
RESIDUAL = "residual-only shuffle"
CONSTANT = "constant radius (unit norm)"
CONDITIONS = (BASELINE, PERMUTE, RESIDUAL, CONSTANT)

KNN_EUCLIDEAN = "k-NN (euclidean)"
KNN_COSINE = "k-NN (cosine, reference)"
LINEAR = "linear probe"
MLP = "MLP (1 hidden layer)"
READERS = (KNN_EUCLIDEAN, KNN_COSINE, LINEAR, MLP)

RETRAIN = "retrained on shuffled train"
FROZEN = "trained on clean train, frozen"
PROTOCOLS = (RETRAIN, FROZEN)

# Weight decays tried for the MLP on the validation split; fraction held out.
MLP_WEIGHT_DECAYS = (1e-4, 1e-2, 1.0)
MLP_VALIDATION_FRACTION = 0.2

EQUIVALENT = "equivalent"
DIFFERENT = "different"
INCONCLUSIVE = "inconclusive"

Predictor = collections.abc.Callable[[FloatArray], IntArray]
Fit = collections.abc.Callable[[FloatArray], Predictor]
Transform = collections.abc.Callable[
    [FloatArray, FloatArray, numpy.random.Generator], FloatArray
]


def center_embeddings(
    z: numpy.typing.NDArray[numpy.floating[typing.Any]],
) -> FloatArray:
    """The center-view embedding of every image from an `(N, K + 1, d)` array."""
    return features.select_views(z, "center")[:, 0]


def with_radius(x: FloatArray, new_radius: FloatArray) -> FloatArray:
    """Rows of `x` rescaled to the norms `new_radius`, directions unchanged."""
    return features.direction(x) * new_radius[:, None]


def permute_radius(x: FloatArray, rng: numpy.random.Generator) -> FloatArray:
    """Each row keeps its direction and takes the radius of a random row."""
    return with_radius(x, features.radius(x)[rng.permutation(x.shape[0])])


def residual_shuffle(
    x: FloatArray, predicted: FloatArray, rng: numpy.random.Generator
) -> FloatArray:
    """Keeps `predicted` (log-radius) per row and permutes the residual.

    Args:
        x: `(N, d)` embeddings.
        predicted: `(N,)` direction-predicted log-radius.
        rng: Generator for the permutation.

    Returns:
        Embeddings with log-radius `predicted + (log r - predicted)[perm]`.
    """
    residual = features.log_radius(x) - predicted
    return with_radius(x, numpy.exp(predicted + residual[rng.permutation(x.shape[0])]))


def constant_radius(x: FloatArray, value: float = 1.0) -> FloatArray:
    """Rows of `x` rescaled to norm `value`."""
    return with_radius(x, numpy.full(x.shape[0], value))


def euclidean_knn_votes(
    bank: numpy.typing.ArrayLike,
    bank_labels: numpy.typing.ArrayLike,
    query: numpy.typing.ArrayLike,
    k: int,
    num_classes: int,
    chunk: int = 1024,
) -> FloatArray:
    """Euclidean k-NN vote shares, the counterpart of `downstream.knn_votes`.

    Args:
        bank: `(M, d)` labeled features.
        bank_labels: `(M,)` integer labels in `[0, num_classes)`.
        query: `(N, d)` query features.
        k: Number of neighbours (capped at M).
        num_classes: Number of classes.
        chunk: Queries processed per block.

    Returns:
        `(N, num_classes)` fraction of the `k` nearest bank rows in each class.
    """
    b = numpy.asarray(bank, dtype=numpy.float64)
    q = numpy.asarray(query, dtype=numpy.float64)
    labels = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    k = min(k, b.shape[0])
    b_sq = numpy.square(b).sum(axis=1)
    votes = numpy.zeros((q.shape[0], num_classes), dtype=numpy.float64)
    for start in range(0, q.shape[0], chunk):
        block = q[start : start + chunk]
        # Squared distance up to a per-query constant (||q||^2 does not rank).
        dist = b_sq[None, :] - 2.0 * (block @ b.T)
        nearest = numpy.argpartition(dist, k - 1, axis=1)[:, :k]
        rows: IntArray = numpy.repeat(
            numpy.arange(nearest.shape[0], dtype=numpy.int64), k
        )
        numpy.add.at(votes[start : start + chunk], (rows, labels[nearest].ravel()), 1)
    return votes / k


@dataclasses.dataclass(frozen=True)
class RadiusModel:
    """Held-out direction -> log-radius predictions for a bank and a test set.

    Attributes:
        bank: Out-of-fold predictions of the bank log-radius, `(M,)`.
        query: Predictions at the test set from a model fitted on the bank, `(N,)`.
        r2_bank: Out-of-fold `R^2` within the bank.
        r2_query: Held-out `R^2` on the test set.
    """

    bank: FloatArray
    query: FloatArray
    r2_bank: float
    r2_query: float


def fit_radius_model(
    bank: FloatArray,
    query: FloatArray,
    cv_folds: int,
    rng: numpy.random.Generator,
) -> RadiusModel:
    """Kernel ridge of the log-radius on the direction (see module docstring)."""
    u_bank, u_query = features.direction(bank), features.direction(query)
    rho_bank, rho_query = features.log_radius(bank), features.log_radius(query)
    oof = redundancy.NestedKernelRidge(u_bank, "exp_dot", cv_folds, rng).fit(rho_bank)
    pred_query = redundancy.fit_predict(u_bank, rho_bank, u_query, rng=rng)
    ss_tot = float(numpy.square(rho_query - rho_query.mean()).sum())
    ss_res = float(numpy.square(rho_query - pred_query).sum())
    return RadiusModel(
        bank=oof.predictions,
        query=pred_query,
        r2_bank=oof.r2,
        r2_query=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    )


def select_weight_decay(
    x: FloatArray,
    y: IntArray,
    num_classes: int,
    seed: int,
    weight_decays: tuple[float, ...] = MLP_WEIGHT_DECAYS,
) -> float:
    """MLP weight decay with the best accuracy on a held-out part of `(x, y)`."""
    rng = numpy.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    num_val = max(1, int(MLP_VALIDATION_FRACTION * x.shape[0]))
    val, fit = order[:num_val], order[num_val:]
    scores = [
        downstream.accuracy(
            downstream.MlpProbe(x[fit], y[fit], num_classes, wd, seed=seed).predict(
                x[val]
            ),
            y[val],
        )
        for wd in weight_decays
    ]
    return weight_decays[int(numpy.argmax(scores))]


def make_fit(
    reader: str,
    labels: IntArray,
    k: int,
    num_classes: int,
    seed: int,
    weight_decay: float,
) -> Fit:
    """The `Fit` of a reader: bank features -> predictor of class indices."""

    def knn_euclidean(bank: FloatArray) -> Predictor:
        return lambda q: euclidean_knn_votes(bank, labels, q, k, num_classes).argmax(
            axis=1
        )

    def knn_cosine(bank: FloatArray) -> Predictor:
        return lambda q: downstream.knn_votes(bank, labels, q, k, num_classes).argmax(
            axis=1
        )

    def linear(bank: FloatArray) -> Predictor:
        return downstream.LinearProbe(bank, labels, num_classes).predict

    def mlp(bank: FloatArray) -> Predictor:
        return downstream.MlpProbe(
            bank, labels, num_classes, weight_decay, seed=seed
        ).predict

    return {
        KNN_EUCLIDEAN: knn_euclidean,
        KNN_COSINE: knn_cosine,
        LINEAR: linear,
        MLP: mlp,
    }[reader]


@dataclasses.dataclass(frozen=True)
class Cell:
    """Outcome of one (reader, protocol, condition) on the test images.

    Attributes:
        correct: Per test image, the fraction of the shuffle repeats in which it
            was classified correctly (0 or 1 for deterministic conditions), `(N,)`.
        repeat_accuracy: Test accuracy of each repeat.
    """

    correct: FloatArray
    repeat_accuracy: tuple[float, ...]


Key = tuple[str, str, str]


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """Result of `evaluate`.

    Attributes:
        cells: Outcome per (reader, protocol, condition).
        model: Rows describing the direction -> radius regression used by the
            residual shuffle (`R^2`, standard deviations of its parts).
        weight_decay: MLP weight decay chosen on the validation split.
    """

    cells: dict[Key, Cell]
    model: list[Row]
    weight_decay: float


def evaluate(
    bank: FloatArray,
    bank_labels: numpy.typing.ArrayLike,
    query: FloatArray,
    query_labels: numpy.typing.ArrayLike,
    k: int,
    num_repeats: int,
    cv_folds: int,
    seed: int,
    readers: tuple[str, ...] = READERS,
) -> Evaluation:
    """Runs all readers, protocols and conditions on one bank/test pair.

    Args:
        bank: `(M, d)` labeled embeddings (STL-10 `train`).
        bank_labels: `(M,)` integer labels.
        query: `(N, d)` test embeddings (STL-10 `test`).
        query_labels: `(N,)` integer labels.
        k: Neighbours of the k-NN readers.
        num_repeats: Shuffles per randomized condition.
        cv_folds: Outer folds of the direction -> radius regression.
        seed: Seed of every random draw (regression folds, shuffles, MLP).
        readers: Subset of `READERS` to run.

    Returns:
        Per-image correctness per (reader, protocol, condition) and regression
        diagnostics.
    """
    by = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    qy = numpy.asarray(query_labels, dtype=numpy.int64).ravel()
    num_classes = int(max(by.max(), qy.max())) + 1
    model = fit_radius_model(bank, query, cv_folds, numpy.random.default_rng(seed))
    weight_decay = (
        select_weight_decay(bank, by, num_classes, seed) if MLP in readers else 0.0
    )

    def permute(
        x: FloatArray, _predicted: FloatArray, rng: numpy.random.Generator
    ) -> FloatArray:
        return permute_radius(x, rng)

    transforms: tuple[tuple[str, Transform], ...] = (
        (PERMUTE, permute),
        (RESIDUAL, residual_shuffle),
    )
    cells: dict[Key, Cell] = {}

    def store(
        reader: str, protocol: str, condition: str, preds: list[IntArray]
    ) -> None:
        hits = numpy.stack([p == qy for p in preds]).astype(numpy.float64)
        cells[reader, protocol, condition] = Cell(
            correct=hits.mean(axis=0),
            repeat_accuracy=tuple(float(h.mean()) for h in hits),
        )

    for reader in readers:
        fit = make_fit(reader, by, k, num_classes, seed, weight_decay)
        clean = fit(bank)
        for protocol in PROTOCOLS:
            store(reader, protocol, BASELINE, [clean(query)])
        # Constant radius is deterministic: no repeats.
        unit_query = constant_radius(query)
        store(reader, FROZEN, CONSTANT, [clean(unit_query)])
        store(reader, RETRAIN, CONSTANT, [fit(constant_radius(bank))(unit_query)])
        for condition, transform in transforms:
            frozen: list[IntArray] = []
            retrained: list[IntArray] = []
            for repeat in range(num_repeats):
                # Independent streams for the bank and the test set of each repeat.
                rng_bank = numpy.random.default_rng([seed, repeat, 0])
                rng_query = numpy.random.default_rng([seed, repeat, 1])
                shuffled_query = transform(query, model.query, rng_query)
                frozen.append(clean(shuffled_query))
                retrained.append(
                    fit(transform(bank, model.bank, rng_bank))(shuffled_query)
                )
            store(reader, FROZEN, condition, frozen)
            store(reader, RETRAIN, condition, retrained)

    rho_bank, rho_query = features.log_radius(bank), features.log_radius(query)
    diagnostics: list[Row] = [
        {
            "quantity": "R^2 of log-radius from direction",
            "bank (out-of-fold)": model.r2_bank,
            "test (held out)": model.r2_query,
        },
        {
            "quantity": "sd of log-radius",
            "bank (out-of-fold)": float(rho_bank.std()),
            "test (held out)": float(rho_query.std()),
        },
        {
            "quantity": "sd of the residual log-radius",
            "bank (out-of-fold)": float((rho_bank - model.bank).std()),
            "test (held out)": float((rho_query - model.query).std()),
        },
    ]
    return Evaluation(cells=cells, model=diagnostics, weight_decay=weight_decay)


def evaluate_shards(
    bank: shards.EmbeddingShard,
    query: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    seed: int,
) -> Evaluation:
    """`evaluate` on the z center views of the `stl10_train` / `stl10_test` dumps."""
    return evaluate(
        center_embeddings(bank.z),
        bank.label,
        center_embeddings(query.z),
        query.label,
        k=cfg.knn_k,
        num_repeats=cfg.num_randomizations,
        cv_folds=cfg.cv_folds,
        seed=seed,
    )


@dataclasses.dataclass(frozen=True)
class RunResult:
    """The evaluation of one run.

    Attributes:
        name: Run name.
        lam: SIGReg weight of the run.
        seed: Training seed of the run.
        evaluation: The `Evaluation`.
    """

    name: str
    lam: float
    seed: int
    evaluation: Evaluation


def _mean_sd(values: collections.abc.Sequence[float], digits: int = 4) -> str:
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    return f"{statistics.fmean(values):.{digits}f} ± {sd:.{digits}f}"


def per_run_rows(results: list[RunResult]) -> list[Row]:
    """One row per run, reader, protocol and condition (accuracy and shuffle sd)."""
    rows: list[Row] = []
    for r in sorted(results, key=lambda r: (r.lam, r.seed)):
        for (reader, protocol, condition), cell in r.evaluation.cells.items():
            acc = cell.repeat_accuracy
            rows.append(
                {
                    "run": r.name,
                    "reader": reader,
                    "protocol": protocol,
                    "condition": condition,
                    "repeats": len(acc),
                    "accuracy": statistics.fmean(acc),
                    "shuffle sd": statistics.stdev(acc) if len(acc) > 1 else "",
                }
            )
    return rows


def accuracy_rows(results: list[RunResult]) -> list[Row]:
    """Accuracy mean ± std over training seeds per lambda, reader, protocol, condition.

    A run's value is its mean over shuffle repeats; `shuffle sd` is the mean over
    seeds of the within-run standard deviation over repeats.
    """
    rows: list[Row] = []
    for lam in sorted({r.lam for r in results}):
        group = [r for r in results if r.lam == lam]
        for reader in READERS:
            for protocol in PROTOCOLS:
                for condition in CONDITIONS:
                    accs = [
                        r.evaluation.cells[reader, protocol, condition].repeat_accuracy
                        for r in group
                    ]
                    sds = [statistics.stdev(a) for a in accs if len(a) > 1]
                    rows.append(
                        {
                            "lambda": lam,
                            "reader": reader,
                            "protocol": protocol,
                            "condition": condition,
                            "seeds": len(group),
                            "accuracy (mean ± std over seeds)": _mean_sd(
                                [statistics.fmean(a) for a in accs]
                            ),
                            "shuffle sd (mean over seeds)": (
                                f"{statistics.fmean(sds):.4f}" if sds else ""
                            ),
                        }
                    )
    return rows


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Paired difference of one condition against the baseline, for one lambda.

    Attributes:
        lam: SIGReg weight.
        reader: Reader name.
        protocol: Protocol name.
        condition: The compared (non-baseline) condition.
        seeds: Training seeds averaged.
        baseline: Mean baseline accuracy over seeds.
        accuracy: Mean accuracy of the condition over seeds.
        difference: `accuracy - baseline`, in accuracy (0-1) units.
        ci90: 90% bootstrap interval of the difference (equivalence test).
        ci95: 95% bootstrap interval of the difference.
        verdict: `EQUIVALENT`, `DIFFERENT` or `INCONCLUSIVE`.
    """

    lam: float
    reader: str
    protocol: str
    condition: str
    seeds: int
    baseline: float
    accuracy: float
    difference: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    verdict: str


def verdict(ci90: tuple[float, float], ci95: tuple[float, float], margin: float) -> str:
    """TOST-style verdict (see the module docstring)."""
    if ci90[0] > -margin and ci90[1] < margin:
        return EQUIVALENT
    if ci95[0] > 0 or ci95[1] < 0:
        return DIFFERENT
    return INCONCLUSIVE


def compare(
    results: list[RunResult],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Comparison]:
    """Paired bootstrap differences vs. the baseline, per lambda.

    Test images are resampled with replacement independently within each seed
    (the same resample for all readers/conditions of that seed, which keeps every
    comparison paired) and the per-seed mean differences are averaged over seeds.

    Args:
        results: Evaluations of all runs.
        margin: Equivalence margin in accuracy units (0.005 = 0.5 points).
        num_bootstrap: Bootstrap replicates.
        rng: Generator for the resamples.

    Returns:
        One `Comparison` per lambda, reader, protocol and non-baseline condition.
    """
    out: list[Comparison] = []
    for lam in sorted({r.lam for r in results}):
        group = sorted((r for r in results if r.lam == lam), key=lambda r: r.seed)
        num_images = next(iter(group[0].evaluation.cells.values())).correct.size
        # One resample per seed, shared by every comparison of that seed.
        draws = [
            rng.integers(0, num_images, size=(num_bootstrap, num_images)) for _ in group
        ]
        for reader in READERS:
            for protocol in PROTOCOLS:
                base_cells = [
                    r.evaluation.cells[reader, protocol, BASELINE] for r in group
                ]
                for condition in CONDITIONS[1:]:
                    cells = [
                        r.evaluation.cells[reader, protocol, condition] for r in group
                    ]
                    diffs = [
                        c.correct - b.correct
                        for c, b in zip(cells, base_cells, strict=True)
                    ]
                    boot = numpy.mean(
                        [
                            d[idx].mean(axis=1)
                            for d, idx in zip(diffs, draws, strict=True)
                        ],
                        axis=0,
                    )
                    lo90, hi90, lo95, hi95 = (
                        float(v)
                        for v in numpy.quantile(boot, [0.05, 0.95, 0.025, 0.975])
                    )
                    base = statistics.fmean(float(b.correct.mean()) for b in base_cells)
                    acc = statistics.fmean(float(c.correct.mean()) for c in cells)
                    out.append(
                        Comparison(
                            lam=lam,
                            reader=reader,
                            protocol=protocol,
                            condition=condition,
                            seeds=len(group),
                            baseline=base,
                            accuracy=acc,
                            difference=acc - base,
                            ci90=(lo90, hi90),
                            ci95=(lo95, hi95),
                            verdict=verdict((lo90, hi90), (lo95, hi95), margin),
                        )
                    )
    return out


def comparison_rows(comparisons: list[Comparison]) -> list[Row]:
    """Comparisons as table rows, differences in accuracy points (x100)."""

    def interval(ci: tuple[float, float]) -> str:
        return f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"

    return [
        {
            "lambda": c.lam,
            "reader": c.reader,
            "protocol": c.protocol,
            "condition": c.condition,
            "baseline acc": f"{c.baseline:.4f}",
            "condition acc": f"{c.accuracy:.4f}",
            "difference (points)": f"{100 * c.difference:+.2f}",
            "90% CI (points)": interval(c.ci90),
            "95% CI (points)": interval(c.ci95),
            "verdict": c.verdict,
        }
        for c in comparisons
    ]
