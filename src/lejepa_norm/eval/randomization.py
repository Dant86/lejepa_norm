"""Is the embedding radius *used*? k-NN accuracy with the radius randomized.

The redundancy study showed that the radius is largely predictable from the
direction. That does not say whether the radius matters. Here a z-space k-NN
classifier (bank: STL-10 `train`, queries: STL-10 `test`, labels = classes) is
run after modifying the radius of the embeddings, and the accuracy change against
the unmodified embeddings is the measurement.

Conditions (each applied to the bank and to the queries *independently*, with the
same transformation type, so that bank and queries stay on the same footing):

* `baseline`: unmodified embeddings, Euclidean k-NN.
* `permute_radius`: every embedding keeps its direction and takes the radius of a
  uniformly random image of its own set (a permutation of the radii; the multiset
  of radii is unchanged). Repeated with different permutations.
* `residual_shuffle`: `log r = pred + resid`, where `pred` is the part of the
  log-radius predicted from the direction by kernel ridge (`redundancy`) and
  `resid = log r - pred`. Only `resid` is permuted across images; `pred` is kept.
  `pred` is always *held out*: for the bank it is the out-of-fold prediction of a
  nested-CV fit within the bank; for the queries it is the prediction of a model
  fitted on the whole bank (so the queries take no part in the fit). Using
  in-sample predictions would make `resid` the (overfit) leftover of the fit and
  hide direction-predictable radius in it. Consequence of the held-out choice:
  `resid` also contains the estimation error of the regression, so it slightly
  over-states the radius information that is *not* predictable from direction;
  the kept part is, if anything, a lower bound of what the direction explains.
  The shuffled log-radius `pred + resid[perm]` has approximately (not exactly)
  the original marginal.
* `constant_radius`: all embeddings rescaled to unit norm. Euclidean k-NN on unit
  vectors ranks neighbours exactly as cosine k-NN does, whatever the common
  radius is, so this row coincides with `cosine k-NN` of the unmodified
  embeddings, which is reported as a reference (and the tests assert the
  equality of predictions).

Euclidean k-NN is sensitive to radius by construction (a long vector is far from
everything); the constant-radius row and the cosine reference separate "radius
matters because of the metric" from "radius carries label information".
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

BASELINE = "baseline (unmodified)"
PERMUTE = "permute radius across images"
RESIDUAL = "residual-only shuffle (direction-predicted part kept)"
CONSTANT = "constant radius (unit norm)"
COSINE = "reference: cosine k-NN, unmodified"
CONDITIONS = (BASELINE, PERMUTE, RESIDUAL, CONSTANT, COSINE)
# Conditions that are randomized and hence repeated.
RANDOMIZED = (PERMUTE, RESIDUAL)

# Embedding of one image: view 0 (un-augmented center view) or the mean over the
# augmented views (the "embedding" feature of `downstream.feature_pair`).
CENTER = "center view"
AUGMENTED_MEAN = "mean of augmented views"
VIEW_KINDS = (CENTER, AUGMENTED_MEAN)


def image_embeddings(
    z: numpy.typing.NDArray[numpy.floating[typing.Any]], view_kind: str
) -> FloatArray:
    """One embedding per image from an `(N, K + 1, d)` array.

    Args:
        z: Embeddings with the center view at index 0.
        view_kind: `CENTER` or `AUGMENTED_MEAN`.

    Returns:
        `(N, d)` float64.
    """
    if view_kind == CENTER:
        return features.select_views(z, "center")[:, 0]
    if view_kind == AUGMENTED_MEAN:
        return features.select_views(z, "augmented").mean(axis=1)
    raise ValueError(f"unknown view kind {view_kind!r}")


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
    """Held-out direction -> log-radius predictions for a bank and its queries.

    Attributes:
        bank: Out-of-fold predictions of the bank log-radius, `(M,)`.
        query: Predictions at the queries from a model fitted on the bank, `(N,)`.
        r2_bank: Out-of-fold `R^2` within the bank.
        r2_query: Held-out `R^2` on the queries.
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


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """Result of `evaluate`.

    Attributes:
        rows: One row per condition (accuracy, spread over repeats).
        model: Rows describing the direction -> radius regression used by the
            residual shuffle (`R^2`, standard deviations of its parts).
    """

    rows: list[Row]
    model: list[Row]


_Transform = collections.abc.Callable[
    [FloatArray, FloatArray, numpy.random.Generator], FloatArray
]


def _permute(
    x: FloatArray, _predicted: FloatArray, rng: numpy.random.Generator
) -> FloatArray:
    return permute_radius(x, rng)


def _accuracy(
    bank: FloatArray,
    bank_y: IntArray,
    query: FloatArray,
    query_y: IntArray,
    k: int,
    num_classes: int,
    cosine: bool = False,
) -> float:
    vote = downstream.knn_votes if cosine else euclidean_knn_votes
    votes = vote(bank, bank_y, query, k, num_classes)
    return downstream.accuracy(votes.argmax(axis=1), query_y)


def _row(condition: str, metric: str, accuracies: list[float]) -> Row:
    return {
        "condition": condition,
        "k-NN metric": metric,
        "repeats": len(accuracies),
        "accuracy": statistics.fmean(accuracies),
        "sd over repeats": statistics.stdev(accuracies)
        if len(accuracies) > 1
        else float("nan"),
        "min": min(accuracies),
        "max": max(accuracies),
    }


def evaluate(
    bank: FloatArray,
    bank_labels: numpy.typing.ArrayLike,
    query: FloatArray,
    query_labels: numpy.typing.ArrayLike,
    k: int,
    num_repeats: int,
    cv_folds: int,
    seed: int,
) -> Evaluation:
    """Runs the conditions on one bank/query pair of image embeddings.

    Args:
        bank: `(M, d)` labeled embeddings (STL-10 `train`).
        bank_labels: `(M,)` integer labels.
        query: `(N, d)` query embeddings (STL-10 `test`).
        query_labels: `(N,)` integer labels.
        k: Neighbours.
        num_repeats: Randomizations (permutations) per randomized condition.
        cv_folds: Outer folds of the direction -> radius regression.
        seed: Seed of every random draw (regression folds, permutations).

    Returns:
        Accuracy rows per condition and regression diagnostics.
    """
    by = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    qy = numpy.asarray(query_labels, dtype=numpy.int64).ravel()
    num_classes = int(max(by.max(), qy.max())) + 1
    model = fit_radius_model(bank, query, cv_folds, numpy.random.default_rng(seed))

    def knn(b: FloatArray, q: FloatArray, cosine: bool = False) -> float:
        return _accuracy(b, by, q, qy, k, num_classes, cosine)

    rows = [
        _row(BASELINE, "euclidean", [knn(bank, query)]),
    ]
    transforms: tuple[tuple[str, _Transform], ...] = (
        (PERMUTE, _permute),
        (RESIDUAL, residual_shuffle),
    )
    for condition, transform in transforms:
        accuracies: list[float] = []
        for repeat in range(num_repeats):
            # Independent streams for the bank and the queries of each repeat.
            rng_bank = numpy.random.default_rng([seed, repeat, 0])
            rng_query = numpy.random.default_rng([seed, repeat, 1])
            accuracies.append(
                knn(
                    transform(bank, model.bank, rng_bank),
                    transform(query, model.query, rng_query),
                )
            )
        rows.append(_row(condition, "euclidean", accuracies))
    rows.append(
        _row(
            CONSTANT, "euclidean", [knn(constant_radius(bank), constant_radius(query))]
        )
    )
    rows.append(_row(COSINE, "cosine", [knn(bank, query, cosine=True)]))

    rho = features.log_radius(bank)
    diagnostics: list[Row] = [
        {
            "quantity": "R^2 of log-radius from direction",
            "bank (out-of-fold)": model.r2_bank,
            "queries (held out)": model.r2_query,
        },
        {
            "quantity": "sd of log-radius (bank)",
            "bank (out-of-fold)": float(rho.std()),
            "queries (held out)": float(features.log_radius(query).std()),
        },
        {
            "quantity": "sd of the residual log-radius",
            "bank (out-of-fold)": float((rho - model.bank).std()),
            "queries (held out)": float(
                (features.log_radius(query) - model.query).std()
            ),
        },
    ]
    return Evaluation(rows=rows, model=diagnostics)


@dataclasses.dataclass(frozen=True)
class RunResult:
    """The accuracy rows of one run for one view kind.

    Attributes:
        name: Run name.
        lam: SIGReg weight of the run.
        seed: Training seed of the run.
        view_kind: `CENTER` or `AUGMENTED_MEAN`.
        rows: `Evaluation.rows`.
    """

    name: str
    lam: float
    seed: int
    view_kind: str
    rows: list[Row]


def _mean_sd(values: list[float]) -> str:
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    return f"{statistics.fmean(values):.4f} ± {sd:.4f}"


def aggregate(results: list[RunResult]) -> list[Row]:
    """Mean ± std over training seeds, per lambda, view kind and condition.

    The value of a run is its mean accuracy over repeats; `repeat sd` is the mean
    over seeds of the within-run standard deviation over repeats, and `change` is
    the mean ± std over seeds of the accuracy minus the same run's baseline.

    Args:
        results: Per-run, per-view-kind results.

    Returns:
        Rows ordered by view kind, lambda, then condition.
    """
    rows: list[Row] = []
    for view_kind in dict.fromkeys(r.view_kind for r in results):
        for lam in sorted({r.lam for r in results if r.view_kind == view_kind}):
            group = [r for r in results if r.view_kind == view_kind and r.lam == lam]
            for condition in CONDITIONS:
                cells = [
                    next(row for row in r.rows if row["condition"] == condition)
                    for r in group
                ]
                base = [
                    float(
                        next(row for row in r.rows if row["condition"] == BASELINE)[
                            "accuracy"
                        ]
                    )
                    for r in group
                ]
                acc = [float(c["accuracy"]) for c in cells]
                sds = [float(c["sd over repeats"]) for c in cells]
                rows.append(
                    {
                        "view": view_kind,
                        "lambda": lam,
                        "condition": condition,
                        "seeds": len(group),
                        "accuracy (mean ± std over seeds)": _mean_sd(acc),
                        "change vs baseline": _mean_sd(
                            [a - b for a, b in zip(acc, base, strict=True)]
                        ),
                        "repeat sd (mean over seeds)": (
                            ""
                            if numpy.isnan(sds[0])
                            else f"{statistics.fmean(sds):.4f}"
                        ),
                    }
                )
    return rows


def per_run_rows(results: list[RunResult]) -> list[Row]:
    """Flattens `RunResult`s into one table (one row per run, view, condition)."""
    return [
        {"run": r.name, "view": r.view_kind, **row}
        for r in sorted(results, key=lambda r: (r.view_kind, r.lam, r.seed))
        for row in r.rows
    ]


def evaluate_shards(
    bank: shards.EmbeddingShard,
    query: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    seed: int,
) -> dict[str, Evaluation]:
    """`evaluate` in z-space for every view kind of one run.

    Args:
        bank: The `stl10_train` dump (labels = classes).
        query: The `stl10_test` dump.
        cfg: `knn_k`, `num_randomizations` and `cv_folds` are used.
        seed: Seed of all random draws.

    Returns:
        `Evaluation` per view kind (`VIEW_KINDS`).
    """
    return {
        view_kind: evaluate(
            image_embeddings(bank.z, view_kind),
            bank.label,
            image_embeddings(query.z, view_kind),
            query.label,
            k=cfg.knn_k,
            num_repeats=cfg.num_randomizations,
            cv_folds=cfg.cv_folds,
            seed=seed,
        )
        for view_kind in VIEW_KINDS
    }


def accuracy_by_lambda(
    results: list[RunResult], view_kind: str
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """Per condition: lambdas, mean and std over seeds of the run accuracies."""
    out: dict[str, tuple[list[float], list[float], list[float]]] = {}
    group = [r for r in results if r.view_kind == view_kind]
    for condition in CONDITIONS:
        lams = sorted({r.lam for r in group})
        means: list[float] = []
        stds: list[float] = []
        for lam in lams:
            acc = [
                float(
                    next(row for row in r.rows if row["condition"] == condition)[
                        "accuracy"
                    ]
                )
                for r in group
                if r.lam == lam
            ]
            means.append(statistics.fmean(acc))
            stds.append(statistics.stdev(acc) if len(acc) > 1 else 0.0)
        out[condition] = (lams, means, stds)
    return out
