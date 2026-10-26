"""Does the radius track the readers' confidence and errors, beyond the direction?

Pre-registration, definitions, predictions and reading rules:
`research_notes/radius_confidence/confidence.md` (written before any result).

For one run, the center-view embedding `x` (z for LeJEPA; h for the E2 baselines,
an explicit exception) of every query image is summarized by

* `rho = log ||x||` and `resid = rho - f(u)`, `u = x / ||x||`, `f` the kernel ridge
  of the log-radius on the direction fitted on the clean bank (`redundancy`);
* the outputs of three readers fitted on the clean bank only: `probe_z` (logistic
  probe on `x`), `probe_u` (the same on `u`, direction only) and `knn_u` (cosine
  k-NN, direction only): top class probability `p1`, margin `p1 - p2`, and
  correctness.

The statistics are error-detection AUROCs (errors positive) of `-rho`, `-resid`,
`-p1`, `-margin`; the nested comparison `M0 = [logit p1, margin]` against
`M1 = M0 + rho` and `M2 = M0 + resid` (5-fold cross-fitted logistic regression of
"error", the out-of-fold scores are compared); accuracy at 50% / 80% coverage when
abstaining on the lowest scores; Spearman correlations; a class-centered radius
control; per-cell radius shifts of the corrupted images against their clean twins;
and the same for ID vs OOD detection. Every statistic is a vector: the point
estimate followed by bootstrap replicates (images resampled; the same resample for
every cell of a unit, so group means are paired). The out-of-fold scores of the
nested models are computed once; the bootstrap does not refit them, and the
Spearman replicates use the ranks of the original data (a standard shortcut).
Seeds are combined by `aggregate`: points and replicates are averaged over the
training seeds (independent resamples per seed, as in `randomization`).
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import collections.abc
import dataclasses
import json
import math
import os
import typing

import numpy
import numpy.typing
import scipy.stats
import sklearn.linear_model
import sklearn.model_selection

from lejepa_norm.eval import downstream
from lejepa_norm.eval import features
from lejepa_norm.eval import ood_radius
from lejepa_norm.eval import randomization
from lejepa_norm.eval import redundancy

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
BoolArray = numpy.typing.NDArray[numpy.bool_]
Row = dict[str, str | int | float]

PROBE_Z = "probe_z"
PROBE_U = "probe_u"
KNN_U = "knn_u"
READERS = (PROBE_Z, PROBE_U, KNN_U)
# The readers that cannot see the radius: "beyond the direction" is read on these.
DIRECTION_READERS = (PROBE_U, KNN_U)

# Fixed in advance (pre-registration): AUROC differences, accuracy at coverage.
AUROC_MARGIN = ood_radius.OOD_AUROC_MARGIN
COVERAGE_MARGIN = 0.005
# A single signal is informative iff the lower 95% end of its AUROC exceeds this,
# uninformative iff its 90% interval lies inside the band.
INFORMATIVE_LOWER = 0.51
UNINFORMATIVE_BAND = (0.49, 0.51)
COVERAGES = (0.5, 0.8)
CURVE_COVERAGES = tuple(c / 10 for c in range(1, 11))
# Replicates for the 50 corrupted cells (the clean and OOD sets use the config's).
CORRUPTED_BOOTSTRAP = 200

ADDS = "adds"
EQUIVALENT = "equivalent"
WORSE = "worse"
INCONCLUSIVE = "inconclusive"
INFORMATIVE = "informative"
UNINFORMATIVE = "uninformative"

CLEAN = "clean"
ALL_CORRUPTIONS = "all corruptions"
# Rows per chunk of the vectorized bootstrap (bounds memory).
_CHUNK = 100
_RADIUS_BATCH = 90_000
_EPS = 1e-4
FOLDS = 5


@dataclasses.dataclass
class ReaderOutput:
    """Top-2 outputs of one reader on a query set.

    Attributes:
        p1: Top class probability (vote share for k-NN), `(N,)`.
        margin: `p1 - p2`, `(N,)`.
        pred: Predicted class, `(N,)`.
    """

    p1: FloatArray
    margin: FloatArray
    pred: IntArray


def _top2(probs: FloatArray) -> ReaderOutput:
    part = numpy.partition(probs, -2, axis=1)
    return ReaderOutput(
        p1=part[:, -1],
        margin=part[:, -1] - part[:, -2],
        pred=probs.argmax(axis=1).astype(numpy.int64),
    )


class Readers:
    """The three readers, fitted once on the clean labeled bank."""

    def __init__(
        self, bank: FloatArray, labels: IntArray, k: int, num_classes: int
    ) -> None:
        """Fits the probes; the k-NN reader only stores the bank."""
        self._bank = bank
        self._labels = labels
        self._k = k
        self._num_classes = num_classes
        self._probe_z = downstream.LinearProbe(bank, labels, num_classes)
        self._probe_u = downstream.LinearProbe(
            features.direction(bank), labels, num_classes
        )

    def __call__(self, x: FloatArray) -> dict[str, ReaderOutput]:
        """Outputs of every reader on the query embeddings `x`."""
        votes = downstream.knn_votes(
            self._bank, self._labels, x, self._k, self._num_classes
        )
        return {
            PROBE_Z: _top2(self._probe_z.predict_proba(x)),
            PROBE_U: _top2(self._probe_u.predict_proba(features.direction(x))),
            KNN_U: _top2(votes),
        }


def predict_log_radius(
    bank: FloatArray, queries: collections.abc.Sequence[FloatArray], seed: int
) -> list[FloatArray]:
    """Held-out `f(u)` at every query set (the ridge is refitted per batch).

    Args:
        bank: Clean bank embeddings `(M, d)`.
        queries: Query embedding arrays; they are batched together.
        seed: Seed of the bandwidth estimate.

    Returns:
        One `(N_i,)` array of predicted log-radii per query array.
    """
    u_bank, rho_bank = features.direction(bank), features.log_radius(bank)
    out: list[FloatArray] = []
    pending: list[FloatArray] = []
    sizes: list[int] = []

    def flush() -> None:
        if not pending:
            return
        pred = redundancy.fit_predict(
            u_bank,
            rho_bank,
            numpy.concatenate([features.direction(p) for p in pending]),
            rng=numpy.random.default_rng(seed),
        )
        offset = 0
        for n in sizes:
            out.append(pred[offset : offset + n])
            offset += n
        pending.clear()
        sizes.clear()

    for query in queries:
        if sum(sizes) + query.shape[0] > _RADIUS_BATCH and pending:
            flush()
        pending.append(query)
        sizes.append(query.shape[0])
    flush()
    return out


def _logit(p: FloatArray) -> FloatArray:
    clipped = numpy.clip(p, _EPS, 1 - _EPS)
    return numpy.log(clipped / (1 - clipped))


def out_of_fold_scores(design: FloatArray, target: BoolArray, seed: int) -> FloatArray:
    """Cross-fitted logit of a logistic regression of `target` on `design`.

    Args:
        design: Features `(N, f)`; standardized here (unsupervised).
        target: Positive class (error, or OOD), `(N,)`.
        seed: Fold assignment.

    Returns:
        Out-of-fold decision values `(N,)` (high = more likely positive); NaN if a
        class has fewer than `FOLDS` members.
    """
    out = numpy.full(design.shape[0], numpy.nan)
    if target.sum() < FOLDS or (~target).sum() < FOLDS:
        return out
    std = design.std(axis=0)
    x = (design - design.mean(axis=0)) / numpy.where(std > 0, std, 1.0)
    folds = sklearn.model_selection.StratifiedKFold(
        FOLDS, shuffle=True, random_state=seed
    )
    for train, test in folds.split(x, target):
        model = sklearn.linear_model.LogisticRegression(C=1.0, max_iter=500)
        model.fit(x[train], target[train])
        out[test] = model.decision_function(x[test])
    return out


def _auroc_rows(score: FloatArray, positive: BoolArray) -> FloatArray:
    """AUROC (P(score of a positive > score of a negative)) of every row."""
    n = score.shape[1]
    ranks = scipy.stats.rankdata(score, axis=1)
    npos = positive.sum(axis=1).astype(numpy.float64)
    nneg = n - npos
    with numpy.errstate(invalid="ignore", divide="ignore"):
        return ((ranks * positive).sum(axis=1) - npos * (npos + 1) / 2) / (npos * nneg)


def _coverage_rows(
    score: FloatArray, correct: FloatArray, coverage: float
) -> FloatArray:
    """Accuracy of the `coverage` fraction with the highest score, per row."""
    n = score.shape[1]
    keep = max(1, math.ceil(coverage * n))
    top = numpy.argpartition(-score, keep - 1, axis=1)[:, :keep]
    return numpy.take_along_axis(correct, top, axis=1).mean(axis=1)


def _pearson_rows(a: FloatArray, b: FloatArray) -> FloatArray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / numpy.sqrt(
            (a * a).sum(axis=1) * (b * b).sum(axis=1)
        )


class Booter:
    """Point estimate plus bootstrap replicates of per-image statistics.

    Row 0 of `rows` is the identity; rows `1..B` are image resamples. Every method
    returns an array `(1 + B,)`.
    """

    def __init__(self, num_images: int, num_bootstrap: int, seed: int) -> None:
        """Draws `num_bootstrap` resamples of `num_images` images."""
        rng = numpy.random.default_rng(seed)
        self.num_images = num_images
        self.rows = numpy.concatenate(
            [
                numpy.arange(num_images, dtype=numpy.int64)[None],
                rng.integers(0, num_images, size=(num_bootstrap, num_images)),
            ]
        )
        self._jitter = 1e-7 * rng.random(num_images)

    def _chunked(
        self, fn: collections.abc.Callable[[IntArray], FloatArray]
    ) -> FloatArray:
        return numpy.concatenate(
            [fn(self.rows[i : i + _CHUNK]) for i in range(0, len(self.rows), _CHUNK)]
        )

    def mean(self, values: FloatArray) -> FloatArray:
        """Mean of `values` over the resampled images."""
        return self._chunked(lambda r: values[r].mean(axis=1))

    def auroc(self, score: FloatArray, positive: BoolArray) -> FloatArray:
        """AUROC of `score` for the `positive` images."""
        return self._chunked(lambda r: _auroc_rows(score[r], positive[r]))

    def coverage_accuracy(
        self, score: FloatArray, correct: BoolArray, coverage: float
    ) -> FloatArray:
        """Accuracy of the `coverage` fraction with the highest `score`."""
        jittered = score + self._jitter
        hits = correct.astype(numpy.float64)
        return self._chunked(lambda r: _coverage_rows(jittered[r], hits[r], coverage))

    def spearman(self, a: FloatArray, b: FloatArray) -> FloatArray:
        """Spearman correlation (resampling the ranks of the original data)."""
        ra, rb = scipy.stats.rankdata(a), scipy.stats.rankdata(b)
        return self._chunked(lambda r: _pearson_rows(ra[r], rb[r]))


@dataclasses.dataclass
class Query:
    """Per-image summaries of one query set (all `(N,)`, rows in `image_id` order).

    Attributes:
        rho: Log-radius.
        resid: `rho - f(u)`.
        label: Class labels.
        readers: Reader outputs.
    """

    rho: FloatArray
    resid: FloatArray
    label: IntArray
    readers: dict[str, ReaderOutput]


def _nested_scores(
    out: ReaderOutput, rho: FloatArray, resid: FloatArray, target: BoolArray, seed: int
) -> dict[str, FloatArray]:
    base = numpy.stack([_logit(out.p1), out.margin], axis=1)
    return {
        "M0": out_of_fold_scores(base, target, seed),
        "M1": out_of_fold_scores(numpy.column_stack([base, rho]), target, seed),
        "M2": out_of_fold_scores(numpy.column_stack([base, resid]), target, seed),
    }


def _model_statistics(
    prefix: str,
    booter: Booter,
    scores: dict[str, FloatArray],
    positive: BoolArray,
    out: dict[str, FloatArray],
) -> None:
    """AUROC of the three nested models and their differences."""
    for name, score in scores.items():
        out[f"{prefix}auroc[{name}]"] = booter.auroc(score, positive)
    out[f"{prefix}d_auroc[M1-M0]"] = (
        out[f"{prefix}auroc[M1]"] - out[f"{prefix}auroc[M0]"]
    )
    out[f"{prefix}d_auroc[M2-M0]"] = (
        out[f"{prefix}auroc[M2]"] - out[f"{prefix}auroc[M0]"]
    )


def set_statistics(
    query: Query,
    class_rho: FloatArray,
    booter: Booter,
    seed: int,
    twin: Query | None = None,
) -> dict[str, FloatArray]:
    """All statistics of one clean or corrupted query set.

    Args:
        query: The set.
        class_rho: Mean log-radius of each class in the clean bank.
        booter: Resamples (the same for every set of a unit).
        seed: Fold assignment of the nested models.
        twin: The clean twins of the same images (same row order); adds the radius
            shift and the within-image correlations.

    Returns:
        Metric name to `(1 + B,)` array (`(1,)` for point-only metrics).
    """
    out: dict[str, FloatArray] = {}
    rho_cc = query.rho - class_rho[query.label]
    if twin is not None:
        out["shift_rho"] = booter.mean(query.rho - twin.rho)
        out["shift_resid"] = booter.mean(query.resid - twin.resid)
    for reader, o in query.readers.items():
        p = f"{reader}|"
        error = o.pred != query.label
        correct = ~error
        out[f"{p}accuracy"] = booter.mean(correct.astype(numpy.float64))
        out[f"{p}auroc[-rho]"] = booter.auroc(-query.rho, error)
        out[f"{p}auroc[-resid]"] = booter.auroc(-query.resid, error)
        out[f"{p}auroc[-p1]"] = booter.auroc(-o.p1, error)
        out[f"{p}auroc[-margin]"] = booter.auroc(-o.margin, error)
        out[f"{p}auroc[-rho_cc]"] = booter.auroc(-rho_cc, error)
        per_class = [
            float(_auroc_rows((-query.rho[m])[None], error[m][None])[0])
            for c in numpy.unique(query.label)
            for m in [query.label == c]
            if error[m].any() and correct[m].any()
        ]
        out[f"{p}auroc_within_class[-rho]"] = numpy.array([numpy.mean(per_class)])
        out[f"{p}spearman[rho,p1]"] = booter.spearman(query.rho, o.p1)
        out[f"{p}spearman[resid,p1]"] = booter.spearman(query.resid, o.p1)
        scores = _nested_scores(o, query.rho, query.resid, error, seed)
        _model_statistics(p, booter, scores, error, out)
        confidence = {
            "p1": o.p1,
            "rho": query.rho,
            "resid": query.resid,
            **{k: -v for k, v in scores.items()},
        }
        for coverage in COVERAGES:
            tag = f"acc@{int(100 * coverage)}"
            for name, score in confidence.items():
                out[f"{p}{tag}[{name}]"] = booter.coverage_accuracy(
                    score, correct, coverage
                )
            for m in ("M1", "M2"):
                out[f"{p}d_{tag}[{m}-M0]"] = (
                    out[f"{p}{tag}[{m}]"] - out[f"{p}{tag}[M0]"]
                )
        for coverage in CURVE_COVERAGES:
            for name in ("p1", "rho", "resid", "M0", "M2"):
                out[f"{p}curve[{name}]@{coverage:g}"] = booter.coverage_accuracy(
                    confidence[name], correct, coverage
                )[:1]
        if twin is not None:
            delta_p1 = o.p1 - twin.readers[reader].p1
            out[f"{p}spearman[d_rho,d_p1]"] = booter.spearman(
                query.rho - twin.rho, delta_p1
            )
    return out


def ood_statistics(
    seen: Query,
    other: Query,
    booter_id: Booter,
    booter_ood: Booter,
    seed: int,
) -> dict[str, FloatArray]:
    """ID (`seen`) vs OOD (`other`) detection statistics, OOD positive.

    The ID and OOD images are resampled separately (`booter_id`, `booter_ood`).
    """
    out: dict[str, FloatArray] = {}
    n_id = seen.rho.shape[0]
    n_ood = other.rho.shape[0]
    positive = numpy.concatenate([numpy.zeros(n_id, bool), numpy.ones(n_ood, bool)])
    rho = numpy.concatenate([seen.rho, other.rho])
    resid = numpy.concatenate([seen.resid, other.resid])

    rows = numpy.concatenate([booter_id.rows, booter_ood.rows + n_id], axis=1)

    def auroc(score: FloatArray) -> FloatArray:
        return numpy.concatenate(
            [
                _auroc_rows(score[r], positive[r])
                for r in (rows[i : i + _CHUNK] for i in range(0, len(rows), _CHUNK))
            ]
        )

    out["shift_rho"] = booter_ood.mean(other.rho) - booter_id.mean(seen.rho)
    for reader in seen.readers:
        p = f"{reader}|"
        a, b = seen.readers[reader], other.readers[reader]
        p1 = numpy.concatenate([a.p1, b.p1])
        margin = numpy.concatenate([a.margin, b.margin])
        out[f"{p}auroc_ood[-rho]"] = auroc(-rho)
        out[f"{p}auroc_ood[-resid]"] = auroc(-resid)
        out[f"{p}auroc_ood[-p1]"] = auroc(-p1)
        out[f"{p}auroc_ood[-margin]"] = auroc(-margin)
        base = numpy.stack([_logit(p1), margin], axis=1)
        scores = {
            "M0": out_of_fold_scores(base, positive, seed),
            "M1": out_of_fold_scores(numpy.column_stack([base, rho]), positive, seed),
            "M2": out_of_fold_scores(numpy.column_stack([base, resid]), positive, seed),
        }
        for name, score in scores.items():
            out[f"{p}auroc_ood[{name}]"] = auroc(score)
        out[f"{p}d_auroc_ood[M1-M0]"] = (
            out[f"{p}auroc_ood[M1]"] - out[f"{p}auroc_ood[M0]"]
        )
        out[f"{p}d_auroc_ood[M2-M0]"] = (
            out[f"{p}auroc_ood[M2]"] - out[f"{p}auroc_ood[M0]"]
        )
    return out


def _nanmean(values: FloatArray, axis: int) -> FloatArray:
    """NaN-ignoring mean that is NaN (without a warning) where all are NaN."""
    count = (~numpy.isnan(values)).sum(axis=axis)
    total = numpy.nansum(values, axis=axis)
    return numpy.where(count > 0, total / numpy.maximum(count, 1), numpy.nan)


def mean_groups(
    members: collections.abc.Sequence[dict[str, FloatArray]],
) -> dict[str, FloatArray]:
    """Mean over sets of every metric (NaN-ignoring; the sets share the resamples)."""
    keys = members[0].keys()
    return {k: _nanmean(numpy.stack([m[k] for m in members]), 0) for k in keys}


# ---------------------------------------------------------------- unit files


@dataclasses.dataclass
class Unit:
    """The statistics of one run.

    Attributes:
        name: Run name.
        group: Aggregation group (`lam<lambda>` for LeJEPA, the loss variant for
            the baselines).
        seed: Training seed.
        space: `"z"` or `"h"`.
        stats: `(set group, metric)` to the `(1 + B,)` array.
    """

    name: str
    group: str
    seed: int
    space: str
    stats: dict[tuple[str, str], FloatArray]


SEP = "||"


def save_unit(path: str | os.PathLike[str], unit: Unit) -> None:
    """Writes a unit as an uncompressed `.npz` (atomically)."""
    arrays: dict[str, typing.Any] = {
        f"{g}{SEP}{m}": v for (g, m), v in unit.stats.items()
    }
    arrays["meta"] = numpy.array(
        json.dumps(
            {
                "name": unit.name,
                "group": unit.group,
                "seed": unit.seed,
                "space": unit.space,
            }
        )
    )
    partial = f"{os.fspath(path)}.partial"
    with open(partial, "wb") as handle:
        numpy.savez(handle, **arrays)
    os.replace(partial, path)


def load_unit(path: str | os.PathLike[str]) -> Unit:
    """Reads a unit written by `save_unit`."""
    with numpy.load(path, allow_pickle=False) as arrays:
        meta = json.loads(str(arrays["meta"]))
        stats: dict[tuple[str, str], FloatArray] = {}
        for key in arrays.files:
            if key == "meta":
                continue
            group, metric = key.split(SEP, 1)
            stats[group, metric] = numpy.asarray(arrays[key], dtype=numpy.float64)
    return Unit(meta["name"], meta["group"], meta["seed"], meta["space"], stats)


# --------------------------------------------------------------- aggregation


@dataclasses.dataclass(frozen=True)
class Estimate:
    """A seed-averaged statistic.

    Attributes:
        value: Mean over seeds of the point estimates.
        ci90: 90% bootstrap interval, or None for point-only metrics.
        ci95: 95% bootstrap interval, or None.
        seeds: Number of units averaged.
    """

    value: float
    ci90: tuple[float, float] | None
    ci95: tuple[float, float] | None
    seeds: int


def aggregate(
    units: collections.abc.Sequence[Unit], group: str, metric: str
) -> Estimate:
    """Seed average of one metric of one set group (NaN if absent)."""
    arrays = [u.stats[group, metric] for u in units if (group, metric) in u.stats]
    if not arrays:
        return Estimate(float("nan"), None, None, 0)
    value = float(_nanmean(numpy.array([a[0] for a in arrays]), 0))
    if any(a.shape[0] < 2 for a in arrays):
        return Estimate(value, None, None, len(arrays))
    boot = _nanmean(numpy.stack([a[1:] for a in arrays]), 0)
    ci90, ci95 = randomization.intervals(boot)
    return Estimate(value, ci90, ci95, len(arrays))


def signal_verdict(e: Estimate) -> str:
    """`informative` / `uninformative` / `inconclusive` for an AUROC against 0.5."""
    if e.ci95 is None or e.ci90 is None:
        return "n/a"
    if e.ci95[0] > INFORMATIVE_LOWER:
        return INFORMATIVE
    if UNINFORMATIVE_BAND[0] < e.ci90[0] and e.ci90[1] < UNINFORMATIVE_BAND[1]:
        return UNINFORMATIVE
    return INCONCLUSIVE


def difference_verdict(e: Estimate, margin: float) -> str:
    """`adds` / `equivalent` / `worse` / `inconclusive` for a paired difference."""
    if e.ci95 is None or e.ci90 is None:
        return "n/a"
    if e.ci95[0] > margin:
        return ADDS
    if -margin < e.ci90[0] and e.ci90[1] < margin:
        return EQUIVALENT
    if e.ci95[1] < -margin:
        return WORSE
    return INCONCLUSIVE


def _interval(ci: tuple[float, float] | None, digits: int = 3) -> str:
    if ci is None:
        return ""
    return f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def signal_rows(
    units: collections.abc.Sequence[Unit], group: str, label: str
) -> list[Row]:
    """Error-detection AUROC of each signal per reader for one set group."""
    rows: list[Row] = []
    for reader in READERS:
        accuracy = aggregate(units, group, f"{reader}|accuracy")
        for signal in ("-rho", "-resid", "-p1", "-margin", "-rho_cc"):
            e = aggregate(units, group, f"{reader}|auroc[{signal}]")
            rows.append(
                {
                    "set": label,
                    "reader": reader,
                    "signal": signal,
                    "reader accuracy": round(accuracy.value, 4),
                    "AUROC (errors)": round(e.value, 4),
                    "95% CI": _interval(e.ci95).replace("+", ""),
                    "verdict": signal_verdict(e),
                }
            )
        e = aggregate(units, group, f"{reader}|auroc_within_class[-rho]")
        rows.append(
            {
                "set": label,
                "reader": reader,
                "signal": "-rho, within class (macro)",
                "reader accuracy": round(accuracy.value, 4),
                "AUROC (errors)": round(e.value, 4),
                "95% CI": "",
                "verdict": "n/a",
            }
        )
    return rows


def correlation_rows(
    units: collections.abc.Sequence[Unit], group: str, label: str
) -> list[Row]:
    """Spearman correlations of the radius signals with each reader's `p1`."""
    rows: list[Row] = []
    for reader in READERS:
        row: Row = {"set": label, "reader": reader}
        for name in ("rho,p1", "resid,p1", "d_rho,d_p1"):
            e = aggregate(units, group, f"{reader}|spearman[{name}]")
            if e.seeds:
                row[f"Spearman({name})"] = round(e.value, 3)
                row[f"{name} 95% CI"] = _interval(e.ci95)
        rows.append(row)
    return rows


def nested_rows(
    units: collections.abc.Sequence[Unit], group: str, label: str
) -> list[Row]:
    """Does the radius add error detection beyond the reader's own confidence?"""
    rows: list[Row] = []
    for reader in READERS:
        for model in ("M1", "M2"):
            for metric, margin, scale, unit in (
                ("d_auroc", AUROC_MARGIN, 1.0, "AUROC"),
                ("d_acc@50", COVERAGE_MARGIN, 100.0, "points"),
                ("d_acc@80", COVERAGE_MARGIN, 100.0, "points"),
            ):
                e = aggregate(units, group, f"{reader}|{metric}[{model}-M0]")
                rows.append(
                    {
                        "set": label,
                        "reader": reader,
                        "comparison": f"{metric.replace('d_', '')}: {model} - M0",
                        "difference": round(scale * e.value, 4),
                        "unit": unit,
                        "95% CI": _interval(
                            None
                            if e.ci95 is None
                            else (scale * e.ci95[0], scale * e.ci95[1]),
                            4,
                        ),
                        "verdict": difference_verdict(e, margin),
                    }
                )
    return rows


def ood_rows(units: collections.abc.Sequence[Unit], label: str) -> list[Row]:
    """ID-vs-OOD AUROC of the radius signals and the nested comparisons."""
    rows: list[Row] = []
    for reader in READERS:
        for signal in ("-rho", "-resid", "-p1", "-margin"):
            e = aggregate(units, label, f"{reader}|auroc_ood[{signal}]")
            rows.append(
                {
                    "OOD set": label,
                    "reader": reader,
                    "comparison": f"AUROC of {signal}",
                    "value": round(e.value, 4),
                    "95% CI": _interval(e.ci95).replace("+", ""),
                    "verdict": "",
                }
            )
        for model in ("M1", "M2"):
            e = aggregate(units, label, f"{reader}|d_auroc_ood[{model}-M0]")
            rows.append(
                {
                    "OOD set": label,
                    "reader": reader,
                    "comparison": f"AUROC: {model} - M0",
                    "value": round(e.value, 4),
                    "95% CI": _interval(e.ci95, 4),
                    "verdict": difference_verdict(e, AUROC_MARGIN),
                }
            )
    return rows
