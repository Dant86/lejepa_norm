"""Does the radius carry out-of-distribution information beyond the direction?

STL-10 `test` (in-distribution, ID) versus DTD `test` and SVHN `test` (OOD), from
the center-view z embeddings only (LeJEPA's product is z; the bank is the clean
STL-10 `train` set). Every score is oriented *higher = more OOD*; AUROC is
`P(score of a random OOD image > score of a random ID image)` and FPR@95TPR treats
ID as the positive class: the threshold keeps 95% of the ID images and the FPR is
the fraction of OOD images that are kept (lower is better).

Scores (`k` = `EvalConfig.knn_k`, distances to the k-th nearest bank embedding):

* `radius ||z||`: the raw radius (AUROC below 0.5 means OOD images have the
  *smaller* radius). Its orientation is not chosen after looking at the data; both
  directions are reported by the value itself.
* `radius deviation`: `|log r - mu| / sd` with `mu`, `sd` of the bank log-radius:
  an orientation-free, unsupervised radius score (OOD images with a larger or a
  smaller radius than the ID images both score high).
* `kNN, direction`: k-NN distance of the unit-normalized z (direction only).
* `kNN, raw z`: k-NN distance of the raw z (direction and radius).
* `direction + radius`: the equal-weight sum of the bank-standardized direction
  score and the bank-standardized radius deviation. Standardization uses the bank
  alone (leave-one-out k-NN scores for the bank), no OOD data is used to build it.
* `direction + signed log-radius, best weight` (optimistic upper bound): the best of
  the weights `ORACLE_WEIGHTS` on the standardized signed log-radius, chosen on the
  evaluation data (re-chosen inside every bootstrap replicate). If even this
  does not beat the direction score, no linear use of the radius helps.

Controls analogous to the radius randomization of E1 (`NUM_PERMUTATIONS`
repeats, averaged AUROC): the radius is permuted across images, and the
permutation is over the *pooled* query set (ID test and the OOD set), which
destroys any association between radius and ID/OOD status while leaving every
direction untouched; the bank radii are permuted within the bank. Controls:
`kNN, raw z` with permuted radii, the radius deviation of permuted radii (must be
at 0.5), and `direction + radius` with permuted radii. The constant-radius control
is the `kNN, direction` score itself (k-NN on unit vectors). A residual-only
shuffle is not run: the clean direction -> radius map is off-support for OOD
images (see E6), so the "residual" of an OOD image is not defined.

Pre-registered criteria (fixed before any result was looked at, same status as the
0.5-point accuracy margin of E1; 1 AUROC point is the smallest difference regarded
as practically meaningful, about the size of what separates neighbouring OOD
scores in the literature, and far above the sampling noise of 8000 + 1880 / 26032
images):

* `radius adds information` for an OOD set if the lower end of the 95% paired
  bootstrap CI of `AUROC(direction + radius) - AUROC(direction)` exceeds
  `OOD_AUROC_MARGIN` = 0.01 (averaged over the three training seeds of a lambda).
  `does not add` if the 90% CI lies inside `(-0.01, +0.01)` (TOST analogue);
  `worse` if the upper end of the 95% CI is below `-0.01`; otherwise
  `inconclusive`.
* `radius alone is informative` if the 95% CI of its AUROC lies entirely outside
  the band `[0.5 - RADIUS_NULL_BAND, 0.5 + RADIUS_NULL_BAND]` = [0.45, 0.55];
  `not informative` if it lies entirely inside the band; otherwise
  `inconclusive`.

Bootstrap: the ID and the OOD images are resampled separately (with replacement),
the same resample for all scores of a seed (paired), and the per-seed statistics are
averaged over seeds. Seed variability beyond averaging is not modeled.
"""

import dataclasses
import os
import typing

import numpy
import numpy.typing

from lejepa_norm.eval import features

FloatArray = numpy.typing.NDArray[numpy.float64]
Row = dict[str, str | int | float]

# Fixed in advance (AUROC units).
OOD_AUROC_MARGIN = 0.01
RADIUS_NULL_BAND = 0.05
NUM_PERMUTATIONS = 5
ORACLE_WEIGHTS = (-4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TPR = 0.95

RADIUS = "radius ||z|| (high = OOD)"
DEVIATION = "radius deviation |log r - mu| / sd"
KNN_DIRECTION = "kNN distance, direction only (unit z)"
KNN_RAW = "kNN distance, raw z"
COMBINED = "direction + radius deviation (equal weights)"
ORACLE = "direction + signed log-radius, best weight (optimistic)"
KNN_RAW_PERMUTED = "control: kNN distance, raw z, radius permuted"
DEVIATION_PERMUTED = "control: radius deviation, radius permuted"
COMBINED_PERMUTED = "control: direction + radius deviation, radius permuted"
SCORES = (
    RADIUS,
    DEVIATION,
    KNN_DIRECTION,
    KNN_RAW,
    COMBINED,
    ORACLE,
    KNN_RAW_PERMUTED,
    DEVIATION_PERMUTED,
    COMBINED_PERMUTED,
)
# Scores computed as several columns (repeats or weights) and reduced to one.
PERMUTED = (KNN_RAW_PERMUTED, DEVIATION_PERMUTED, COMBINED_PERMUTED)

ADDS = "radius adds information"
NOT_ADD = "does not add (equivalent)"
WORSE = "worse"
INCONCLUSIVE = "inconclusive"
INFORMATIVE = "informative"
NOT_INFORMATIVE = "not informative"


def knn_distance(
    bank: FloatArray,
    query: FloatArray,
    k: int,
    exclude_self: bool = False,
    chunk: int = 2048,
) -> FloatArray:
    """Euclidean distance to the k-th nearest bank row.

    Args:
        bank: `(M, d)` reference embeddings.
        query: `(N, d)` embeddings.
        k: Rank of the neighbour (capped at M).
        exclude_self: `query` is `bank` itself: skip each row's own entry.
        chunk: Queries per block.

    Returns:
        `(N,)` distances.
    """
    rank = min(k, bank.shape[0] - (1 if exclude_self else 0))
    b_sq = numpy.square(bank).sum(axis=1)
    out = numpy.empty(query.shape[0])
    for start in range(0, query.shape[0], chunk):
        block = query[start : start + chunk]
        sq = (
            numpy.square(block).sum(axis=1)[:, None]
            + b_sq[None, :]
            - 2.0 * (block @ bank.T)
        )
        index = rank if exclude_self else rank - 1
        out[start : start + chunk] = numpy.partition(sq, index, axis=1)[:, index]
    return numpy.sqrt(numpy.maximum(out, 0.0))


@dataclasses.dataclass(frozen=True)
class BankStats:
    """Standardization constants from the bank alone.

    Attributes:
        direction_mean: Mean leave-one-out k-NN distance of the unit bank.
        direction_sd: Its standard deviation.
        log_radius_mean: Mean bank log-radius.
        log_radius_sd: Its standard deviation.
        deviation_mean: Mean of `|log r - mu| / sd` over the bank.
        deviation_sd: Its standard deviation.
    """

    direction_mean: float
    direction_sd: float
    log_radius_mean: float
    log_radius_sd: float
    deviation_mean: float
    deviation_sd: float


def bank_stats(bank: FloatArray, k: int) -> BankStats:
    """Leave-one-out k-NN distances (direction) and log-radius moments of the bank."""
    loo = knn_distance(features.direction(bank), features.direction(bank), k, True)
    rho = features.log_radius(bank)
    mu, sd = float(rho.mean()), float(rho.std())
    dev = numpy.abs(rho - mu) / sd
    return BankStats(
        float(loo.mean()),
        float(loo.std()),
        mu,
        sd,
        float(dev.mean()),
        float(dev.std()),
    )


def _deviation(x: FloatArray, stats: BankStats) -> FloatArray:
    return numpy.abs(features.log_radius(x) - stats.log_radius_mean) / (
        stats.log_radius_sd
    )


def compute_scores(
    bank: FloatArray,
    id_test: FloatArray,
    ood: FloatArray,
    k: int,
    seed: int,
) -> dict[str, FloatArray]:
    """All score columns for the pooled query set `[id_test, ood]`.

    Args:
        bank: `(M, d)` clean STL-10 `train` center z.
        id_test: `(N_id, d)` STL-10 `test` center z.
        ood: `(N_ood, d)` center z of one OOD set.
        k: Neighbour rank of the k-NN scores.
        seed: Seed of the radius permutations.

    Returns:
        Column name to `(N_id + N_ood,)` scores (higher = more OOD). Columns:
        the names in `SCORES` that are single columns, `"<ORACLE> | w=<w>"` per
        weight and `"<name> | repeat <r>"` per permutation repeat of `PERMUTED`.
    """
    query = numpy.concatenate([id_test, ood]).astype(numpy.float64)
    bank = bank.astype(numpy.float64)
    stats = bank_stats(bank, k)
    bank_u, query_u = features.direction(bank), features.direction(query)
    direction = (
        knn_distance(bank_u, query_u, k) - stats.direction_mean
    ) / stats.direction_sd
    deviation = (_deviation(query, stats) - stats.deviation_mean) / stats.deviation_sd
    rho = features.log_radius(query)
    signed = (rho - stats.log_radius_mean) / stats.log_radius_sd
    columns: dict[str, FloatArray] = {
        RADIUS: features.radius(query),
        DEVIATION: _deviation(query, stats),
        KNN_DIRECTION: knn_distance(bank_u, query_u, k),
        KNN_RAW: knn_distance(bank, query, k),
        COMBINED: direction + deviation,
    }
    for w in ORACLE_WEIGHTS:
        columns[f"{ORACLE} | w={w:g}"] = direction + w * signed
    for repeat in range(NUM_PERMUTATIONS):
        rng = numpy.random.default_rng([seed, repeat])
        perm_bank = (
            features.direction(bank)
            * features.radius(bank)[rng.permutation(bank.shape[0])][:, None]
        )
        perm_query = (
            query_u * features.radius(query)[rng.permutation(query.shape[0])][:, None]
        )
        dev_perm = _deviation(perm_query, stats)
        columns[f"{KNN_RAW_PERMUTED} | repeat {repeat}"] = knn_distance(
            perm_bank, perm_query, k
        )
        columns[f"{DEVIATION_PERMUTED} | repeat {repeat}"] = dev_perm
        columns[f"{COMBINED_PERMUTED} | repeat {repeat}"] = (
            direction + (dev_perm - stats.deviation_mean) / stats.deviation_sd
        )
    return columns


@dataclasses.dataclass
class OodUnit:
    """Score columns of one run for every OOD set.

    Attributes:
        name: Run name.
        lam: SIGReg weight.
        seed: Training seed.
        num_id: Number of ID images (the first rows of every column).
        scores: OOD set name to column name to `(N_id + N_ood,)` scores.
    """

    name: str
    lam: float
    seed: int
    num_id: int
    scores: dict[str, dict[str, FloatArray]]


def save_unit(path: str | os.PathLike[str], unit: OodUnit) -> None:
    """Writes a unit as an uncompressed `.npz` (atomically)."""
    arrays: dict[str, numpy.typing.NDArray[typing.Any]] = {
        "name": numpy.asarray(unit.name),
        "lam": numpy.asarray(unit.lam),
        "seed": numpy.asarray(unit.seed),
        "num_id": numpy.asarray(unit.num_id),
    }
    for ood_set, columns in unit.scores.items():
        for name, values in columns.items():
            arrays[f"x::{ood_set}::{name}"] = values
    partial = str(path) + ".partial"
    with open(partial, "wb") as f:
        numpy.savez(f, **arrays)  # pyright: ignore[reportArgumentType]
    os.replace(partial, path)


def load_unit(path: str | os.PathLike[str]) -> OodUnit:
    """Reads a unit written by `save_unit`."""
    scores: dict[str, dict[str, FloatArray]] = {}
    with numpy.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            if key.startswith("x::"):
                _, ood_set, name = key.split("::", 2)
                scores.setdefault(ood_set, {})[name] = arrays[key]
        return OodUnit(
            name=str(arrays["name"]),
            lam=float(arrays["lam"]),
            seed=int(arrays["seed"]),
            num_id=int(arrays["num_id"]),
            scores=scores,
        )


def roc_statistics(
    scores: FloatArray, is_ood: numpy.typing.NDArray[numpy.bool_], counts: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """AUROC and FPR@95TPR of every column for weighted (resampled) images.

    Args:
        scores: `(n, C)` score columns, higher = more OOD.
        is_ood: `(n,)` booleans.
        counts: `(B, n)` image multiplicities of `B` resamples whose ID and OOD
            totals are constant (stratified resampling); ones for the sample.

    Returns:
        `(auroc, fpr)`, each `(B, C)`. Ties are broken by the stable sort order,
        which only matters for exactly equal scores.
    """
    num_id = float(counts[:, ~is_ood].sum(axis=1)[0])
    num_ood = float(counts[:, is_ood].sum(axis=1)[0])
    auroc = numpy.empty((counts.shape[0], scores.shape[1]))
    fpr = numpy.empty_like(auroc)
    for c in range(scores.shape[1]):
        order = numpy.argsort(scores[:, c], kind="stable")
        ood_sorted = is_ood[order]
        w = counts[:, order]
        w_id = w * ~ood_sorted
        w_ood = w * ood_sorted
        cum_id = numpy.cumsum(w_id, axis=1)
        auroc[:, c] = (w_ood * (cum_id - w_id)).sum(axis=1) / (num_id * num_ood)
        # First position that keeps 95% of the ID images; FPR = OOD kept there.
        cut = numpy.argmax(cum_id >= TPR * num_id - 1e-9, axis=1)
        cum_ood = numpy.cumsum(w_ood, axis=1)
        fpr[:, c] = cum_ood[numpy.arange(cum_ood.shape[0]), cut] / num_ood
    return auroc, fpr


def _stratified_counts(
    num_id: int,
    num_ood: int,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> FloatArray:
    def one(n: int) -> FloatArray:
        idx = rng.integers(0, n, size=(num_bootstrap, n))
        idx += numpy.arange(num_bootstrap)[:, None] * n
        return (
            numpy.bincount(idx.ravel(), minlength=num_bootstrap * n)
            .reshape(num_bootstrap, n)
            .astype(numpy.float64)
        )

    return numpy.concatenate([one(num_id), one(num_ood)], axis=1)


def _reduce(
    names: list[str], auroc: FloatArray, fpr: FloatArray
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    """Reduces columns to scores.

    Permutation repeats are averaged; the oracle weights are maximized per
    replicate (FPR taken at the weight with the best AUROC).
    """
    out_a: dict[str, FloatArray] = {}
    out_f: dict[str, FloatArray] = {}
    for score in SCORES:
        members = [
            i for i, n in enumerate(names) if n == score or n.startswith(score + " | ")
        ]
        if score == ORACLE:
            best = auroc[:, members].argmax(axis=1)
            rows = numpy.arange(auroc.shape[0], dtype=numpy.intp)
            out_a[score] = auroc[rows, numpy.asarray(members)[best]]
            out_f[score] = fpr[rows, numpy.asarray(members)[best]]
        else:
            out_a[score] = auroc[:, members].mean(axis=1)
            out_f[score] = fpr[:, members].mean(axis=1)
    return out_a, out_f


@dataclasses.dataclass(frozen=True)
class OodResult:
    """Seed-averaged statistics per (lambda, OOD set), as arrays over bootstrap.

    Attributes:
        lam: SIGReg weight.
        ood_set: OOD set name.
        seeds: Number of seeds averaged.
        num_id: ID images.
        num_ood: OOD images.
        auroc: Score name to `(1 + B,)` values; element 0 is the full sample.
        fpr: Same for FPR@95TPR.
        oracle_weights: Best oracle weight on the full sample, per seed.
    """

    lam: float
    ood_set: str
    seeds: int
    num_id: int
    num_ood: int
    auroc: dict[str, FloatArray]
    fpr: dict[str, FloatArray]
    oracle_weights: list[float]


def evaluate(
    units: list[OodUnit],
    num_bootstrap: int,
    rng: numpy.random.Generator,
    chunk: int = 125,
) -> list[OodResult]:
    """Seed-averaged AUROC / FPR@95TPR with paired image bootstrap, per lambda.

    Args:
        units: One per run.
        num_bootstrap: Bootstrap replicates (the full-sample value is element 0).
        rng: Generator of the resamples.
        chunk: Replicates processed at once (bounds memory).
    """
    results: list[OodResult] = []
    for lam in sorted({u.lam for u in units}):
        group = sorted((u for u in units if u.lam == lam), key=lambda u: u.seed)
        for ood_set in sorted({s for u in group for s in u.scores}):
            sums_a: dict[str, FloatArray] = {}
            sums_f: dict[str, FloatArray] = {}
            best_weights: list[float] = []
            n_id = n_ood = 0
            for unit in group:
                columns = unit.scores[ood_set]
                names = list(columns)
                matrix = numpy.stack([columns[n] for n in names], axis=1)
                n_id = unit.num_id
                n_ood = matrix.shape[0] - n_id
                is_ood = numpy.arange(matrix.shape[0], dtype=numpy.intp) >= n_id
                full_a, full_f = roc_statistics(
                    matrix, is_ood, numpy.ones((1, matrix.shape[0]))
                )
                parts_a, parts_f = [full_a], [full_f]
                done = 0
                while done < num_bootstrap:
                    size = min(chunk, num_bootstrap - done)
                    counts = _stratified_counts(n_id, n_ood, size, rng)
                    a, f = roc_statistics(matrix, is_ood, counts)
                    parts_a.append(a)
                    parts_f.append(f)
                    done += size
                red_a, red_f = _reduce(
                    names, numpy.concatenate(parts_a), numpy.concatenate(parts_f)
                )
                for score in SCORES:
                    sums_a[score] = sums_a.get(score, 0.0) + red_a[score] / len(group)
                    sums_f[score] = sums_f.get(score, 0.0) + red_f[score] / len(group)
                oracle = [
                    i for i, n in enumerate(names) if n.startswith(ORACLE + " | ")
                ]
                best_weights.append(ORACLE_WEIGHTS[int(full_a[0, oracle].argmax())])
            results.append(
                OodResult(
                    lam, ood_set, len(group), n_id, n_ood, sums_a, sums_f, best_weights
                )
            )
    return results


def _ci(values: FloatArray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    lo, hi = numpy.quantile(values[1:], [tail, 1.0 - tail])
    return float(lo), float(hi)


def _fmt(values: FloatArray, scale: float = 1.0, digits: int = 3) -> str:
    lo, hi = _ci(values, 0.95)
    point, low, high = (f"{scale * v:.{digits}f}" for v in (values[0], lo, hi))
    return f"{point} [{low}, {high}]"


def radius_verdict(auroc: FloatArray) -> str:
    """Pre-registered reading of the radius-alone AUROC (see module docstring)."""
    lo, hi = _ci(auroc, 0.95)
    if lo > 0.5 + RADIUS_NULL_BAND or hi < 0.5 - RADIUS_NULL_BAND:
        return INFORMATIVE
    if lo >= 0.5 - RADIUS_NULL_BAND and hi <= 0.5 + RADIUS_NULL_BAND:
        return NOT_INFORMATIVE
    return INCONCLUSIVE


def addition_verdict(difference: FloatArray, margin: float = OOD_AUROC_MARGIN) -> str:
    """Pre-registered reading of an AUROC difference (see module docstring)."""
    lo95, hi95 = _ci(difference, 0.95)
    lo90, hi90 = _ci(difference, 0.90)
    if lo95 > margin:
        return ADDS
    if lo90 > -margin and hi90 < margin:
        return NOT_ADD
    if hi95 < -margin:
        return WORSE
    return INCONCLUSIVE


def score_rows(results: list[OodResult]) -> list[Row]:
    """AUROC and FPR@95TPR per lambda, OOD set and score (seed means, 95% CIs)."""
    rows: list[Row] = []
    for r in results:
        for score in SCORES:
            row: Row = {
                "lambda": r.lam,
                "OOD set": r.ood_set,
                "score": score,
                "AUROC [95% CI]": _fmt(r.auroc[score]),
                "FPR@95TPR [95% CI]": _fmt(r.fpr[score]),
            }
            if score in (RADIUS, DEVIATION):
                row["reading (pre-registered)"] = radius_verdict(r.auroc[score])
            rows.append(row)
    return rows


# (name, minuend, subtrahend): differences of AUROC reported with a verdict.
DIFFERENCES = (
    ("direction + radius  minus  direction only", COMBINED, KNN_DIRECTION),
    ("best-weight signed radius  minus  direction only", ORACLE, KNN_DIRECTION),
    ("raw kNN  minus  direction only", KNN_RAW, KNN_DIRECTION),
    ("raw kNN  minus  raw kNN with radius permuted", KNN_RAW, KNN_RAW_PERMUTED),
    (
        "control: direction + permuted radius  minus  direction only",
        COMBINED_PERMUTED,
        KNN_DIRECTION,
    ),
)


def difference_rows(results: list[OodResult]) -> list[Row]:
    """AUROC differences with the pre-registered verdict (margin 0.01)."""
    rows: list[Row] = []
    for r in results:
        for label, plus, minus in DIFFERENCES:
            diff = r.auroc[plus] - r.auroc[minus]
            rows.append(
                {
                    "lambda": r.lam,
                    "OOD set": r.ood_set,
                    "comparison": label,
                    "AUROC difference [95% CI]": _fmt(diff),
                    "90% CI": "[{:+.3f}, {:+.3f}]".format(*_ci(diff, 0.90)),
                    "verdict": addition_verdict(diff),
                }
            )
    return rows


def oracle_rows(results: list[OodResult]) -> list[Row]:
    """Which weight of the optimistic combination won on each seed."""
    return [
        {
            "lambda": r.lam,
            "OOD set": r.ood_set,
            "best weight per seed": ", ".join(f"{w:g}" for w in r.oracle_weights),
            "images (ID / OOD)": f"{r.num_id} / {r.num_ood}",
        }
        for r in results
    ]
