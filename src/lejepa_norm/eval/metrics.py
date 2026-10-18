"""Scalar statistics on embeddings: kappa, ICC, rank correlations, AURC, AUROC.

All functions are pure NumPy, deterministic given a `numpy.random.Generator`, and
agnostic to whether their inputs come from `z` or `h`. Confidence intervals are
percentile bootstraps over images (the unit of independent sampling).
"""

import collections.abc
import dataclasses
import typing

import numpy
import numpy.typing

FloatArray = numpy.typing.NDArray[numpy.float64]
ArrayLike = numpy.typing.ArrayLike


@dataclasses.dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap percentile confidence interval.

    Attributes:
        value: Estimate on the full sample.
        low: Lower CI bound (NaN if no bootstrap was run).
        high: Upper CI bound (NaN if no bootstrap was run).
        n: Number of independent units (images) used.
    """

    value: float
    low: float
    high: float
    n: int


def bootstrap(
    statistic: collections.abc.Callable[[numpy.typing.NDArray[numpy.intp]], float],
    n: int,
    num_bootstrap: int,
    rng: numpy.random.Generator,
    level: float = 0.95,
) -> Estimate:
    """Percentile bootstrap of `statistic` over `n` resampled units.

    Args:
        statistic: Maps an index array (into the `n` units) to a scalar.
        n: Number of units.
        num_bootstrap: Number of resamples; 0 skips the CI.
        rng: Random generator.
        level: Two-sided confidence level.

    Returns:
        The full-sample estimate with its CI. Resamples on which the statistic
        is not finite are dropped.
    """
    value = float(statistic(numpy.arange(n)))
    if num_bootstrap <= 0:
        return Estimate(value, float("nan"), float("nan"), n)
    draws = numpy.array(
        [statistic(rng.integers(0, n, size=n)) for _ in range(num_bootstrap)]
    )
    draws = draws[numpy.isfinite(draws)]
    if draws.size == 0:
        return Estimate(value, float("nan"), float("nan"), n)
    tail = (1.0 - level) / 2.0
    low, high = numpy.quantile(draws, [tail, 1.0 - tail])
    return Estimate(value, float(low), float(high), n)


def kappa(x: ArrayLike) -> float:
    """Normalized radius dispersion `Var(||x||^2) / (2 d)`.

    Equals 1 in expectation under `N(0, I_d)` (where `||x||^2 ~ chi^2_d`) and 0
    when all samples share one radius.

    Args:
        x: Samples of shape `(..., d)`; all leading axes are pooled.

    Returns:
        The statistic (unbiased variance).
    """
    arr = numpy.asarray(x, dtype=numpy.float64)
    sq = numpy.square(arr).sum(axis=-1).ravel()
    return float(sq.var(ddof=1) / (2.0 * arr.shape[-1]))


@dataclasses.dataclass(frozen=True)
class IccResult:
    """One-way random-effects ICC(1) with its ANOVA mean squares.

    Attributes:
        icc: Point estimate with bootstrap CI over images.
        msb: Between-image mean square.
        msw: Within-image mean square.
        num_views: Views per image `K`.
    """

    icc: Estimate
    msb: float
    msw: float
    num_views: int


def _icc_parts(y: FloatArray) -> tuple[float, float, float]:
    n, k = y.shape
    means = y.mean(axis=1)
    msb = k * float(numpy.square(means - means.mean()).sum()) / (n - 1)
    msw = float(numpy.square(y - means[:, None]).sum()) / (n * (k - 1))
    denom = msb + (k - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else float("nan")
    return icc, msb, msw


def icc1(
    y: ArrayLike,
    num_bootstrap: int = 0,
    rng: numpy.random.Generator | None = None,
) -> IccResult:
    """One-way random-effects intraclass correlation ICC(1).

    `ICC(1) = (MSB - MSW) / (MSB + (K - 1) MSW)`, estimating
    `Var(E[y | image]) / Var(y)` from K exchangeable views per image.

    Args:
        y: `(N, K)` per-view values (e.g. log-radius of the augmented views).
        num_bootstrap: Bootstrap resamples over images for the CI.
        rng: Random generator (required if `num_bootstrap > 0`).

    Returns:
        The ICC with CI and the mean squares.

    Raises:
        ValueError: Fewer than 2 images or 2 views.
    """
    arr = numpy.asarray(y, dtype=numpy.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("ICC needs an (N, K) array with N >= 2 and K >= 2")
    _, msb, msw = _icc_parts(arr)
    est = bootstrap(
        lambda idx: _icc_parts(arr[idx])[0],
        arr.shape[0],
        num_bootstrap,
        rng if rng is not None else numpy.random.default_rng(0),
    )
    return IccResult(icc=est, msb=msb, msw=msw, num_views=arr.shape[1])


def rankdata(x: ArrayLike) -> FloatArray:
    """Ranks starting at 1, with ties given their average rank."""
    arr = numpy.asarray(x, dtype=numpy.float64).ravel()
    n = arr.size
    order = numpy.argsort(arr, kind="stable")
    sorted_x = arr[order]
    is_new = numpy.concatenate([[True], sorted_x[1:] != sorted_x[:-1]])
    starts = numpy.flatnonzero(is_new)
    ends = numpy.append(starts[1:], n)
    average = (starts + ends - 1) / 2.0 + 1.0
    ranks = numpy.empty(n, dtype=numpy.float64)
    ranks[order] = average[numpy.cumsum(is_new) - 1]
    return ranks


def _pearson(a: FloatArray, b: FloatArray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(numpy.sqrt(numpy.dot(a, a) * numpy.dot(b, b)))
    return float(numpy.dot(a, b)) / denom if denom > 0 else float("nan")


def _remove_group_means(
    x: FloatArray, groups: numpy.typing.NDArray[typing.Any]
) -> FloatArray:
    _, inverse = numpy.unique(groups, return_inverse=True)
    sums = numpy.bincount(inverse, weights=x)
    counts = numpy.bincount(inverse)
    return x - (sums / counts)[inverse]


def spearman(
    x: ArrayLike,
    y: ArrayLike,
    num_bootstrap: int = 0,
    rng: numpy.random.Generator | None = None,
) -> Estimate:
    """Spearman rank correlation with a bootstrap CI over paired samples."""
    xa = numpy.asarray(x, dtype=numpy.float64).ravel()
    ya = numpy.asarray(y, dtype=numpy.float64).ravel()
    return bootstrap(
        lambda idx: _pearson(rankdata(xa[idx]), rankdata(ya[idx])),
        xa.size,
        num_bootstrap,
        rng if rng is not None else numpy.random.default_rng(0),
    )


def partial_spearman(
    x: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    num_bootstrap: int = 0,
    rng: numpy.random.Generator | None = None,
) -> Estimate:
    """Spearman correlation of `x` and `y` controlling for a categorical variable.

    Both variables are rank-transformed, the per-group mean rank is removed from
    each (i.e. the ranks are regressed on the group one-hot), and the Pearson
    correlation of the residuals is returned.

    Args:
        x: First variable, `(N,)`.
        y: Second variable, `(N,)`.
        groups: Categorical control (e.g. class label), `(N,)`.
        num_bootstrap: Bootstrap resamples over samples for the CI.
        rng: Random generator.

    Returns:
        The partial rank correlation with CI.
    """
    xa = numpy.asarray(x, dtype=numpy.float64).ravel()
    ya = numpy.asarray(y, dtype=numpy.float64).ravel()
    ga = numpy.asarray(groups).ravel()

    def stat(idx: numpy.typing.NDArray[numpy.intp]) -> float:
        g = ga[idx]
        return _pearson(
            _remove_group_means(rankdata(xa[idx]), g),
            _remove_group_means(rankdata(ya[idx]), g),
        )

    return bootstrap(
        stat,
        xa.size,
        num_bootstrap,
        rng if rng is not None else numpy.random.default_rng(0),
    )


def aurc(confidence: ArrayLike, errors: ArrayLike) -> float:
    """Area under the risk-coverage curve for selective classification.

    Samples are accepted in order of decreasing confidence; the risk at coverage
    `k / N` is the error rate among the `k` most confident samples, and AURC is
    the mean risk over `k = 1..N`. Lower is better; a confidence score unrelated
    to correctness gives about the overall error rate.

    Args:
        confidence: `(N,)` scores, higher means more confident.
        errors: `(N,)` booleans, True where the prediction is wrong.

    Returns:
        The AURC in `[0, 1]`.
    """
    conf = numpy.asarray(confidence, dtype=numpy.float64).ravel()
    err = numpy.asarray(errors, dtype=numpy.float64).ravel()
    order = numpy.argsort(-conf, kind="stable")
    risk = numpy.cumsum(err[order]) / numpy.arange(1, err.size + 1)
    return float(risk.mean())


def auroc(scores: ArrayLike, positive: ArrayLike) -> float:
    """Area under the ROC curve (Mann-Whitney U), ties counted as one half.

    Args:
        scores: `(N,)` scores; higher should indicate the positive class.
        positive: `(N,)` booleans marking the positive class.

    Returns:
        P(score of a random positive > score of a random negative).

    Raises:
        ValueError: Either class is empty.
    """
    s = numpy.asarray(scores, dtype=numpy.float64).ravel()
    pos = numpy.asarray(positive, dtype=bool).ravel()
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both positive and negative samples")
    rank_sum = float(rankdata(s)[pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def retrieval_map(features: ArrayLike, labels: ArrayLike) -> float:
    """Leave-one-out cosine retrieval mean average precision.

    Each sample queries all others ranked by cosine similarity; relevant items
    share its label.

    Args:
        features: `(N, d)` features.
        labels: `(N,)` labels.

    Returns:
        mAP over queries that have at least one relevant item.
    """
    f = numpy.asarray(features, dtype=numpy.float64)
    f = f / numpy.maximum(numpy.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    lab = numpy.asarray(labels).ravel()
    sims = f @ f.T
    numpy.fill_diagonal(sims, -numpy.inf)
    order = numpy.argsort(-sims, axis=1, kind="stable")[:, :-1]
    relevant = (lab[order] == lab[:, None]).astype(numpy.float64)
    hits = relevant.sum(axis=1)
    ranks = numpy.arange(1, order.shape[1] + 1, dtype=numpy.float64)
    precision: FloatArray = numpy.cumsum(relevant, axis=1) / ranks
    ap: FloatArray = (precision * relevant).sum(axis=1) / numpy.maximum(hits, 1)
    return float(ap[hits > 0].mean())
