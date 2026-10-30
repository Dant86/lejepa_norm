"""Within-image regression of the view radius on the view's augmentation parameters.

Pre-registered in `research_notes/radius_aug_regression/preregistration.md`. Rows are
(image, augmented view) pairs; the outcome is `y = log ||z||`. The primary model has
an image fixed effect, `y_iv = alpha_i + beta' x_iv + eps_iv`, fitted by within-image
demeaning, so `beta` is identified from variation across the views of one image.

All fits use per-image sufficient statistics (`WithinStats`: Gram matrix, cross
product, sum of squares of the demeaned data), so a bootstrap over images, which
resamples all views of an image together, is a weighted sum of those statistics and
costs almost nothing. Part C asks whether the radius carries information about the
view beyond its augmentation parameters (`analyze_view_quality`).
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import dataclasses
import typing

import numpy
import numpy.typing
from sklearn import ensemble
import threadpoolctl

from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
BoolArray = numpy.typing.NDArray[numpy.bool_]

# Knots of the restricted cubic spline of log crop area, as quantiles of the rows.
KNOT_QUANTILES = (0.05, 0.35, 0.65, 0.95)
COLUMN_NAMES = (
    "log_area",
    "log_area_spline_1",
    "log_area_spline_2",
    "log_aspect",
    "center_offset",
    "log_area_x_blur",
    "flip",
    "jitter",
    "brightness",
    "contrast",
    "saturation",
    "abs_hue",
    "grayscale",
    "blur",
    "sigma",
)
# Columns of each augmentation type. The area x blur interaction (column 5) belongs
# to both the crop and the blur block: removing either removes it.
BLOCKS: dict[str, tuple[int, ...]] = {
    "crop": (0, 1, 2, 3, 4, 5),
    "flip": (6,),
    "color": (7, 8, 9, 10, 11),
    "grayscale": (12,),
    "blur": (5, 13, 14),
}
SPLINE_COLUMNS = (0, 1, 2)
# OpenMP threads of the boosted trees. Uncapped, scikit-learn uses every core of the
# node, which on a shared node is orders of magnitude slower than a few threads.
TREE_THREADS = 4


def _column(aug: FloatArray, name: str) -> FloatArray:
    return aug[..., shards.AUG_FIELDS.index(name)]


def rcs_basis(x: FloatArray, knots: FloatArray) -> FloatArray:
    """Restricted (natural) cubic spline basis with 4 knots: `[x, s_1, s_2]`.

    Linear beyond the outer knots; the nonlinear terms follow Harrell's
    parametrization, scaled by `(t_4 - t_1)^2`.
    """
    t = knots
    last, before = t[-1], t[-2]

    def cube(v: FloatArray) -> FloatArray:
        return numpy.maximum(v, 0.0) ** 3

    columns = [x]
    for j in range(len(t) - 2):
        term = (
            cube(x - t[j])
            - cube(x - before) * (last - t[j]) / (last - before)
            + cube(x - last) * (before - t[j]) / (last - before)
        )
        columns.append(term / (last - t[0]) ** 2)
    return numpy.stack(columns, axis=-1)


@dataclasses.dataclass(frozen=True)
class Design:
    """Standardized covariates of every (image, view) row.

    Attributes:
        x: `(N, K, P)` standardized covariates (`COLUMN_NAMES`).
        raw: `(N, K, P)` the same covariates before standardization.
        mean: `(P,)` pooled mean of `raw`.
        std: `(P,)` pooled standard deviation of `raw` (1 where it is 0).
        knots: Spline knots of log area.
        log_area_mean: Mean log area, used to center the area x blur interaction.
    """

    x: FloatArray
    raw: FloatArray
    mean: FloatArray
    std: FloatArray
    knots: FloatArray
    log_area_mean: float


def build_design(aug: FloatArray, height: int, width: int) -> Design:
    """Covariates from the saved augmentation parameters `(N, K, len(AUG_FIELDS))`.

    Jitter factors are set to their neutral values where the jitter gate is off, and
    blur sigma to 0 where the blur gate is off (sigma is drawn for every view).
    """
    aug = aug.astype(numpy.float64)
    crop_h, crop_w = _column(aug, "height"), _column(aug, "width")
    log_area = numpy.log(crop_h * crop_w / (height * width))
    center_y = _column(aug, "top") + crop_h / 2 - height / 2
    center_x = _column(aug, "left") + crop_w / 2 - width / 2
    offset = numpy.hypot(center_y / height, center_x / width)
    jitter = _column(aug, "jitter") > 0
    blur = _column(aug, "blur") > 0
    knots = numpy.quantile(log_area, KNOT_QUANTILES)
    area_mean = float(log_area.mean())
    spline = rcs_basis(log_area, knots)
    columns = [
        *(spline[..., j] for j in range(spline.shape[-1])),
        numpy.log(crop_w / crop_h),
        offset,
        (log_area - area_mean) * blur,
        _column(aug, "flip"),
        jitter.astype(numpy.float64),
        numpy.where(jitter, _column(aug, "brightness"), 1.0),
        numpy.where(jitter, _column(aug, "contrast"), 1.0),
        numpy.where(jitter, _column(aug, "saturation"), 1.0),
        numpy.where(jitter, numpy.abs(_column(aug, "hue")), 0.0),
        _column(aug, "grayscale"),
        blur.astype(numpy.float64),
        numpy.where(blur, _column(aug, "sigma"), 0.0),
    ]
    raw = numpy.stack(columns, axis=-1)
    flat = raw.reshape(-1, raw.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = numpy.where(std > 0, std, 1.0)
    return Design((raw - mean) / std, raw, mean, std, knots, area_mean)


def area_curve(design: Design, beta: FloatArray, grid: FloatArray) -> FloatArray:
    """Fitted change of `log ||z||` along `grid`, a range of log crop areas.

    The curve is 0 at the first grid point and holds blur off.
    """
    basis = rcs_basis(grid, design.knots)
    cols = list(SPLINE_COLUMNS)
    standardized = (basis - design.mean[cols]) / design.std[cols]
    effect = standardized @ beta[cols]
    return effect - effect[0]


def demean(a: FloatArray) -> FloatArray:
    """Subtracts the per-image mean over views (axis 1)."""
    return a - a.mean(axis=1, keepdims=True)


@dataclasses.dataclass(frozen=True)
class WithinStats:
    """Per-image sufficient statistics of the demeaned regression.

    Attributes:
        gram: `(N, P, P)` per-image `X'X`.
        cross: `(N, P)` per-image `X'y`.
        syy: `(N,)` per-image `y'y`.
    """

    gram: FloatArray
    cross: FloatArray
    syy: FloatArray


def within_stats(xd: FloatArray, yd: FloatArray) -> WithinStats:
    """Statistics of demeaned covariates `(N, K, P)` and outcome `(N, K)`."""
    return WithinStats(
        numpy.einsum("nkp,nkq->npq", xd, xd),
        numpy.einsum("nkp,nk->np", xd, yd),
        numpy.einsum("nk,nk->n", yd, yd),
    )


def fit(
    stats: WithinStats, weights: FloatArray, cols: typing.Sequence[int]
) -> tuple[FloatArray, float]:
    """Weighted within-image OLS on columns `cols`: coefficients and within `R^2`.

    `weights` are per-image (ones, bootstrap counts, or 0/1 fold indicators).
    """
    index = list(cols)
    n, p = stats.cross.shape
    gram = (weights @ stats.gram.reshape(n, p * p)).reshape(p, p)[
        numpy.ix_(index, index)
    ]
    cross = (weights @ stats.cross)[index]
    beta = numpy.linalg.lstsq(gram, cross, rcond=None)[0]
    total = float(weights @ stats.syy)
    sse = total - 2 * float(beta @ cross) + float(beta @ gram @ beta)
    return beta, 1.0 - sse / total


def _sse(
    stats: WithinStats,
    weights: FloatArray,
    beta: FloatArray,
    cols: typing.Sequence[int],
) -> float:
    index = list(cols)
    n, p = stats.cross.shape
    gram = (weights @ stats.gram.reshape(n, p * p)).reshape(p, p)[
        numpy.ix_(index, index)
    ]
    cross = (weights @ stats.cross)[index]
    return (
        float(weights @ stats.syy) - 2 * float(beta @ cross) + float(beta @ gram @ beta)
    )


def image_folds(n: int, num_folds: int, rng: numpy.random.Generator) -> IntArray:
    """Random assignment of `n` images to `num_folds` folds."""
    return rng.permutation(n) % num_folds


def cv_r2(
    stats: WithinStats, cols: typing.Sequence[int], folds: IntArray, num_folds: int
) -> float:
    """Held-out within-image `R^2` of the linear model (image-grouped folds)."""
    sse = 0.0
    total = 0.0
    for f in range(num_folds):
        train = (folds != f).astype(numpy.float64)
        test = (folds == f).astype(numpy.float64)
        beta, _ = fit(stats, train, cols)
        sse += _sse(stats, test, beta, cols)
        total += float(test @ stats.syy)
    return 1.0 - sse / total


def cross_fit(
    xd: FloatArray,
    yd: FloatArray,
    folds: IntArray,
    num_folds: int,
    cols: typing.Sequence[int],
) -> FloatArray:
    """Held-out predictions `(N, K)` of `yd` from `xd[..., cols]` (demeaned)."""
    stats = within_stats(xd, yd)
    out = numpy.zeros_like(yd)
    index = list(cols)
    for f in range(num_folds):
        beta, _ = fit(stats, (folds != f).astype(numpy.float64), index)
        test = folds == f
        out[test] = xd[test][..., index] @ beta
    return out


def tree_cv_r2(
    x: FloatArray,
    xd: FloatArray,
    yd: FloatArray,
    folds: IntArray,
    num_folds: int,
    seed: int,
) -> float:
    """Held-out within-image `R^2` of gradient-boosted trees on `x` and `xd`.

    Predictions are demeaned within image (the image effect absorbs their mean), so
    the number is comparable to `cv_r2`.
    """
    with threadpoolctl.threadpool_limits(limits=TREE_THREADS):
        features = numpy.concatenate([x, xd], axis=-1)
        k, width = features.shape[1:]
        sse = 0.0
        total = 0.0
        for f in range(num_folds):
            train, test = folds != f, folds == f
            model = ensemble.HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=seed
            )
            model.fit(features[train].reshape(-1, width), yd[train].reshape(-1))
            pred = numpy.asarray(
                model.predict(features[test].reshape(-1, width))
            ).reshape(-1, k)
            sse += float(((yd[test] - demean(pred)) ** 2).sum())
            total += float((yd[test] ** 2).sum())
    return 1.0 - sse / total


def percentile_interval(draws: FloatArray, level: float = 0.95) -> tuple[float, float]:
    """Central percentile interval of bootstrap draws."""
    tail = 100 * (1 - level) / 2
    low, high = numpy.percentile(draws, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_weights(
    n: int, num_bootstrap: int, rng: numpy.random.Generator
) -> list[FloatArray]:
    """Image-resampling weights: multinomial counts over `n` images."""
    return [
        rng.multinomial(n, numpy.full(n, 1.0 / n)).astype(numpy.float64)
        for _ in range(num_bootstrap)
    ]


@dataclasses.dataclass(frozen=True)
class RegressionResult:
    """Part A/B results of one checkpoint.

    Attributes:
        beta: Standardized coefficients (`COLUMN_NAMES`), point estimate.
        beta_interval: `(P, 2)` 95% bootstrap interval of `beta`.
        within_r2: In-sample within-image `R^2` of the linear model and its interval.
        cv_r2_linear: Held-out within-image `R^2` of the linear model.
        cv_r2_trees: Held-out within-image `R^2` of the tree ceiling.
        within_share: Share of the total variance of `y` that is within-image.
        block_drop: Per block, drop in within `R^2` when it is removed, with 95%
            interval.
        curve_grid: Log-area grid of the effect curve.
        curve: Fitted effect of log area on `y` at those points (blur off).
        slope_sd: Estimated sd of the image-specific log-area slope (two-stage,
            sampling variance removed); nan if the corrected variance is negative.
        slope_alpha_corr: Correlation of the per-image slope with the image mean of
            `y` (attenuated by the noise of the slopes).
    """

    beta: FloatArray
    beta_interval: FloatArray
    within_r2: tuple[float, float, float]
    cv_r2_linear: float
    cv_r2_trees: float
    within_share: float
    block_drop: dict[str, tuple[float, float, float]]
    curve_grid: FloatArray
    curve: FloatArray
    slope_sd: float
    slope_alpha_corr: float


def analyze_regression(
    y: FloatArray,
    design: Design,
    num_bootstrap: int,
    num_folds: int,
    rng: numpy.random.Generator,
    seed: int,
) -> RegressionResult:
    """Fits the primary model and its summaries on `y` `(N, K)`."""
    n = y.shape[0]
    p = design.x.shape[-1]
    all_cols = list(range(p))
    yd = demean(y)
    xd = demean(design.x)
    stats = within_stats(xd, yd)
    ones = numpy.ones(n)
    beta, r2 = fit(stats, ones, all_cols)

    reduced = {
        name: [c for c in all_cols if c not in cols] for name, cols in BLOCKS.items()
    }
    drops_point = {
        name: r2 - fit(stats, ones, keep)[1] for name, keep in reduced.items()
    }
    beta_draws: list[FloatArray] = []
    r2_draws: list[float] = []
    drop_draws: dict[str, list[float]] = {name: [] for name in BLOCKS}
    for w in bootstrap_weights(n, num_bootstrap, rng):
        b, r = fit(stats, w, all_cols)
        beta_draws.append(b)
        r2_draws.append(r)
        for name, keep in reduced.items():
            drop_draws[name].append(r - fit(stats, w, keep)[1])
    beta_stack = numpy.stack(beta_draws)
    interval = numpy.array(
        [percentile_interval(beta_stack[:, j]) for j in range(p)], dtype=numpy.float64
    )
    r2_low, r2_high = percentile_interval(numpy.asarray(r2_draws))
    block_drop = {
        name: (drops_point[name], *percentile_interval(numpy.asarray(d)))
        for name, d in drop_draws.items()
    }

    folds = image_folds(n, num_folds, rng)
    total_var = float(((y - y.mean()) ** 2).sum())
    grid = numpy.linspace(float(design.knots[0]), float(design.raw[..., 0].max()), 25)

    # Image-specific slope on centered log area (two-stage; see the note).
    area = demean(design.raw[..., 0])
    area_ss = (area**2).sum(axis=1)
    slope = (area * yd).sum(axis=1) / area_ss
    residual = yd - xd @ beta
    sigma2 = float((residual**2).sum()) / (yd.size - n - p)
    noise = float((sigma2 / area_ss).mean())
    true_var = float(slope.var()) - noise
    corr = float(numpy.corrcoef(slope, y.mean(axis=1))[0, 1])
    return RegressionResult(
        beta=beta,
        beta_interval=interval,
        within_r2=(r2, r2_low, r2_high),
        cv_r2_linear=cv_r2(stats, all_cols, folds, num_folds),
        cv_r2_trees=tree_cv_r2(design.x, xd, yd, folds, num_folds, seed),
        within_share=float(stats.syy.sum()) / total_var,
        block_drop=block_drop,
        curve_grid=grid,
        curve=area_curve(design, beta, grid),
        slope_sd=float(numpy.sqrt(true_var)) if true_var > 0 else float("nan"),
        slope_alpha_corr=corr,
    )


def angle_to_mean_direction(z: FloatArray) -> FloatArray:
    """Angle (radians) of each view's direction to the mean direction of its image.

    `z` is `(N, K, d)`; the mean is over the `K` views.
    """
    unit = z / numpy.linalg.norm(z, axis=-1, keepdims=True)
    mean = unit.mean(axis=1)
    mean /= numpy.linalg.norm(mean, axis=-1, keepdims=True)
    cos = numpy.einsum("nkd,nd->nk", unit, mean)
    return numpy.arccos(numpy.clip(cos, -1.0, 1.0))


def within_auc_counts(
    score: FloatArray, positive: BoolArray
) -> tuple[FloatArray, FloatArray]:
    """Per image: wins and pairs of (positive view, negative view) comparisons.

    A win is a positive view with a higher score than a negative view of the same
    image (ties count 1/2). `score`, `positive` are `(N, K)`.
    """
    pair = positive[:, :, None] & ~positive[:, None, :]
    wins = (score[:, :, None] > score[:, None, :]) * 1.0
    wins += (score[:, :, None] == score[:, None, :]) * 0.5
    return (wins * pair).sum(axis=(1, 2)), pair.sum(axis=(1, 2)).astype(numpy.float64)


@dataclasses.dataclass(frozen=True)
class ViewQualityResult:
    """Part C results of one checkpoint.

    Attributes:
        correct_auc: Within-image AUC (with 95% interval) of the scores `yhat`,
            `r` and `both` for view correctness.
        both_weights: Coefficients of `[yhat, r]` in the `both` score.
        retains: `r` keeps more than half of the way from 0.5 to the AUC of `both`.
        angle_r2: Within `R^2` of the angle to the mean direction on `yhat`, `r` and
            both (each with a 95% interval), and the `r` increment over `yhat`.
        angle_beta: Standardized coefficients of `[yhat, r]` for the angle.
    """

    correct_auc: dict[str, tuple[float, float, float]]
    both_weights: FloatArray
    retains: bool
    angle_r2: dict[str, tuple[float, float, float]]
    angle_beta: FloatArray


def _standardize(a: FloatArray) -> FloatArray:
    return a / a.std()


def analyze_view_quality(
    yd: FloatArray,
    xd: FloatArray,
    correct: BoolArray,
    angle: FloatArray,
    num_bootstrap: int,
    num_folds: int,
    rng: numpy.random.Generator,
) -> ViewQualityResult:
    """Does the radius residual `r` predict view quality beyond the parameters?

    `yhat` is the cross-fitted prediction of the demeaned radius from the
    parameters and `r = yd - yhat`. Both are standardized; outcomes get image fixed
    effects.
    """
    n = yd.shape[0]
    folds = image_folds(n, num_folds, rng)
    yhat = cross_fit(xd, yd, folds, num_folds, list(range(xd.shape[-1])))
    resid = yd - yhat
    pair = numpy.stack([_standardize(yhat), _standardize(resid)], axis=-1)
    weights = bootstrap_weights(n, num_bootstrap, rng)

    # Correctness: linear-probability model with image effects gives the combined
    # score; AUCs use per-image pair counts so the bootstrap is a weighted sum.
    stats_c = within_stats(pair, demean(correct.astype(numpy.float64)))
    both, _ = fit(stats_c, numpy.ones(n), [0, 1])
    scores = {
        "yhat": yhat,
        "r": resid,
        "both": pair @ both,
    }
    auc: dict[str, tuple[float, float, float]] = {}
    for name, s in scores.items():
        wins, pairs = within_auc_counts(s, correct)
        point = float(wins.sum() / pairs.sum())
        draws = numpy.asarray([w @ wins / (w @ pairs) for w in weights])
        auc[name] = (point, *percentile_interval(draws))
    retains = auc["r"][0] - 0.5 > 0.5 * (auc["both"][0] - 0.5)

    # Angle to the mean direction.
    stats_a = within_stats(pair, demean(angle))
    angle_beta, _ = fit(stats_a, numpy.ones(n), [0, 1])
    models = {"yhat": [0], "r": [1], "both": [0, 1]}
    angle_r2: dict[str, tuple[float, float, float]] = {}
    draws_by: dict[str, list[float]] = {k: [] for k in models}
    for w in weights:
        for name, cols in models.items():
            draws_by[name].append(fit(stats_a, w, cols)[1])
    for name, cols in models.items():
        point = fit(stats_a, numpy.ones(n), cols)[1]
        angle_r2[name] = (point, *percentile_interval(numpy.asarray(draws_by[name])))
    increment = numpy.asarray(draws_by["both"]) - numpy.asarray(draws_by["yhat"])
    angle_r2["r_increment"] = (
        angle_r2["both"][0] - angle_r2["yhat"][0],
        *percentile_interval(increment),
    )
    return ViewQualityResult(auc, both, bool(retains), angle_r2, angle_beta)
