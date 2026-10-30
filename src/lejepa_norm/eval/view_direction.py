"""Is the within-image radius view direction or crop content? (`view_direction`).

Pre-registered in `research_notes/radius_view_direction/preregistration.md`. Extends
the within-image fixed-effects regression of `aug_regression` with two blocks of
covariates: the view's direction (its angle to the mean direction of the image's
*other* views, and a cross-fitted ridge prediction of the radius from the direction)
and crop content (luminance, contrast, edge energy and color shift of the cropped
pixels). Each model is scored by its held-out within-image `R^2`, and the radius left
over after each model is scored for predicting the probe's correctness on the view.
"""

# pyright: reportUnknownVariableType=false

import dataclasses

import numpy
import numpy.typing

from lejepa_norm.eval import aug_regression
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
BoolArray = numpy.typing.NDArray[numpy.bool_]

DIRECTION_COLUMNS = ("angle_to_others", "direction_fit")
CONTENT_COLUMNS = ("luminance", "luminance_sd", "gradient", "color_shift")
# Ridge penalties, as multiples of the mean diagonal of the Gram matrix.
RIDGE_PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1)
CHUNK = 500  # images per chunk of the integral-image content statistics


def unit(z: FloatArray) -> FloatArray:
    """Directions `z / ||z||`."""
    return z / numpy.linalg.norm(z, axis=-1, keepdims=True)


def angle_to_others(z: FloatArray) -> FloatArray:
    """Angle (radians) of each view to the mean direction of the other views.

    `z` is `(N, K, d)`. Leaving the view out keeps its own direction out of its own
    covariate.
    """
    u = unit(z)
    others: FloatArray = (u.sum(axis=1, keepdims=True) - u) / (u.shape[1] - 1)
    others = others / numpy.linalg.norm(others, axis=-1, keepdims=True)
    cos = numpy.einsum("nkd,nkd->nk", u, others)
    return numpy.arccos(numpy.clip(cos, -1.0, 1.0))


@dataclasses.dataclass(frozen=True)
class Spread:
    """Angles (degrees) between view directions.

    Attributes:
        within_mean: Mean angle between two views of one image.
        within_p90: 90th percentile of the same.
        across_mean: Mean angle between views of two different images (same view
            slot, images randomly paired).
        to_mean: Mean angle of a view to its image's mean direction.
    """

    within_mean: float
    within_p90: float
    across_mean: float
    to_mean: float


def direction_spread(z: FloatArray, rng: numpy.random.Generator) -> Spread:
    """How far apart the views of one image are, against views of other images."""
    u = unit(z)
    n, k, _ = u.shape
    gram = numpy.einsum("nkd,nld->nkl", u, u)
    rows, cols = numpy.triu_indices(k, 1)
    within = numpy.degrees(numpy.arccos(numpy.clip(gram[:, rows, cols], -1, 1)))
    other = u[rng.permutation(n)]
    across = numpy.degrees(
        numpy.arccos(numpy.clip(numpy.einsum("nkd,nkd->nk", u, other), -1, 1))
    )
    return Spread(
        within_mean=float(within.mean()),
        within_p90=float(numpy.percentile(within, 90)),
        across_mean=float(across.mean()),
        to_mean=float(numpy.degrees(aug_regression.angle_to_mean_direction(z)).mean()),
    )


def _ridge_fit(x: FloatArray, y: FloatArray, penalty: float) -> FloatArray:
    gram = x.T @ x
    ridge = penalty * float(numpy.trace(gram)) / gram.shape[0]
    penalized = gram + ridge * numpy.eye(gram.shape[0])
    return numpy.linalg.lstsq(penalized, x.T @ y, rcond=None)[0]


def direction_fit(
    z: FloatArray, yd: FloatArray, folds: IntArray, num_folds: int
) -> tuple[FloatArray, float, float]:
    """Cross-fitted ridge prediction `(N, K)` of the demeaned radius from direction.

    The inputs are the within-image demeaned unit directions, so only the variation
    between views of an image is used. The penalty is the one of `RIDGE_PENALTIES`
    with the best held-out within-image `R^2`. Returns the prediction, the penalty
    and that `R^2`.
    """
    ud = aug_regression.demean(unit(z))
    d = ud.shape[-1]
    best: tuple[float, float, FloatArray] | None = None
    for penalty in RIDGE_PENALTIES:
        pred = numpy.zeros_like(yd)
        for f in range(num_folds):
            train, test = folds != f, folds == f
            beta = _ridge_fit(ud[train].reshape(-1, d), yd[train].reshape(-1), penalty)
            pred[test] = ud[test] @ beta
        r2 = 1.0 - float(((yd - pred) ** 2).sum()) / float((yd**2).sum())
        if best is None or r2 > best[0]:
            best = (r2, penalty, pred)
    assert best is not None
    return best[2], best[1], best[0]


def _box_sums(
    integral: FloatArray,
    n: IntArray,
    r0: IntArray,
    r1: IntArray,
    c0: IntArray,
    c1: IntArray,
) -> FloatArray:
    return (
        integral[n, r1, c1]
        - integral[n, r0, c1]
        - integral[n, r1, c0]
        + integral[n, r0, c0]
    )


def _integral(a: FloatArray) -> FloatArray:
    """Zero-padded 2-D cumulative sum over the last two axes of `(n, H, W)`."""
    out = numpy.zeros((a.shape[0], a.shape[1] + 1, a.shape[2] + 1))
    out[:, 1:, 1:] = a.cumsum(axis=1).cumsum(axis=2)
    return out


def crop_content(images: numpy.ndarray, aug: FloatArray) -> FloatArray:  # pyright: ignore[reportMissingTypeArgument]
    """Content statistics `(N, K, 4)` of the cropped pixels of every view.

    `images` is the `(N, H, W, 3)` uint8 source images and `aug` the saved
    parameters `(N, K, len(AUG_FIELDS))`. Statistics of the crop box on the raw
    pixels (before color jitter, blur and flip): mean luminance, luminance sd, mean
    gradient magnitude, and the distance of the crop's mean color from the image's.
    """
    n_images, height, width = images.shape[:3]
    k = aug.shape[1]
    top, left, box_h, box_w = (
        aug[..., shards.AUG_FIELDS.index(name)]
        for name in ("top", "left", "height", "width")
    )
    r0 = numpy.clip(numpy.rint(top), 0, height - 1).astype(numpy.int64)
    c0 = numpy.clip(numpy.rint(left), 0, width - 1).astype(numpy.int64)
    r1 = numpy.clip(numpy.rint(top + box_h), r0 + 1, height).astype(numpy.int64)
    c1 = numpy.clip(numpy.rint(left + box_w), c0 + 1, width).astype(numpy.int64)
    out = numpy.zeros((n_images, k, len(CONTENT_COLUMNS)))
    for start in range(0, n_images, CHUNK):
        sl = slice(start, min(start + CHUNK, n_images))
        rgb = images[sl].astype(numpy.float64) / 255.0
        lum = rgb @ numpy.array([0.299, 0.587, 0.114])
        grad_y = numpy.abs(numpy.diff(lum, axis=1, append=lum[:, -1:]))
        grad_x = numpy.abs(numpy.diff(lum, axis=2, append=lum[:, :, -1:]))
        grad = numpy.hypot(grad_x, grad_y)
        planes = [_integral(a) for a in (lum, lum**2, grad, *numpy.moveaxis(rgb, 3, 0))]
        m = rgb.shape[0]
        n: IntArray = numpy.repeat(
            numpy.arange(m, dtype=numpy.int64)[:, None], k, axis=1
        )
        idx = (n, r0[sl], r1[sl], c0[sl], c1[sl])
        area = (r1[sl] - r0[sl]) * (c1[sl] - c0[sl])
        mean_lum = _box_sums(planes[0], *idx) / area
        mean_sq = _box_sums(planes[1], *idx) / area
        out[sl, :, 0] = mean_lum
        out[sl, :, 1] = numpy.sqrt(numpy.maximum(mean_sq - mean_lum**2, 0.0))
        out[sl, :, 2] = _box_sums(planes[2], *idx) / area
        shift = numpy.zeros_like(mean_lum)
        for channel, plane in enumerate(planes[3:]):
            whole = rgb[..., channel].mean(axis=(1, 2))[:, None]
            shift += (_box_sums(plane, *idx) / area - whole) ** 2
        out[sl, :, 3] = numpy.sqrt(shift)
    return out


def _standardized(a: FloatArray) -> FloatArray:
    std = a.std(axis=(0, 1), keepdims=True)
    return (a - a.mean(axis=(0, 1), keepdims=True)) / numpy.where(std > 0, std, 1.0)


@dataclasses.dataclass(frozen=True)
class DirectionResult:
    """Results of one checkpoint.

    Attributes:
        ridge_penalty: Selected ridge penalty of `direction_fit`.
        ridge_cv_r2: Held-out within `R^2` of the ridge prediction itself.
        r2: Per model, `[in-sample within R^2, low, high, held-out R^2]`.
        increments: Per comparison, `[held-out gain, in-sample gain, low, high]`.
        resid_auc: Per model, within-image AUC (point, low, high) of the radius left
            over after the model for the probe being correct on the view.
    """

    ridge_penalty: float
    ridge_cv_r2: float
    r2: dict[str, list[float]]
    increments: dict[str, list[float]]
    resid_auc: dict[str, tuple[float, float, float]]


MODELS = {
    "params": "P",
    "direction": "D",
    "params+direction": "PD",
    "params+content": "PC",
    "params+direction+content": "PDC",
}
# (name, with the block, without it)
COMPARISONS = (
    ("direction | params", "PD", "P"),
    ("content | params", "PC", "P"),
    ("content | params+direction", "PDC", "PD"),
    ("direction | params+content", "PDC", "PC"),
)


def analyze_direction(
    y: FloatArray,
    design: aug_regression.Design,
    z: FloatArray,
    content: FloatArray,
    correct: BoolArray,
    num_bootstrap: int,
    num_folds: int,
    rng: numpy.random.Generator,
) -> DirectionResult:
    """Fits the nested models on `y` `(N, K)` and scores their residuals."""
    n = y.shape[0]
    folds = aug_regression.image_folds(n, num_folds, rng)
    yd = aug_regression.demean(y)
    fit_d, penalty, ridge_r2 = direction_fit(z, yd, folds, num_folds)
    p = design.x.shape[-1]
    extra = numpy.concatenate(
        [
            _standardized(angle_to_others(z)[..., None]),
            _standardized(fit_d[..., None]),
            _standardized(content),
        ],
        axis=-1,
    )
    xd = aug_regression.demean(numpy.concatenate([design.x, extra], axis=-1))
    blocks = {
        "P": list(range(p)),
        "D": [p, p + 1],
        "C": list(range(p + 2, p + 2 + len(CONTENT_COLUMNS))),
    }

    def columns(code: str) -> list[int]:
        return [c for ch in code for c in blocks[ch]]

    stats = aug_regression.within_stats(xd, yd)
    weights = aug_regression.bootstrap_weights(n, num_bootstrap, rng)
    ones = numpy.ones(n)
    point = {
        code: aug_regression.fit(stats, ones, columns(code))[1]
        for code in MODELS.values()
    }
    draws = {
        code: numpy.asarray(
            [aug_regression.fit(stats, w, columns(code))[1] for w in weights]
        )
        for code in MODELS.values()
    }
    cv = {
        code: aug_regression.cv_r2(stats, columns(code), folds, num_folds)
        for code in MODELS.values()
    }
    r2 = {
        name: [point[code], *aug_regression.percentile_interval(draws[code]), cv[code]]
        for name, code in MODELS.items()
    }
    increments = {
        name: [
            cv[big] - cv[small],
            point[big] - point[small],
            *aug_regression.percentile_interval(draws[big] - draws[small]),
        ]
        for name, big, small in COMPARISONS
    }

    resid_auc: dict[str, tuple[float, float, float]] = {}
    residuals = {"radius (no model)": yd}
    for name, code in MODELS.items():
        pred = aug_regression.cross_fit(xd, yd, folds, num_folds, columns(code))
        residuals[name] = yd - pred
    for name, resid in residuals.items():
        wins, pairs = aug_regression.within_auc_counts(resid, correct)
        auc = [w @ wins / (w @ pairs) for w in weights]
        resid_auc[name] = (
            float(wins.sum() / pairs.sum()),
            *aug_regression.percentile_interval(numpy.asarray(auc)),
        )
    return DirectionResult(penalty, ridge_r2, r2, increments, resid_auc)
