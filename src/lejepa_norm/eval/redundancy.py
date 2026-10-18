"""Is the image-level radius redundant with the image-level direction?

Regresses the mean log-radius `rho_bar` on the normalized mean direction `u_hat`
with cross-validated kernel ridge regression and reports the held-out `R^2`.
Baselines: linear ridge on `u_hat`, and the class one-hot only (how much of the
radius is just class).

Design choices:
    * Kernel: `k(u, v) = exp((u . v - 1) / tau)`. On unit vectors this is both
      the exponential dot-product kernel and the RBF kernel
      `exp(-||u - v||^2 / (2 tau))`, so the two options in the design coincide.
    * An unpenalized intercept is fitted by centering the kernel in feature
      space (equivalently, ridge with intercept for the linear kernel).
    * Nested CV: the outer loop is K-fold; within each outer training fold,
      bandwidth and penalty are chosen by exact leave-one-out CV, which is free
      once the training kernel is eigendecomposed (one eigendecomposition per
      bandwidth per outer fold). `R^2` is pooled over the out-of-fold
      predictions: `1 - SS_res / SS_tot`.
"""

import dataclasses
import typing

import numpy
import numpy.typing

FloatArray = numpy.typing.NDArray[numpy.float64]
Kernel = typing.Literal["exp_dot", "linear"]

DEFAULT_TAU_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_ALPHAS: tuple[float, ...] = tuple(float(a) for a in numpy.logspace(-3, 2, 11))


@dataclasses.dataclass(frozen=True)
class RegressionResult:
    """Out-of-fold predictions of a cross-validated regression.

    Attributes:
        r2: Pooled held-out `R^2`.
        fold_r2: Held-out `R^2` of each outer fold.
        predictions: Out-of-fold predictions, `(N,)`.
        residuals: `y - predictions`, `(N,)`.
        params: Selected hyperparameters per outer fold (empty for baselines
            without hyperparameters).
    """

    r2: float
    fold_r2: tuple[float, ...]
    predictions: FloatArray
    residuals: FloatArray
    params: tuple[dict[str, float], ...]


def _r2(y: FloatArray, pred: FloatArray) -> float:
    ss_tot = float(numpy.square(y - y.mean()).sum())
    ss_res = float(numpy.square(y - pred).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _gram(a: FloatArray, b: FloatArray, kernel: Kernel, tau: float) -> FloatArray:
    dots = a @ b.T
    return dots if kernel == "linear" else numpy.exp((dots - 1.0) / tau)


def _folds(
    n: int, num_folds: int, rng: numpy.random.Generator
) -> list[numpy.typing.NDArray[numpy.int64]]:
    return list(numpy.array_split(rng.permutation(n), num_folds))


def _median_cosine_distance(u: FloatArray, rng: numpy.random.Generator) -> float:
    idx = rng.permutation(u.shape[0])[:1000]
    dist = 1.0 - u[idx] @ u[idx].T
    off_diag = dist[numpy.triu_indices(idx.size, k=1)]
    med = float(numpy.median(off_diag)) if off_diag.size else 1.0
    return med if med > 1e-8 else 1.0


class _CenteredKernelRidge:
    """Kernel ridge with intercept and LOO selection of the penalty."""

    def __init__(self, gram: FloatArray, y: FloatArray) -> None:
        self.col_mean = gram.mean(axis=0)
        self.all_mean = float(self.col_mean.mean())
        centered = gram - self.col_mean[None, :] - self.col_mean[:, None]
        centered += self.all_mean
        eigvals, self.eigvecs = numpy.linalg.eigh(centered)
        self.eigvals = numpy.clip(eigvals, 0.0, None)
        self.y_mean = float(y.mean())
        self.qty = self.eigvecs.T @ (y - self.y_mean)
        self.yc = y - self.y_mean

    def loo_mse(self, alpha: float) -> float:
        # Residuals and 1 - H_ii are formed from alpha / (lambda + alpha) directly
        # (not as 1 - shrinkage) to avoid cancellation when alpha << lambda.
        # The intercept adds 1/n to H_ii; the constant eigenvector (lambda = 0)
        # contributes exactly 1/n to the sum below, so it is subtracted.
        keep = alpha / (self.eigvals + alpha)
        resid = self.eigvecs @ (keep * self.qty)
        one_minus_h = numpy.square(self.eigvecs) @ keep - 1.0 / self.yc.size
        loo = resid / numpy.maximum(one_minus_h, 1e-12)
        return float(numpy.square(loo).mean())

    def predict(self, test_gram: FloatArray, alpha: float) -> FloatArray:
        coef = self.eigvecs @ (self.qty / (self.eigvals + alpha))
        centered = (
            test_gram
            - test_gram.mean(axis=1, keepdims=True)
            - self.col_mean[None, :]
            + self.all_mean
        )
        return self.y_mean + centered @ coef


def kernel_ridge_cv(
    u: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    kernel: Kernel = "exp_dot",
    num_folds: int = 5,
    rng: numpy.random.Generator | None = None,
    tau_multipliers: tuple[float, ...] = DEFAULT_TAU_MULTIPLIERS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> RegressionResult:
    """Nested-CV kernel (or linear) ridge regression of `y` on `u`.

    Args:
        u: `(N, d)` inputs, typically unit directions `u_hat`.
        y: `(N,)` targets, typically `rho_bar`.
        kernel: `"exp_dot"` (RBF on the sphere) or `"linear"` (linear ridge).
        num_folds: Outer folds.
        rng: Random generator for the fold split.
        tau_multipliers: Bandwidth grid, as multiples of the median pairwise
            cosine distance `1 - u . v` in the outer training fold.
        alphas: Ridge penalty grid.

    Returns:
        Out-of-fold predictions, residuals and pooled `R^2`.
    """
    rng = rng if rng is not None else numpy.random.default_rng(0)
    x = numpy.asarray(u, dtype=numpy.float64)
    t = numpy.asarray(y, dtype=numpy.float64).ravel()
    pred = numpy.empty_like(t)
    fold_r2: list[float] = []
    params: list[dict[str, float]] = []
    for test_idx in _folds(t.size, num_folds, rng):
        train = numpy.ones(t.size, dtype=bool)
        train[test_idx] = False
        x_tr, t_tr = x[train], t[train]
        taus = (
            (1.0,)
            if kernel == "linear"
            else tuple(m * _median_cosine_distance(x_tr, rng) for m in tau_multipliers)
        )
        best: tuple[float, float, float, _CenteredKernelRidge] | None = None
        for tau in taus:
            model = _CenteredKernelRidge(_gram(x_tr, x_tr, kernel, tau), t_tr)
            for alpha in alphas:
                mse = model.loo_mse(alpha)
                if best is None or mse < best[0]:
                    best = (mse, tau, alpha, model)
        assert best is not None
        _, tau, alpha, model = best
        pred[test_idx] = model.predict(_gram(x[test_idx], x_tr, kernel, tau), alpha)
        fold_r2.append(_r2(t[test_idx], pred[test_idx]))
        params.append(
            {"alpha": alpha} if kernel == "linear" else {"tau": tau, "alpha": alpha}
        )
    return RegressionResult(
        r2=_r2(t, pred),
        fold_r2=tuple(fold_r2),
        predictions=pred,
        residuals=t - pred,
        params=tuple(params),
    )


def group_mean_cv(
    groups: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    num_folds: int = 5,
    rng: numpy.random.Generator | None = None,
) -> RegressionResult:
    """Cross-validated regression of `y` on a one-hot categorical (group means).

    Args:
        groups: `(N,)` categorical predictor (e.g. class label).
        y: `(N,)` targets.
        num_folds: Outer folds.
        rng: Random generator for the fold split.

    Returns:
        Out-of-fold predictions (training-fold group means, falling back to the
        training-fold mean for unseen groups) and pooled `R^2`.
    """
    rng = rng if rng is not None else numpy.random.default_rng(0)
    g = numpy.asarray(groups).ravel()
    t = numpy.asarray(y, dtype=numpy.float64).ravel()
    _, codes = numpy.unique(g, return_inverse=True)
    num_groups = int(codes.max()) + 1 if codes.size else 0
    pred = numpy.empty_like(t)
    fold_r2: list[float] = []
    for test_idx in _folds(t.size, num_folds, rng):
        train = numpy.ones(t.size, dtype=bool)
        train[test_idx] = False
        sums = numpy.bincount(codes[train], weights=t[train], minlength=num_groups)
        counts = numpy.bincount(codes[train], minlength=num_groups)
        means = numpy.where(
            counts > 0, sums / numpy.maximum(counts, 1), t[train].mean()
        )
        pred[test_idx] = means[codes[test_idx]]
        fold_r2.append(_r2(t[test_idx], pred[test_idx]))
    return RegressionResult(
        r2=_r2(t, pred),
        fold_r2=tuple(fold_r2),
        predictions=pred,
        residuals=t - pred,
        params=(),
    )
