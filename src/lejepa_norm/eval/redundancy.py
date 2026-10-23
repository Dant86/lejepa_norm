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
import scipy.stats  # pyright: ignore[reportMissingTypeStubs]

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


class _Spectrum:
    """Eigendecomposition of a centered training kernel (independent of the target)."""

    def __init__(self, gram: FloatArray) -> None:
        self.col_mean = gram.mean(axis=0)
        self.all_mean = float(self.col_mean.mean())
        centered = gram - self.col_mean[None, :] - self.col_mean[:, None]
        centered += self.all_mean
        eigvals, self.eigvecs = numpy.linalg.eigh(centered)
        self.eigvals = numpy.clip(eigvals, 0.0, None)
        self._leverage: tuple[bytes, FloatArray] | None = None

    def keep(self, alphas: FloatArray) -> FloatArray:
        """`alpha / (lambda + alpha)`, shape `(n, len(alphas))`.

        Formed directly (not as 1 - shrinkage) to avoid cancellation when
        alpha << lambda.
        """
        return alphas[None, :] / (self.eigvals[:, None] + alphas[None, :])

    def one_minus_h(self, alphas: FloatArray) -> FloatArray:
        """`1 - H_ii` of the leave-one-out formula, `(n, len(alphas))`.

        It does not depend on the target, so it is computed once and cached (the
        permutation null re-fits the same spectrum hundreds of times). The
        intercept adds 1/n to H_ii; the constant eigenvector (lambda = 0)
        contributes exactly 1/n to the sum, so it is subtracted.
        """
        key = alphas.tobytes()
        if self._leverage is None or self._leverage[0] != key:
            value = numpy.square(self.eigvecs) @ self.keep(alphas)
            self._leverage = (key, value - 1.0 / self.eigvals.size)
        return self._leverage[1]


class _CenteredKernelRidge:
    """Kernel ridge with intercept and LOO selection of the penalty."""

    def __init__(self, spectrum: _Spectrum, y: FloatArray) -> None:
        self.spectrum = spectrum
        self.y_mean = float(y.mean())
        self.yc = y - self.y_mean
        self.qty = spectrum.eigvecs.T @ self.yc

    def loo_mse(self, alphas: FloatArray) -> FloatArray:
        """Exact leave-one-out MSE for each penalty in `alphas`."""
        eigvecs = self.spectrum.eigvecs
        keep = self.spectrum.keep(alphas)
        resid = eigvecs @ (keep * self.qty[:, None])
        loo = resid / numpy.maximum(self.spectrum.one_minus_h(alphas), 1e-12)
        return numpy.square(loo).mean(axis=0)

    def predict(self, test_gram: FloatArray, alpha: float) -> FloatArray:
        spec = self.spectrum
        coef = spec.eigvecs @ (self.qty / (spec.eigvals + alpha))
        centered = (
            test_gram
            - test_gram.mean(axis=1, keepdims=True)
            - spec.col_mean[None, :]
            + spec.all_mean
        )
        return self.y_mean + centered @ coef


@dataclasses.dataclass(frozen=True)
class _FoldCache:
    """Target-independent state of one outer fold."""

    test_idx: numpy.typing.NDArray[numpy.int64]
    train_idx: numpy.typing.NDArray[numpy.int64]
    taus: tuple[float, ...]
    spectra: tuple[_Spectrum, ...]
    test_grams: tuple[FloatArray, ...]


class NestedKernelRidge:
    """Nested-CV kernel (or linear) ridge regression on fixed inputs.

    Everything that does not depend on the target (outer folds, bandwidths,
    training-kernel eigendecompositions, test kernels) is computed once in the
    constructor, so `fit` can be called for many targets (for example permuted
    ones) at a cost of a few matrix-vector products per fold and bandwidth.
    Memory: one `(n_train, n_train)` eigenvector matrix per fold and bandwidth.

    The random generator is consumed exactly as `kernel_ridge_cv` always has:
    first the fold split, then one bandwidth estimate per fold.
    """

    def __init__(
        self,
        u: numpy.typing.ArrayLike,
        kernel: Kernel = "exp_dot",
        num_folds: int = 5,
        rng: numpy.random.Generator | None = None,
        tau_multipliers: tuple[float, ...] = DEFAULT_TAU_MULTIPLIERS,
        alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    ) -> None:
        """Prepares the folds and kernels.

        Args:
            u: Inputs, `(N, d)`; unit vectors for the `exp_dot` kernel.
            kernel: `"exp_dot"` (kernel ridge) or `"linear"` (ridge).
            num_folds: Outer folds.
            rng: Random generator for the fold split; seeded with 0 if None.
            tau_multipliers: Bandwidth grid, as multiples of the median pairwise
                cosine distance `1 - u . v` in the outer training fold.
            alphas: Ridge penalty grid.
        """
        rng = rng if rng is not None else numpy.random.default_rng(0)
        x = numpy.asarray(u, dtype=numpy.float64)
        self.num_samples = x.shape[0]
        self.alphas = numpy.asarray(alphas, dtype=numpy.float64)
        self._folds: list[_FoldCache] = []
        for test_idx in _folds(self.num_samples, num_folds, rng):
            train = numpy.ones(self.num_samples, dtype=bool)
            train[test_idx] = False
            x_tr = x[train]
            taus = (
                (1.0,)
                if kernel == "linear"
                else tuple(
                    m * _median_cosine_distance(x_tr, rng) for m in tau_multipliers
                )
            )
            self._folds.append(
                _FoldCache(
                    test_idx=test_idx,
                    train_idx=numpy.flatnonzero(train),
                    taus=taus,
                    spectra=tuple(
                        _Spectrum(_gram(x_tr, x_tr, kernel, tau)) for tau in taus
                    ),
                    test_grams=tuple(
                        _gram(x[test_idx], x_tr, kernel, tau) for tau in taus
                    ),
                )
            )
        self._kernel = kernel

    def r2_many(
        self, ys: FloatArray, chunk: int = 64
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Pooled held-out `R^2` of several targets at once, `(P,)`.

        Equivalent to `[self.fit(y).r2 for y in ys]` (hyperparameters are
        re-selected for every target) but the eigenvector products are batched
        over targets, which turns matrix-vector into matrix-matrix products:
        about an order of magnitude faster for permutation nulls.

        Args:
            ys: Targets, `(P, N)`.
            chunk: Targets processed per batch (bounds memory).
        """
        targets = numpy.asarray(ys, dtype=numpy.float64)
        if targets.ndim != 2 or targets.shape[1] != self.num_samples:
            raise ValueError("ys must have shape (P, N)")
        out = [
            self._r2_batch(targets[i : i + chunk])
            for i in range(0, targets.shape[0], chunk)
        ]
        return numpy.concatenate(out) if out else numpy.empty(0)

    def _r2_batch(self, targets: FloatArray) -> FloatArray:
        num = targets.shape[0]
        pred = numpy.empty_like(targets)
        for fold in self._folds:
            y_tr = targets[:, fold.train_idx]
            y_mean = y_tr.mean(axis=1)
            yc = (y_tr - y_mean[:, None]).T  # (n_train, P)
            best_mse = numpy.full(num, numpy.inf)
            best_tau = numpy.zeros(num, dtype=numpy.int64)
            best_alpha = numpy.zeros(num, dtype=numpy.int64)
            qtys: list[FloatArray] = []
            for k, spec in enumerate(fold.spectra):
                qty = spec.eigvecs.T @ yc  # (n_train, P)
                qtys.append(qty)
                keep = spec.keep(self.alphas)  # (n_train, A)
                omh = numpy.maximum(spec.one_minus_h(self.alphas), 1e-12)
                mse = numpy.empty((self.alphas.size, num))
                for a in range(self.alphas.size):
                    resid = spec.eigvecs @ (keep[:, a : a + 1] * qty)
                    mse[a] = numpy.square(resid / omh[:, a : a + 1]).mean(axis=0)
                arg = mse.argmin(axis=0)
                val = mse[arg, numpy.arange(num)]
                better = val < best_mse
                best_mse = numpy.where(better, val, best_mse)
                best_tau = numpy.where(better, k, best_tau)
                best_alpha = numpy.where(better, arg, best_alpha)
            for k, spec in enumerate(fold.spectra):
                sel = numpy.flatnonzero(best_tau == k)
                if sel.size == 0:
                    continue
                alpha = self.alphas[best_alpha[sel]]
                coef = spec.eigvecs @ (
                    qtys[k][:, sel] / (spec.eigvals[:, None] + alpha[None, :])
                )
                tg = fold.test_grams[k]
                centered = (
                    tg
                    - tg.mean(axis=1, keepdims=True)
                    - spec.col_mean[None, :]
                    + spec.all_mean
                )
                pred[numpy.ix_(sel, fold.test_idx)] = (
                    y_mean[sel, None] + (centered @ coef).T
                )
        ss_res = numpy.square(targets - pred).sum(axis=1)
        ss_tot = numpy.square(targets - targets.mean(axis=1, keepdims=True)).sum(axis=1)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            return numpy.where(ss_tot > 0, 1.0 - ss_res / ss_tot, numpy.nan)

    def fit(self, y: numpy.typing.ArrayLike) -> RegressionResult:
        """Held-out predictions and `R^2` for the target `y`, `(N,)`."""
        t = numpy.asarray(y, dtype=numpy.float64).ravel()
        if t.size != self.num_samples:
            raise ValueError("y must have one entry per row of u")
        pred = numpy.empty_like(t)
        fold_r2: list[float] = []
        params: list[dict[str, float]] = []
        for fold in self._folds:
            best: tuple[float, int, float, _CenteredKernelRidge] | None = None
            for k, spectrum in enumerate(fold.spectra):
                model = _CenteredKernelRidge(spectrum, t[fold.train_idx])
                mse = model.loo_mse(self.alphas)
                a = int(mse.argmin())
                if best is None or mse[a] < best[0]:
                    best = (float(mse[a]), k, float(self.alphas[a]), model)
            assert best is not None
            _, k, alpha, model = best
            pred[fold.test_idx] = model.predict(fold.test_grams[k], alpha)
            fold_r2.append(_r2(t[fold.test_idx], pred[fold.test_idx]))
            params.append(
                {"alpha": alpha}
                if self._kernel == "linear"
                else {"tau": fold.taus[k], "alpha": alpha}
            )
        return RegressionResult(
            r2=_r2(t, pred),
            fold_r2=tuple(fold_r2),
            predictions=pred,
            residuals=t - pred,
            params=tuple(params),
        )


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
        u: Inputs, `(N, d)`; unit vectors for the `exp_dot` kernel.
        y: Targets, `(N,)`.
        kernel: `"exp_dot"` (kernel ridge) or `"linear"` (ridge).
        num_folds: Outer folds.
        rng: Random generator for the fold split; seeded with 0 if None.
        tau_multipliers: Bandwidth grid, as multiples of the median pairwise
            cosine distance `1 - u . v` in the outer training fold.
        alphas: Ridge penalty grid.

    Returns:
        Out-of-fold predictions, residuals and pooled `R^2`.
    """
    model = NestedKernelRidge(u, kernel, num_folds, rng, tau_multipliers, alphas)
    return model.fit(y)


def fit_predict(
    u_train: numpy.typing.ArrayLike,
    y_train: numpy.typing.ArrayLike,
    u_test: numpy.typing.ArrayLike,
    kernel: Kernel = "exp_dot",
    rng: numpy.random.Generator | None = None,
    tau_multipliers: tuple[float, ...] = DEFAULT_TAU_MULTIPLIERS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> FloatArray:
    """Kernel (or linear) ridge fitted on all of `u_train`, applied to `u_test`.

    Bandwidth and penalty are chosen by exact leave-one-out CV on the training
    rows, exactly as inside one outer fold of `NestedKernelRidge`; the test rows
    take no part in the fit, so the predictions are held out.

    Args:
        u_train: Training inputs, `(M, d)`; unit vectors for `exp_dot`.
        y_train: Training targets, `(M,)`.
        u_test: Inputs to predict at, `(N, d)`.
        kernel: `"exp_dot"` or `"linear"`.
        rng: Random generator for the bandwidth estimate; seeded with 0 if None.
        tau_multipliers: Bandwidth grid (multiples of the median cosine distance).
        alphas: Ridge penalty grid.

    Returns:
        Predictions at `u_test`, `(N,)`.
    """
    rng = rng if rng is not None else numpy.random.default_rng(0)
    x = numpy.asarray(u_train, dtype=numpy.float64)
    t = numpy.asarray(y_train, dtype=numpy.float64).ravel()
    xt = numpy.asarray(u_test, dtype=numpy.float64)
    if t.size != x.shape[0]:
        raise ValueError("y_train must have one entry per row of u_train")
    grid = numpy.asarray(alphas, dtype=numpy.float64)
    median = _median_cosine_distance(x, rng) if kernel == "exp_dot" else 1.0
    taus = (1.0,) if kernel == "linear" else tuple(m * median for m in tau_multipliers)
    best: tuple[float, float, float, _CenteredKernelRidge] | None = None
    for tau in taus:
        model = _CenteredKernelRidge(_Spectrum(_gram(x, x, kernel, tau)), t)
        mse = model.loo_mse(grid)
        a = int(mse.argmin())
        if best is None or mse[a] < best[0]:
            best = (float(mse[a]), tau, float(grid[a]), model)
    assert best is not None
    _, tau, alpha, model = best
    return model.predict(_gram(xt, x, kernel, tau), alpha)


@dataclasses.dataclass(frozen=True)
class PermutationTest:
    """Permutation test of "the target is independent of the inputs".

    Attributes:
        observed: Held-out `R^2` with the true targets.
        null: Held-out `R^2` for each permutation of the targets, run through the
            identical nested-CV pipeline (hyperparameters re-selected each time).
        p_value: `(1 + #{null >= observed}) / (1 + len(null))`; never below
            `1 / (1 + len(null))`.
    """

    observed: float
    null: FloatArray
    p_value: float


def permutation_test(
    model: NestedKernelRidge,
    y: numpy.typing.ArrayLike,
    num_permutations: int,
    rng: numpy.random.Generator,
) -> PermutationTest:
    """Permutation null of the held-out `R^2` of `model` for the target `y`."""
    t = numpy.asarray(y, dtype=numpy.float64).ravel()
    observed = model.fit(t).r2
    shuffled = numpy.empty((num_permutations, t.size))
    for row in shuffled:
        row[:] = rng.permutation(t)
    null = model.r2_many(shuffled)
    p_value = (1 + int((null >= observed).sum())) / (1 + num_permutations)
    return PermutationTest(observed=observed, null=null, p_value=p_value)


@dataclasses.dataclass(frozen=True)
class FTest:
    """Classical F-test of the linear regression of `y` on `x` (with intercept).

    Valid under independent Gaussian errors; it only detects linear dependence
    and uses the in-sample fit, so it complements, not replaces, the permutation
    test of the held-out `R^2`.

    Attributes:
        r2: In-sample `R^2`.
        f_statistic: `(explained / df1) / (residual / df2)`.
        df1: Rank of the centered design (numerator degrees of freedom).
        df2: `N - df1 - 1`.
        p_value: Upper tail of the `F(df1, df2)` distribution.
    """

    r2: float
    f_statistic: float
    df1: int
    df2: int
    p_value: float


def f_test(x: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike) -> FTest:
    """Overall F-test of the linear regression of `y` on the columns of `x`."""
    design = numpy.asarray(x, dtype=numpy.float64)
    t = numpy.asarray(y, dtype=numpy.float64).ravel()
    full = numpy.column_stack([numpy.ones(t.size), design])
    beta, _, rank, _ = numpy.linalg.lstsq(full, t, rcond=None)
    ss_res = float(numpy.square(t - full @ beta).sum())
    ss_tot = float(numpy.square(t - t.mean()).sum())
    df1, df2 = int(rank) - 1, t.size - int(rank)
    if df1 < 1 or df2 < 1 or ss_res <= 0 or ss_tot <= 0:
        return FTest(float("nan"), float("nan"), df1, df2, float("nan"))
    f_statistic = ((ss_tot - ss_res) / df1) / (ss_res / df2)
    return FTest(
        r2=1.0 - ss_res / ss_tot,
        f_statistic=f_statistic,
        df1=df1,
        df2=df2,
        p_value=float(scipy.stats.f.sf(f_statistic, df1, df2)),  # pyright: ignore[reportUnknownMemberType]
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
