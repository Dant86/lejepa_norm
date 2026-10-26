"""E2 controls of what a high redundancy R^2 of the radius means (z only).

The redundancy result is the held-out `R^2` of the image-level mean log-radius
`rho_bar` from the mean direction `u_hat` (nested-CV kernel ridge,
`redundancy.NestedKernelRidge`). Three controls of its meaning, computed per run
from its `stl10_train` and `stl10_test` embedding dumps (K = 8 augmented views):

1. **Redundancy controls** (`controls`): the existing same-view, disjoint-halves,
   single-view, permutation-null and F-test rows (`eval.controls`), with the
   held-out predictions kept so that `R^2` gets a bootstrap interval over images.
   Applied to trained LeJEPA runs and to random-init networks.
2. **Random smooth functions of the direction** (`smooth`): targets drawn from a
   Gaussian-process prior on the sphere with the *same kernel family* as the fit,
   `k(u, v) = exp((u . v - 1) / tau)`, on the same `u_hat` and predicted by the
   identical nested-CV pipeline (`NestedKernelRidge.r2_many`: bandwidth and penalty
   re-selected per target). Families, with `tau_hat` the median bandwidth selected
   by the real fit and `eps` the iid noise fraction `1 - reliability`, where the
   reliability of `rho_bar` is the Spearman-Brown step-up of the split-half
   correlation:

   * `GP tau = m tau_hat` for `m` in `SMOOTH_MULTIPLIERS`, noiseless: how much of a
     function of that smoothness the fit can recover at all (the ceiling) and how
     the number falls for rougher functions (`m < 1`);
   * the same with iid noise of fraction `eps` added: functions *matched* to the
     learned map in both smoothness (`m = 1`) and noise level;
   * random linear functions `u . w` (noiseless / with noise).

   Draws are standardized to the standard deviation of `rho_bar` (R^2 is scale
   free; this only fixes the scale of the noise).
3. **Learning curve** (`curve`): held-out `R^2` against the number of training
   images N, on the pooled `stl10_train` + `stl10_test` images. One fixed random
   held-out set (the same images for every run) and nested random training subsets
   of the remaining images (several draws per N below the largest), `u -> rho_bar`
   fitted by `redundancy.FittedKernelRidge` (bandwidth and penalty by exact LOO on
   the training rows, as inside one outer fold; for N above 4000 the bandwidth is
   the one selected at N = 4000 on the same training draw, which saves four of
   five eigendecompositions of an N x N matrix, and only the penalty is
   re-selected by exact LOO). Two setups: same views, and disjoint
   halves (`u_hat` of views 0-3 to `rho_bar` of views 4-7). Per-image squared
   errors are stored, so the aggregation bootstraps held-out images paired across
   N. Saturation of N = 4000 is judged by `R^2(largest N) - R^2(4000)` against
   the margin `eval.curve_margin` (fixed in advance, in R^2 units) with the
   equivalence verdict shared with E1 (`randomization.verdict`).

Per-run results are written to one `.npz` shard (`save_run`); `*_rows` aggregate
shards per group (a group is a run name without its `_s<seed>` suffix) with the E1
bootstrap helpers (`randomization.bootstrap_draws`, `bootstrap_seed_mean`,
`intervals`, `verdict`): the images are resampled within each seed and the mean
over seeds is taken.
"""

import collections.abc
import dataclasses
import json
import os
import pathlib
import re
import typing

import numpy
import numpy.typing

from lejepa_norm import config
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import analysis
from lejepa_norm.eval import controls
from lejepa_norm.eval import features
from lejepa_norm.eval import randomization
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
Row = analysis.Row

SMOOTH_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
CURVE_SIZES: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 8000)
# Training subsets drawn per N (the largest N is the whole training pool).
CURVE_REPEATS = {250: 5, 500: 5, 1000: 5, 2000: 3, 4000: 3, 8000: 2}
CURVE_SETUPS = ("same views", "disjoint halves")
# The held-out images of the curve are the same for every run.
HOLDOUT_SEED = 0
REFERENCE_N = 4000
PARTS = ("controls", "smooth", "curve")

_RUN_NAME = re.compile(r"^(.*)_s(\d+)$")


def split_run_name(name: str) -> tuple[str, int]:
    """`radius_marginal_lam0.05_s2` -> (`radius_marginal_lam0.05`, 2)."""
    match = _RUN_NAME.match(name)
    if match is None:
        raise ValueError(f"run name {name!r} does not end in _s<seed>")
    return match.group(1), int(match.group(2))


@dataclasses.dataclass
class RunResult:
    """Everything one run contributes (see the module docstring).

    Attributes:
        name: Run name; the group is `split_run_name(name)[0]`.
        info: JSON-able scalars and tables (control rows, smooth parameters,
            curve sizes).
        arrays: Float arrays (held-out targets, predictions, per-draw `R^2`,
            per-image squared errors) under `"part|..."` keys.
    """

    name: str
    info: dict[str, typing.Any]
    arrays: dict[str, FloatArray]

    @property
    def group(self) -> str:
        """Run name without the seed."""
        return split_run_name(self.name)[0]

    @property
    def seed(self) -> int:
        """Training (or initialization) seed from the run name."""
        return split_run_name(self.name)[1]


def save_run(path: str | os.PathLike[str], result: RunResult) -> None:
    """Writes a run result atomically as an `.npz` file."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    payload: dict[str, typing.Any] = dict(result.arrays)
    payload["meta|name"] = numpy.asarray(result.name)
    payload["meta|info"] = numpy.asarray(json.dumps(result.info))
    with partial.open("wb") as handle:
        numpy.savez_compressed(handle, **payload)
    partial.replace(target)


def load_run(path: str | os.PathLike[str]) -> RunResult:
    """Reads a file written by `save_run`."""
    with numpy.load(path, allow_pickle=False) as data:
        arrays = {
            k: numpy.asarray(data[k], dtype=numpy.float64)
            for k in data.files
            if not k.startswith("meta|")
        }
        return RunResult(
            str(data["meta|name"]), json.loads(str(data["meta|info"])), arrays
        )


# ---------------------------------------------------------------------------
# Part 1: redundancy controls with held-out predictions.


def controls_part(
    test: shards.EmbeddingShard, cfg: config.EvalConfig, seed: int
) -> tuple[list[Row], dict[str, FloatArray]]:
    """The z-space redundancy controls of `eval.controls`, keeping predictions.

    The image subset, setups and permutation nulls are those of
    `controls.run_redundancy_controls` (z only).

    Returns:
        The table rows (one per setup, plus the reliability row) and, per setup
        index `i`, the arrays `ctl|i|y` and `ctl|i|pred` of the held-out fit.
    """
    rng = numpy.random.default_rng(seed)
    idx = analysis.subsample(test.label.size, cfg.max_kernel_images, rng)
    x = features.select_views(features.space_array(test, "z"), "augmented")[idx]
    if x.shape[1] < 2:
        raise ValueError("the controls need at least two augmented views")
    rows: list[Row] = []
    arrays: dict[str, FloatArray] = {}
    for i, setup in enumerate(controls.view_setups(x)):
        row, fit = controls.run_setup_detailed(setup, cfg, rng)
        rows.append({"setup": setup.name, **row, "setup_index": i})
        arrays[f"ctl|{i}|y"] = setup.rho
        arrays[f"ctl|{i}|pred"] = fit.predictions
    rows.append(
        {
            "setup": controls.RELIABILITY_SETUP,
            "r2 (nested CV, kernel)": controls.split_half_reliability(x),
        }
    )
    return rows, arrays


# ---------------------------------------------------------------------------
# Part 2: random smooth functions of the direction.


def spearman_brown_noise_fraction(x: FloatArray) -> float:
    """`1 - reliability` of `rho_bar` over all views (Spearman-Brown of split halves).

    Args:
        x: Augmented-view embeddings `(N, K, d)`.
    """
    rho = features.log_radius(x)
    half = x.shape[1] // 2
    r = float(
        numpy.corrcoef(rho[:, :half].mean(axis=1), rho[:, half:].mean(axis=1))[0, 1]
    )
    r = max(r, 0.0)
    return 1.0 - 2.0 * r / (1.0 + r)


def gp_draws(
    u: FloatArray, tau: float, count: int, rng: numpy.random.Generator
) -> FloatArray:
    """Draws from the GP prior with `k(u, v) = exp((u . v - 1) / tau)`, `(count, N)`.

    The draws are `V sqrt(lambda) xi` from the eigendecomposition of the kernel
    matrix, centered and scaled to unit sample standard deviation.
    """
    gram = numpy.exp((u @ u.T - 1.0) / tau)
    eigvals, eigvecs = numpy.linalg.eigh(gram)
    root = numpy.sqrt(numpy.clip(eigvals, 0.0, None))
    draws = (eigvecs @ (root[:, None] * rng.standard_normal((u.shape[0], count)))).T
    draws -= draws.mean(axis=1, keepdims=True)
    return draws / draws.std(axis=1, keepdims=True)


def _with_noise(
    signal: FloatArray, noise_fraction: float, scale: float, rng: numpy.random.Generator
) -> FloatArray:
    """`scale * (sqrt(1 - eps) signal + sqrt(eps) noise)` for unit-variance rows."""
    noise = rng.standard_normal(signal.shape)
    noise = (noise - noise.mean(axis=1, keepdims=True)) / noise.std(
        axis=1, keepdims=True
    )
    return scale * (
        numpy.sqrt(1.0 - noise_fraction) * signal + numpy.sqrt(noise_fraction) * noise
    )


def smooth_part(
    test: shards.EmbeddingShard, cfg: config.EvalConfig, seed: int
) -> tuple[dict[str, typing.Any], dict[str, FloatArray]]:
    """Held-out `R^2` of random smooth functions of `u_hat` (module docstring, 2).

    Returns:
        JSON-able parameters (`tau_hat`, noise fraction, observed `R^2`, family
        names) and the per-draw `R^2` arrays `smooth|<family>`.
    """
    rng = numpy.random.default_rng(seed)
    idx = analysis.subsample(test.label.size, cfg.max_kernel_images, rng)
    x = features.select_views(features.space_array(test, "z"), "augmented")[idx]
    rho = features.log_radius(x).mean(axis=1)
    u = features.direction(features.direction(x).mean(axis=1))
    model = redundancy.NestedKernelRidge(u, "exp_dot", cfg.cv_folds, rng)
    fit = model.fit(rho)
    taus = [p["tau"] for p in fit.params]
    tau_hat = float(numpy.median(taus))
    eps = spearman_brown_noise_fraction(x)
    scale = float(rho.std())
    arrays: dict[str, FloatArray] = {}
    families: list[str] = []

    def add(name: str, targets: FloatArray) -> None:
        arrays[f"smooth|{name}"] = model.r2_many(targets)
        families.append(name)

    for multiplier in SMOOTH_MULTIPLIERS:
        draws = gp_draws(u, multiplier * tau_hat, cfg.smooth_draws, rng)
        add(f"GP tau={multiplier:g}x, noiseless", scale * draws)
        add(
            f"GP tau={multiplier:g}x, matched noise",
            _with_noise(draws, eps, scale, rng),
        )
    w = rng.standard_normal((u.shape[1], cfg.smooth_draws))
    linear = u @ w
    linear = (linear - linear.mean(axis=0)) / linear.std(axis=0)
    add("random linear, noiseless", scale * linear.T)
    add("random linear, matched noise", _with_noise(linear.T, eps, scale, rng))
    info = {
        "tau_hat": tau_hat,
        "tau_per_fold": taus,
        "noise_fraction": eps,
        "observed_r2": fit.r2,
        "families": families,
        "draws": cfg.smooth_draws,
        "images": int(rho.size),
    }
    return info, arrays


MATCHED_FAMILY = "GP tau=1x, matched noise"


# ---------------------------------------------------------------------------
# Part 3: learning curve.


def curve_part(
    bank: shards.EmbeddingShard,
    test: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    seed: int,
) -> tuple[dict[str, typing.Any], dict[str, FloatArray]]:
    """Held-out `R^2` against the number of training images (module docstring, 3).

    Args:
        bank: `stl10_train` dump.
        test: `stl10_test` dump.
        cfg: Settings (`curve_holdout`, kernel-ridge grids are the defaults).
        seed: Seed of the bandwidth estimates (not of the splits, which are fixed).

    Returns:
        JSON-able info (sizes, repeats, holdout size) and the arrays
        `curve|<setup>|y` (held-out targets), `curve|<setup>|<N>|sq` (per-image
        squared error averaged over the training draws) and
        `curve|<setup>|<N>|r2` (the held-out `R^2` of each training draw).
    """
    x = numpy.concatenate(
        [
            features.select_views(features.space_array(s, "z"), "augmented")
            for s in (bank, test)
        ]
    )
    num_views = x.shape[1]
    half = num_views // 2
    unit = features.direction(x)
    rho = features.log_radius(x)
    inputs = {
        CURVE_SETUPS[0]: features.direction(unit.mean(axis=1)),
        CURVE_SETUPS[1]: features.direction(unit[:, :half].mean(axis=1)),
    }
    targets = {
        CURVE_SETUPS[0]: rho.mean(axis=1),
        CURVE_SETUPS[1]: rho[:, half:].mean(axis=1),
    }
    order = numpy.random.default_rng(HOLDOUT_SEED).permutation(x.shape[0])
    holdout, pool = order[: cfg.curve_holdout], order[cfg.curve_holdout :]
    sizes = [*(n for n in CURVE_SIZES if n < pool.size), int(pool.size)]
    rng = numpy.random.default_rng(seed)
    arrays: dict[str, FloatArray] = {}
    for setup in CURVE_SETUPS:
        u, y = inputs[setup], targets[setup]
        y_ho = y[holdout]
        arrays[f"curve|{setup}|y"] = y_ho
        ss_tot = float(numpy.square(y_ho - y_ho.mean()).sum())
        # Bandwidth chosen at N = REFERENCE_N, per training draw, reused above it.
        chosen: dict[int, float] = {}
        for n in sizes:
            repeats = CURVE_REPEATS.get(n, 1)
            sq = numpy.zeros(holdout.size)
            r2s: list[float] = []
            for r in range(repeats):
                train = pool[
                    numpy.random.default_rng(1000 + r).permutation(pool.size)[:n]
                ]
                fixed = (chosen[r],) if n > REFERENCE_N else None
                fitted = redundancy.FittedKernelRidge(
                    u[train], y[train], rng=rng, taus=fixed
                )
                if n == REFERENCE_N:
                    chosen[r] = fitted.tau
                err = numpy.square(y_ho - fitted.predict(u[holdout]))
                sq += err / repeats
                r2s.append(1.0 - float(err.sum()) / ss_tot)
            arrays[f"curve|{setup}|{n}|sq"] = sq
            arrays[f"curve|{setup}|{n}|r2"] = numpy.asarray(r2s)
    info = {
        "sizes": sizes,
        "repeats": {str(n): CURVE_REPEATS.get(n, 1) for n in sizes},
        "holdout": int(holdout.size),
        "pool": int(pool.size),
        "setups": list(CURVE_SETUPS),
    }
    return info, arrays


# ---------------------------------------------------------------------------
# Aggregation.


def run_unit(
    name: str,
    bank: shards.EmbeddingShard,
    test: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    seed: int,
    parts: collections.abc.Collection[str] = PARTS,
) -> RunResult:
    """Computes the requested `parts` (`controls`, `smooth`, `curve`) of one run."""
    info: dict[str, typing.Any] = {}
    arrays: dict[str, FloatArray] = {}
    if "controls" in parts:
        rows, arrays_c = controls_part(test, cfg, seed)
        info["controls"] = rows
        arrays.update(arrays_c)
    if "smooth" in parts:
        info["smooth"], arrays_s = smooth_part(test, cfg, seed)
        arrays.update(arrays_s)
    if "curve" in parts:
        info["curve"], arrays_v = curve_part(bank, test, cfg, seed)
        arrays.update(arrays_v)
    return RunResult(name, info, arrays)


def _r2_boot(
    y: FloatArray, sq_err: FloatArray, idx: numpy.typing.NDArray[numpy.int64]
) -> FloatArray:
    """`R^2` of every resample: `idx` is `(B, n)`, `sq_err` the per-image error."""
    return 1.0 - sq_err[idx].mean(axis=1) / y[idx].var(axis=1)


def _ci(boot: FloatArray) -> str:
    _, (lo, hi) = randomization.intervals(boot)
    return f"[{lo:.4f}, {hi:.4f}]"


def _by_group(results: list[RunResult]) -> dict[str, list[RunResult]]:
    groups: dict[str, list[RunResult]] = {}
    for result in sorted(results, key=lambda r: (r.group, r.seed)):
        groups.setdefault(result.group, []).append(result)
    return groups


def _control_cells(
    result: RunResult,
) -> dict[str, tuple[Row, FloatArray, FloatArray]]:
    """Setup name -> (row, held-out y, squared error) of one run."""
    cells: dict[str, tuple[Row, FloatArray, FloatArray]] = {}
    for row in result.info["controls"]:
        if "setup_index" not in row:
            continue
        i = row["setup_index"]
        y = result.arrays[f"ctl|{i}|y"]
        err = numpy.square(y - result.arrays[f"ctl|{i}|pred"])
        cells[row["setup"]] = (row, y, err)
    return cells


def controls_rows(
    results: list[RunResult], num_bootstrap: int, rng: numpy.random.Generator
) -> list[Row]:
    """Per group and setup: `R^2` (mean over seeds, bootstrap over images), nulls, F."""
    out: list[Row] = []
    for group, runs in _by_group(results).items():
        cells = [_control_cells(r) for r in runs]
        n = next(iter(cells[0].values()))[1].size
        draws = randomization.bootstrap_draws(n, len(runs), num_bootstrap, rng)
        for setup in cells[0]:
            parts = [c[setup] for c in cells]
            per_seed = [
                1.0 - float(p[2].sum()) / float(numpy.square(p[1] - p[1].mean()).sum())
                for p in parts
            ]
            boot = randomization.bootstrap_seed_mean(
                draws, lambda s, idx, p=parts: _r2_boot(p[s][1], p[s][2], idx)
            )
            rows = [p[0] for p in parts]
            out.append(
                {
                    "group": group,
                    "setup": setup,
                    "seeds": len(runs),
                    "r2 (mean over seeds)": float(numpy.mean(per_seed)),
                    "95% CI (images)": _ci(boot),
                    "seed range": f"[{min(per_seed):.4f}, {max(per_seed):.4f}]",
                    "null max": max(float(r["null max"]) for r in rows),
                    "worst permutation p": max(float(r["permutation p"]) for r in rows),
                    "min F": min(float(r["F"]) for r in rows),
                }
            )
        reliab = [
            float(row["r2 (nested CV, kernel)"])
            for r in runs
            for row in r.info["controls"]
            if row["setup"] == controls.RELIABILITY_SETUP
        ]
        out.append(
            {
                "group": group,
                "setup": controls.RELIABILITY_SETUP,
                "seeds": len(runs),
                "r2 (mean over seeds)": float(numpy.mean(reliab)),
                "seed range": f"[{min(reliab):.4f}, {max(reliab):.4f}]",
            }
        )
    return out


def _r2_point(y: FloatArray, sq_err: FloatArray) -> float:
    return 1.0 - float(sq_err.mean()) / float(y.var())


def _mean_boot_r2(
    cells: list[dict[str, tuple[Row, FloatArray, FloatArray]]],
    setup: str,
    idx: numpy.typing.NDArray[numpy.int64],
) -> FloatArray:
    """Bootstrap `R^2` of `setup` averaged over the seeds of one group."""
    return numpy.mean([_r2_boot(c[setup][1], c[setup][2], idx) for c in cells], axis=0)


def contrast_rows(
    results: list[RunResult],
    reference: str,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Row]:
    """Group minus the `reference` group, in `R^2`, per setup (images shared).

    The held-out images are the same in every run, so the resamples of the two
    groups are paired by image; seeds of the two groups are averaged separately.
    """
    groups = _by_group(results)
    base = groups[reference]
    base_cells = [_control_cells(r) for r in base]
    out: list[Row] = []
    n = next(iter(base_cells[0].values()))[1].size
    for group, runs in groups.items():
        if group == reference:
            continue
        cells = [_control_cells(r) for r in runs]
        draws = randomization.bootstrap_draws(n, 1, num_bootstrap, rng)[0]

        for setup in cells[0]:
            diff = _mean_boot_r2(cells, setup, draws) - _mean_boot_r2(
                base_cells, setup, draws
            )
            point = float(
                numpy.mean([_r2_point(c[setup][1], c[setup][2]) for c in cells])
                - numpy.mean([_r2_point(c[setup][1], c[setup][2]) for c in base_cells])
            )
            out.append(
                {
                    "group": group,
                    "minus": reference,
                    "setup": setup,
                    "difference in r2": point,
                    "95% CI (images)": _ci(diff),
                }
            )
    return out


def smooth_rows(
    results: list[RunResult], num_bootstrap: int, rng: numpy.random.Generator
) -> list[Row]:
    """Per group and family: `R^2` of random functions vs the observed `R^2`."""
    out: list[Row] = []
    for group, runs in _by_group(results).items():
        families: list[str] = runs[0].info["smooth"]["families"]
        draws = randomization.bootstrap_draws(
            int(runs[0].info["smooth"]["draws"]), len(runs), num_bootstrap, rng
        )
        observed = [float(r.info["smooth"]["observed_r2"]) for r in runs]
        taus = [float(r.info["smooth"]["tau_hat"]) for r in runs]
        eps = [float(r.info["smooth"]["noise_fraction"]) for r in runs]
        for family in families:
            vals = [r.arrays[f"smooth|{family}"] for r in runs]
            boot = randomization.bootstrap_seed_mean(
                draws, lambda s, idx, v=vals: v[s][idx].mean(axis=1)
            )
            above = [
                float((v >= o).mean()) for v, o in zip(vals, observed, strict=True)
            ]
            out.append(
                {
                    "group": group,
                    "family": family,
                    "seeds": len(runs),
                    "r2 of random functions": float(
                        numpy.mean([v.mean() for v in vals])
                    ),
                    "95% CI (draws)": _ci(boot),
                    "draw sd (mean over seeds)": float(
                        numpy.mean([v.std(ddof=1) for v in vals])
                    ),
                    "observed r2 (same views)": float(numpy.mean(observed)),
                    "share of draws >= observed": float(numpy.mean(above)),
                    "tau_hat": float(numpy.mean(taus)),
                    "noise fraction": float(numpy.mean(eps)),
                }
            )
    return out


def curve_rows(
    results: list[RunResult], num_bootstrap: int, rng: numpy.random.Generator
) -> list[Row]:
    """Per group, setup and N: held-out `R^2` with a bootstrap over images."""
    out: list[Row] = []
    for group, runs in _by_group(results).items():
        sizes: list[int] = runs[0].info["curve"]["sizes"]
        for setup in CURVE_SETUPS:
            y = [r.arrays[f"curve|{setup}|y"] for r in runs]
            draws = randomization.bootstrap_draws(
                y[0].size, len(runs), num_bootstrap, rng
            )
            for n in sizes:
                sq = [r.arrays[f"curve|{setup}|{n}|sq"] for r in runs]
                boot = randomization.bootstrap_seed_mean(
                    draws, lambda s, idx, sq=sq, y=y: _r2_boot(y[s], sq[s], idx)
                )
                per_seed = [
                    1.0 - float(a.mean() / b.var()) for a, b in zip(sq, y, strict=True)
                ]
                out.append(
                    {
                        "group": group,
                        "setup": setup,
                        "N": n,
                        "seeds": len(runs),
                        "r2 (mean over seeds)": float(numpy.mean(per_seed)),
                        "95% CI (held-out images)": _ci(boot),
                        "seed range": f"[{min(per_seed):.4f}, {max(per_seed):.4f}]",
                    }
                )
    return out


def saturation_rows(
    results: list[RunResult],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Row]:
    """`R^2(N) - R^2(4000)` per group and setup, paired over held-out images.

    The verdict is the E1 one with the margin `eval.curve_margin`: `equivalent`
    = N = 4000 is saturated to within the margin.
    """
    out: list[Row] = []
    for group, runs in _by_group(results).items():
        sizes: list[int] = runs[0].info["curve"]["sizes"]
        for setup in CURVE_SETUPS:
            y = [r.arrays[f"curve|{setup}|y"] for r in runs]
            draws = randomization.bootstrap_draws(
                y[0].size, len(runs), num_bootstrap, rng
            )

            sq = {n: [r.arrays[f"curve|{setup}|{n}|sq"] for r in runs] for n in sizes}
            for n in (m for m in sizes if m != REFERENCE_N):
                boot = randomization.bootstrap_seed_mean(
                    draws,
                    lambda s, idx, n=n, y=y, sq=sq: (
                        _r2_boot(y[s], sq[n][s], idx)
                        - _r2_boot(y[s], sq[REFERENCE_N][s], idx)
                    ),
                )
                point = float(
                    numpy.mean(
                        [
                            _r2_point(y[s], sq[n][s])
                            - _r2_point(y[s], sq[REFERENCE_N][s])
                            for s in range(len(runs))
                        ]
                    )
                )
                ci90, ci95 = randomization.intervals(boot)
                out.append(
                    {
                        "group": group,
                        "setup": setup,
                        "contrast": f"R^2({n}) - R^2({REFERENCE_N})",
                        "difference in r2": point,
                        "90% CI": f"[{ci90[0]:+.4f}, {ci90[1]:+.4f}]",
                        "95% CI": f"[{ci95[0]:+.4f}, {ci95[1]:+.4f}]",
                        "verdict": randomization.verdict(ci90, ci95, margin),
                    }
                )
    return out


def list_runs(embed_dir: str | os.PathLike[str], prefixes: list[str]) -> list[str]:
    """Runs `<prefix>_..._s<seed>` with both STL-10 dumps present, sorted by name."""
    root = pathlib.Path(embed_dir)
    found: list[str] = []
    for path in sorted(root.iterdir()):
        if not any(path.name.startswith(f"{p}_") for p in prefixes):
            continue
        if not _RUN_NAME.match(path.name):
            continue
        if all(any((path / s).glob("*.npz")) for s in ("stl10_train", "stl10_test")):
            found.append(path.name)
    return found


def write_notes(
    results: list[RunResult],
    cfg: config.EvalConfig,
    note_base: pathlib.Path,
    reference: str,
    seed: int,
    chrome: str | os.PathLike[str] | None,
    figures: bool = True,
) -> list[pathlib.Path]:
    """Writes the generated sections of the three E2 control notes.

    Args:
        results: One per run (all parts present).
        cfg: Settings (`num_bootstrap`, `curve_margin`).
        note_base: `<dir>/controls.md`; the notes are `<dir>/controls_redundancy.md`,
            `controls_smooth.md` and `controls_curve.md` (created from the template
            if absent; hand-written text is kept).
        reference: Group the other groups are contrasted with in the redundancy note.
        seed: Seed of the bootstrap resamples.
        chrome: Chrome binary for the figure export.
        figures: Whether to export the learning-curve PNGs.

    Returns:
        The note paths.
    """

    def note(suffix: str) -> pathlib.Path:
        return note_base.with_name(f"{note_base.stem}_{suffix}.md")

    boots = cfg.num_bootstrap
    rng = numpy.random.default_rng(seed)
    paths_out: list[pathlib.Path] = []

    red = note("redundancy")
    reporting.update_section(
        red,
        "baseline.controls",
        "### Redundancy controls per group (z)\n\nHeld-out nested-CV kernel-ridge "
        "R^2 of the radius from the direction on the first "
        f"{cfg.max_kernel_images} `stl10_test` images, mean over the seeds of a "
        "group; the interval is a 95% percentile bootstrap over images (resampled "
        "within each seed, then averaged over seeds). `null max`, `worst "
        f"permutation p` ({cfg.num_permutations} permutations) and `min F` are the "
        "least favorable value over seeds.\n\n"
        + reporting.markdown_table(controls_rows(results, boots, rng)),
    )
    groups = {r.group for r in results}
    if reference in groups:
        reporting.update_section(
            red,
            "baseline.contrast",
            f"### R^2 minus the `{reference}` group\n\nSame held-out images, so "
            "the resamples are shared by image; 95% percentile bootstrap, seeds "
            "averaged within each group.\n\n"
            + reporting.markdown_table(contrast_rows(results, reference, boots, rng)),
        )
    paths_out.append(red)

    smooth = note("smooth")
    reporting.update_section(
        smooth,
        "baseline.smooth",
        "### Held-out R^2 of random smooth functions of the direction\n\n"
        "Targets are drawn on the real `u_hat` of the same images and predicted by "
        "the identical nested-CV kernel ridge. `tau_hat` is the median bandwidth "
        "the fit selected for the real log-radius; `noise fraction` is "
        "1 - (Spearman-Brown reliability of rho_bar). The interval is a 95% "
        "percentile bootstrap over draws (seeds averaged).\n\n"
        + reporting.markdown_table(smooth_rows(results, boots, rng)),
    )
    paths_out.append(smooth)

    curve = note("curve")
    reporting.update_section(
        curve,
        "baseline.curve",
        "### Held-out R^2 against the number of training images\n\n"
        "One fixed held-out set (the same images in every run), nested random "
        "training subsets of the rest, 95% percentile bootstrap over held-out "
        "images (seeds averaged). `same views`: u_hat and rho_bar of all 8 views; "
        "`disjoint halves`: u_hat of views 0-3, rho_bar of views 4-7.\n\n"
        + reporting.markdown_table(curve_rows(results, boots, rng)),
    )
    reporting.update_section(
        curve,
        "baseline.saturation",
        f"### Is N = {REFERENCE_N} saturating?\n\nDifference of the held-out R^2 "
        f"at N and at N = {REFERENCE_N} (paired over held-out images). Margin "
        f"{cfg.curve_margin:g} R^2 (`eval.curve_margin`, fixed before any curve "
        "was computed); `equivalent` means the 90% interval is inside the margin "
        "(saturated to within it), `different` means the 95% interval excludes 0.\n\n"
        + reporting.markdown_table(
            saturation_rows(results, cfg.curve_margin, boots, rng)
        ),
    )
    if figures:
        prefix = curve.parent / curve.stem
        links: list[str] = []
        rows = curve_rows(results, boots, numpy.random.default_rng(seed))
        for setup in CURVE_SETUPS:
            series: dict[str, plotting.Series] = {}
            for group in sorted(groups):
                sel = [r for r in rows if r["group"] == group and r["setup"] == setup]
                lows = [
                    float(str(r["95% CI (held-out images)"]).strip("[]").split(",")[0])
                    for r in sel
                ]
                highs = [
                    float(str(r["95% CI (held-out images)"]).strip("[]").split(",")[1])
                    for r in sel
                ]
                series[group] = plotting.Series(
                    x=[float(r["N"]) for r in sel],  # pyright: ignore[reportArgumentType]
                    y=[float(r["r2 (mean over seeds)"]) for r in sel],  # pyright: ignore[reportArgumentType]
                    low=lows,
                    high=highs,
                )
            fig = plotting.line_plot(
                series,
                f"Held-out R^2 of the radius from the direction ({setup})",
                "training images N",
                "held-out R^2",
                log_x=True,
            )
            slug = setup.replace(" ", "_")
            out = plotting.export_png(fig, f"{prefix}_{slug}.png", chrome)
            links.append(reporting.image_link(out, curve, f"learning curve {setup}"))
        reporting.update_section(
            curve, "baseline.figures", "### Figures\n\n" + "\n\n".join(links)
        )
    paths_out.append(curve)
    return paths_out
