"""Does the radius shift under corruption carry information beyond the direction?

A kernel-ridge map `u_hat -> rho_bar` is fitted on clean test images only
(hyperparameters by leave-one-out CV on those images, see
`redundancy.FittedKernelRidge`) and then applied to the paired corrupted images
using *their own* directions. If the radius is a function of the direction, the map
transfers: the corrupted radius is predicted as well as the clean one and the mean
residual (observed - predicted) is close to zero, so the raw radius shift is just
the direction moving. If the residual keeps the raw shift, the radius carries
shift information the direction does not.

Leakage control: the clean images are split into a fit half and a held-out half.
The map is fitted on the fit half; cells are evaluated on the corrupted versions of
the **held-out** images (neither the image nor its twin entered the fit), and,
for comparison, on the corrupted versions of the fit images (whose clean twin the
map has seen). The reference for "transfer without any shift" is the same map
applied to the clean held-out images.

Caveat built into the design: a corrupted direction may lie where the clean fit has
no support, so a failure can also be an extrapolation failure. `nn_cos` (mean
cosine of a direction to its nearest fit direction) is reported to flag that.
Bootstrap CIs resample images with the fitted map held fixed (they do not include
the uncertainty of the fit).
"""

import dataclasses
import re

import numpy
import numpy.typing

from lejepa_norm import config
from lejepa_norm.eval import analysis
from lejepa_norm.eval import difficulty
from lejepa_norm.eval import features
from lejepa_norm.eval import metrics
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
Row = analysis.Row


@dataclasses.dataclass(frozen=True)
class TransferCell:
    """Transfer statistics of one (corruption, severity) cell.

    Attributes:
        corruption: Corruption name.
        severity: Severity level.
        n_held: Corrupted images whose clean twin was held out of the fit.
        raw_shift: Mean `rho_bar(corrupted) - rho_bar(clean twin)` on those.
        predicted_shift: Mean `f(u_corrupted) - f(u_clean twin)` on those.
        residual_held: Mean `rho_bar(corrupted) - f(u_corrupted)` on those.
        r2_held: Transferred `R^2` on those (around the cell's own mean).
        nn_cos: Mean nearest-fit-direction cosine of those corrupted directions.
        n_fit: Corrupted images whose clean twin was in the fit.
        residual_fit: Mean residual on those.
        r2_fit: Transferred `R^2` on those.
    """

    corruption: str
    severity: int
    n_held: int
    raw_shift: metrics.Estimate
    predicted_shift: float
    residual_held: metrics.Estimate
    r2_held: float
    nn_cos: float
    n_fit: int
    residual_fit: metrics.Estimate
    r2_fit: float


@dataclasses.dataclass(frozen=True)
class TransferResult:
    """Cells plus the clean reference of one run.

    Attributes:
        cells: One entry per (corruption, severity), sorted.
        n_fit: Clean images the map was fitted on.
        n_held: Clean images held out.
        tau: Selected kernel bandwidth.
        alpha: Selected ridge penalty.
        clean_r2: `R^2` of the map on the held-out clean images.
        clean_residual: Mean residual on the held-out clean images (should be 0).
        clean_nn_cos: Mean nearest-fit-direction cosine of the held-out clean
            directions.
    """

    cells: list[TransferCell]
    n_fit: int
    n_held: int
    tau: float
    alpha: float
    clean_r2: float
    clean_residual: metrics.Estimate
    clean_nn_cos: float


def _r2(y: FloatArray, residual: FloatArray) -> float:
    ss_tot = float(numpy.square(y - y.mean()).sum())
    return 1.0 - float(numpy.square(residual).sum()) / ss_tot if ss_tot > 0 else 0.0


def _nn_cos(u: FloatArray, bank: FloatArray, chunk: int = 2048) -> float:
    """Mean over rows of `u` of the largest cosine to any row of `bank`."""
    best = [(u[i : i + chunk] @ bank.T).max(axis=1) for i in range(0, len(u), chunk)]
    return float(numpy.concatenate(best).mean()) if best else float("nan")


def _mean(
    values: FloatArray, num_bootstrap: int, rng: numpy.random.Generator
) -> metrics.Estimate:
    return metrics.bootstrap(
        lambda idx: float(values[idx].mean()), values.size, num_bootstrap, rng
    )


def run_transfer(
    test: shards.EmbeddingShard,
    corrupt: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    seed: int = 0,
) -> TransferResult:
    """Fits the clean map on half of the test images and transfers it.

    Only the projector output `z` is used. Image-level quantities come from the
    augmented views (`rho_bar`, `u_hat`), as in the redundancy analysis.

    Args:
        test: Clean test images (unique `image_id`).
        corrupt: Corrupted versions; `image_id` indexes `test.image_id`.
        cfg: `max_kernel_images` caps the fit size (fit half is
            `min(max_kernel_images, N // 2)`), `num_bootstrap` for the CIs.
        seed: Seed of the split, bandwidth estimate and bootstraps.

    Returns:
        Per-cell statistics and the clean reference.
    """
    rng = numpy.random.default_rng(seed)
    rho_clean, u_clean = features.mean_log_radius_and_direction(test.z)
    n = rho_clean.size
    n_fit = max(1, min(cfg.max_kernel_images, n // 2))
    perm = rng.permutation(n)
    fit_idx, held_idx = numpy.sort(perm[:n_fit]), numpy.sort(perm[n_fit:])
    bank = u_clean[fit_idx]
    model = redundancy.FittedKernelRidge(bank, rho_clean[fit_idx], rng)
    pred_clean = model.predict(u_clean)
    is_fit = numpy.zeros(n, dtype=bool)
    is_fit[fit_idx] = True

    rho_c, u_c = features.mean_log_radius_and_direction(corrupt.z)
    twin = difficulty.twin_index(test.image_id, corrupt.image_id)
    pred_c = model.predict(u_c)
    nb = cfg.num_bootstrap
    names = numpy.asarray(corrupt.corruption)
    levels = numpy.asarray(corrupt.severity)

    cells: list[TransferCell] = []
    for name in sorted(set(names.tolist())):
        for level in sorted(set(levels[names == name].tolist())):
            cell = (names == name) & (levels == level)
            held = cell & ~is_fit[twin]
            fit = cell & is_fit[twin]
            res_held = rho_c[held] - pred_c[held]
            res_fit = rho_c[fit] - pred_c[fit]
            cells.append(
                TransferCell(
                    corruption=str(name),
                    severity=int(level),
                    n_held=int(held.sum()),
                    raw_shift=_mean(rho_c[held] - rho_clean[twin[held]], nb, rng),
                    predicted_shift=float(
                        (pred_c[held] - pred_clean[twin[held]]).mean()
                    ),
                    residual_held=_mean(res_held, nb, rng),
                    r2_held=_r2(rho_c[held], res_held),
                    nn_cos=_nn_cos(u_c[held], bank),
                    n_fit=int(fit.sum()),
                    residual_fit=_mean(res_fit, nb, rng),
                    r2_fit=_r2(rho_c[fit], res_fit),
                )
            )
    res_clean = rho_clean[held_idx] - pred_clean[held_idx]
    return TransferResult(
        cells=cells,
        n_fit=int(fit_idx.size),
        n_held=int(held_idx.size),
        tau=model.tau,
        alpha=model.alpha,
        clean_r2=_r2(rho_clean[held_idx], res_clean),
        clean_residual=_mean(res_clean, nb, rng),
        clean_nn_cos=_nn_cos(u_clean[held_idx], bank),
    )


def table_rows(result: TransferResult) -> list[Row]:
    """One row per cell, then the clean held-out reference row."""
    rows: list[Row] = [
        {
            "corruption": c.corruption,
            "severity": c.severity,
            "N held-out": c.n_held,
            "raw shift": analysis.format_estimate(c.raw_shift),
            "predicted shift": f"{c.predicted_shift:.3f}",
            "residual (held-out twins)": analysis.format_estimate(c.residual_held),
            "R^2 (held-out twins)": c.r2_held,
            "nn_cos": c.nn_cos,
            "residual (fit twins)": analysis.format_estimate(c.residual_fit),
            "R^2 (fit twins)": c.r2_fit,
        }
        for c in result.cells
    ]
    rows.append(
        {
            "corruption": "clean (reference)",
            "severity": 0,
            "N held-out": result.n_held,
            "raw shift": "0",
            "predicted shift": "0",
            "residual (held-out twins)": analysis.format_estimate(
                result.clean_residual
            ),
            "R^2 (held-out twins)": result.clean_r2,
            "nn_cos": result.clean_nn_cos,
            "residual (fit twins)": "",
            "R^2 (fit twins)": "",
        }
    )
    return rows


_LEADING_NUMBER = re.compile(r"^\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _leading_float(cell: str) -> float:
    """The point estimate of a `value [low, high]` table cell."""
    match = _LEADING_NUMBER.match(cell)
    if match is None:
        raise ValueError(f"no number in table cell {cell!r}")
    return float(match.group(1))


def _mean_std(values: list[float]) -> str:
    arr = numpy.asarray(values)
    if arr.size < 2:
        return f"{arr.mean():.3f}"
    return f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"


def aggregate_transfer(runs: dict[float, list[list[dict[str, str]]]]) -> list[Row]:
    """Aggregates the transfer tables of several seeds per lambda.

    Args:
        runs: For each lambda, one parsed `table_rows` table (cells as text, as
            read back from a note) per seed. All tables must list the same cells.

    Returns:
        One row per lambda, corruption and severity: raw shift, residual on
        held-out twins and its transferred `R^2` as mean ± std over seeds, and the
        unexplained fraction `mean residual / mean raw shift` (of the seed means).

    Raises:
        ValueError: The seeds of a lambda list different cells.
    """
    rows: list[Row] = []
    for lam, tables in sorted(runs.items()):
        keys = [(r["corruption"], r["severity"]) for r in tables[0]]
        for table in tables[1:]:
            if [(r["corruption"], r["severity"]) for r in table] != keys:
                raise ValueError(f"lambda={lam}: seeds list different cells")
        for i, (name, level) in enumerate(keys):
            cells = [table[i] for table in tables]
            raw = [_leading_float(c["raw shift"]) for c in cells]
            res = [_leading_float(c["residual (held-out twins)"]) for c in cells]
            mean_raw = float(numpy.mean(raw))
            rows.append(
                {
                    "lambda": lam,
                    "corruption": name,
                    "severity": level,
                    "seeds": len(tables),
                    "raw shift": _mean_std(raw),
                    "residual (held-out twins)": _mean_std(res),
                    "R^2 (held-out twins)": _mean_std(
                        [float(c["R^2 (held-out twins)"]) for c in cells]
                    ),
                    "unexplained fraction": (
                        float(numpy.mean(res)) / mean_raw
                        if abs(mean_raw) > 1e-12
                        else float("nan")
                    ),
                }
            )
    return rows
