"""Sweep-level summaries of training logs: group runs by lambda, aggregate seeds.

A sweep is a set of run directories `<runs_dir>/<prefix>_*/`, each holding the
`config.resolved.yaml` and `metrics.jsonl` written by `apps/train.py`. Runs are
grouped by the `loss.lambda` of their *resolved config* (never by parsing names).
Curves are taken over the epochs that **every** seed of a lambda has reached, so a
partially finished sweep is summarized honestly; the per-run table shows how far
each run got.

Targets under an exact N(0, I_d) projector marginal: `kappa = 1`, `||z||` mean
`sqrt(d - 1/2)` and standard deviation `sqrt(1/2)` (chi_d, large-d).
"""

import dataclasses
import math
import os
import pathlib
import typing

import numpy

from lejepa_norm import config
from lejepa_norm.eval import runlog

KAPPA_Z = "kappa_z_epoch_mean"
Z_NORM_MEAN = "z_norm_mean_epoch_mean"
Z_NORM_STD = "z_norm_std_epoch_mean"
# (key in the epoch records, column title), in table order.
SUMMARY_KEYS = (
    (KAPPA_Z, "kappa_z"),
    ("kappa_h_epoch_mean", "kappa_h"),
    (Z_NORM_MEAN, "z_norm_mean"),
    (Z_NORM_STD, "z_norm_std"),
    ("knn_acc_h", "knn_acc_h"),
    ("knn_acc_z", "knn_acc_z"),
    ("loss_dist_epoch_mean", "loss_dist"),
    ("loss_inv_epoch_mean", "loss_inv"),
)


@dataclasses.dataclass
class Run:
    """One training run of a sweep.

    Attributes:
        name: Run directory name.
        lam: SIGReg weight (`loss.lambda`).
        seed: Training seed.
        embed_dim: Projector output dimension `d`.
        target_epochs: Epochs the run was configured for.
        epochs: Per-epoch numeric values of the epoch records, keyed by epoch.
    """

    name: str
    lam: float
    seed: int
    embed_dim: int
    target_epochs: int
    epochs: dict[int, dict[str, float]]

    @property
    def last_epoch(self) -> int:
        """Last epoch with an epoch record (0 if none)."""
        return max(self.epochs, default=0)

    @property
    def complete(self) -> bool:
        """Whether the run reached its configured number of epochs."""
        return self.last_epoch >= self.target_epochs


@dataclasses.dataclass
class Curve:
    """A metric aggregated over seeds: mean with the min-max range."""

    epochs: list[float]
    mean: list[float]
    low: list[float]
    high: list[float]


def _numeric(value: typing.Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def load_runs(runs_dir: str | os.PathLike[str], prefix: str) -> list[Run]:
    """Loads every run `<prefix>_*` that has a resolved config.

    Runs without a `metrics.jsonl` are returned with no epochs, so they show up as
    "not started" instead of disappearing.

    Returns:
        Runs sorted by (lambda, seed).
    """
    runs: list[Run] = []
    for directory in sorted(pathlib.Path(runs_dir).glob(f"{prefix}_*")):
        resolved = directory / "config.resolved.yaml"
        if not resolved.is_file():
            continue
        cfg = config.load_config(resolved)
        metrics = directory / runlog.METRICS_FILE
        records = runlog.read_metrics(metrics) if metrics.exists() else []
        epochs: dict[int, dict[str, float]] = {}
        for record in records:
            if record.get("kind") == "epoch" and _numeric(record.get("epoch")):
                epochs[int(record["epoch"])] = {
                    k: float(v) for k, v in record.items() if _numeric(v)
                }
        runs.append(
            Run(
                name=directory.name,
                lam=cfg.loss.lambda_,
                seed=cfg.seed,
                embed_dim=cfg.model.embed_dim,
                target_epochs=cfg.train.epochs,
                epochs=epochs,
            )
        )
    return sorted(runs, key=lambda r: (r.lam, r.seed))


def by_lambda(runs: list[Run]) -> dict[float, list[Run]]:
    """Groups runs by lambda, in increasing lambda order."""
    groups: dict[float, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.lam, []).append(run)
    return dict(sorted(groups.items()))


def curve(runs: list[Run], key: str) -> Curve:
    """Mean and min-max over seeds, on the epochs where every run has `key`."""
    if not runs:
        return Curve([], [], [], [])
    common: set[int] = {e for e, values in runs[0].epochs.items() if key in values}
    for run in runs[1:]:
        common &= {e for e, values in run.epochs.items() if key in values}
    epochs = sorted(common)
    if not epochs:
        return Curve([], [], [], [])
    values = numpy.array([[r.epochs[e][key] for e in epochs] for r in runs])
    return Curve(
        epochs=[float(e) for e in epochs],
        mean=values.mean(axis=0).tolist(),
        low=values.min(axis=0).tolist(),
        high=values.max(axis=0).tolist(),
    )


def _final(runs: list[Run], key: str) -> tuple[int, numpy.ndarray] | None:
    """Values of `key` at the last epoch that every run has, or None."""
    c = curve(runs, key)
    if not c.epochs:
        return None
    last = int(c.epochs[-1])
    return last, numpy.array([r.epochs[last][key] for r in runs])


def _mean_std(values: numpy.ndarray) -> str:
    mean = float(values.mean())
    if len(values) < 2:
        return f"{mean:.4g}"
    std = float(values.std(ddof=1))
    if std < 1e-9 * max(1.0, abs(mean)):  # float noise on identical values
        std = 0.0
    return f"{mean:.4g} ± {std:.2g}"


def summary_rows(runs: list[Run]) -> list[dict[str, str | int]]:
    """One row per lambda (mean ± std over seeds at the last common epoch).

    A first row gives the targets of an exact N(0, I_d) marginal. The `epoch`
    column is the last epoch every seed of that lambda has reached (by `kappa_z`,
    which is logged every epoch). Sparsely logged metrics (k-NN) are taken at their
    own last common epoch.
    """
    rows: list[dict[str, str | int]] = []
    if not runs:
        return rows
    d = runs[0].embed_dim
    rows.append(
        {
            "lambda": "target (N(0, I))",
            "seeds": "",
            "epoch": "",
            "kappa_z": "1",
            "z_norm_mean": f"{math.sqrt(d - 0.5):.4g}",
            "z_norm_std": f"{math.sqrt(0.5):.4g}",
        }
    )
    for lam, group in by_lambda(runs).items():
        row: dict[str, str | int] = {"lambda": f"{lam:g}", "seeds": len(group)}
        for key, title in SUMMARY_KEYS:
            final = _final(group, key)
            if final is not None:
                row[title] = _mean_std(final[1])
        kappa = _final(group, KAPPA_Z)  # logged every epoch
        row["epoch"] = kappa[0] if kappa is not None else 0
        rows.append(row)
    return rows


def run_rows(runs: list[Run]) -> list[dict[str, str | int | float]]:
    """One row per run: progress and its latest values."""
    rows: list[dict[str, str | int | float]] = []
    for run in runs:
        if not run.epochs:
            status = "no epoch records"
        elif run.complete:
            status = "complete"
        else:
            status = "partial"
        row: dict[str, str | int | float] = {
            "run": run.name,
            "lambda": run.lam,
            "seed": run.seed,
            "epochs": f"{run.last_epoch}/{run.target_epochs}",
            "status": status,
        }
        for key, title in SUMMARY_KEYS:
            # Latest available value (k-NN is only logged every few epochs).
            for epoch in sorted(run.epochs, reverse=True):
                if key in run.epochs[epoch]:
                    row[title] = run.epochs[epoch][key]
                    break
        rows.append(row)
    return rows
