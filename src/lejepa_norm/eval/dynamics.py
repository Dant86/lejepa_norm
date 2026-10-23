"""Redundancy of the radius across training: the epoch sweep.

For one run, the `stl10_test` embeddings of every epoch checkpoint (dumps under
`<embed_dir>/radius_dynamics/<run>/epoch_XXXX/stl10_test`, written by
`apps/embed.py` with `dump_name`) are reduced to one row of statistics per epoch,
all in the projector space `z` (LeJEPA's product; `h` is not analyzed):

* `kappa_z`: `Var(||z||^2) / (2 d)` on the augmented views of all test images
  (`metrics.kappa`; 1 for N(0, I)); the center-view value is reported too.
* `icc`: ICC(1) of the log-radius over the augmented views.
* `reliability`: split-half reliability of `rho_bar` (`controls`).
* `r2_same`: held-out R^2 of `rho_bar` from `u_hat`, both from all augmented views.
* `r2_cross`: the same with disjoint view halves (mean of the two directions).

The R^2 values use the nested-CV kernel ridge of `controls`/`redundancy` on one
fixed random subset of images (the same images at every epoch). The permutation
null is run with only a few repetitions (`num_permutations`); it is a sanity check
here, not a test.

The work is split into independent, resumable units: one (run, epoch) is one
`run_unit` call that writes one JSON shard `<embed_dir>/radius_dynamics_rows/<run>/
epoch_XXXX.json` (atomically; an existing shard computed with the same settings is
reused). `collect` reads the shards back for the cheap final aggregation. Each unit
can therefore be its own SLURM job (`slurm/analyze_epochs.sbatch`).
"""

import collections.abc
import dataclasses
import json
import os
import pathlib
import re
import time

import numpy
import numpy.typing

from lejepa_norm import config
from lejepa_norm.eval import analysis
from lejepa_norm.eval import controls
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import metrics
from lejepa_norm.eval import shards

Row = analysis.Row
FloatArray = numpy.typing.NDArray[numpy.float64]

DUMP_ROOT = "radius_dynamics"
SHARD_ROOT = "radius_dynamics_rows"
Progress = collections.abc.Callable[[str], None]
# Row keys, in table order.
METRICS = (
    "kappa_z",
    "kappa_z_center",
    "icc",
    "reliability",
    "r2_same",
    "r2_cross",
    "null_max",
)
_EPOCH_DIR = re.compile(r"epoch_(\d+)$")
_CHECKPOINT = re.compile(r"epoch_(\d+)\.pt$")


def print_progress(message: str) -> None:
    """Default progress sink: a UTC-stamped line on stdout, flushed at once."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{stamp}] {message}", flush=True)


def _no_progress(message: str) -> None:
    del message


def unit_path(embed_dir: str | os.PathLike[str], run: str, epoch: int) -> pathlib.Path:
    """The JSON shard of one (run, epoch) unit."""
    return pathlib.Path(embed_dir) / SHARD_ROOT / run / f"epoch_{epoch:04d}.json"


def dump_name(run: str, epoch: int) -> str:
    """The `apps/embed.py` run name under which one epoch's dump is written."""
    return f"{DUMP_ROOT}/{run}/epoch_{epoch:04d}"


def checkpoint_epoch(path: str | os.PathLike[str]) -> int:
    """Epoch of a periodic checkpoint file name `epoch_XXXX.pt`.

    Raises:
        ValueError: The file name does not match.
    """
    match = _CHECKPOINT.search(pathlib.Path(path).name)
    if match is None:
        raise ValueError(f"{path} is not an epoch_XXXX.pt checkpoint")
    return int(match.group(1))


def available_epochs(embed_dir: str | os.PathLike[str], run: str) -> list[int]:
    """Epochs that have a `stl10_test` dump for `run`, ascending."""
    root = pathlib.Path(embed_dir) / DUMP_ROOT / run
    epochs = [
        int(m.group(1))
        for d in root.glob("epoch_*")
        if (m := _EPOCH_DIR.match(d.name)) is not None
        and any((d / dumps.STL10_TEST).glob("*.npz"))
    ]
    return sorted(epochs)


def epoch_row(
    test: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    seed: int,
    num_permutations: int,
    progress: Progress = _no_progress,
) -> Row:
    """All statistics of one epoch's `stl10_test` dump.

    Args:
        test: The dump (at least four augmented views, for the half splits).
        cfg: Analysis settings (`max_kernel_images`, `cv_folds`).
        seed: Seeds the image subset (identical for every epoch), the fold
            splits and the permutations.
        num_permutations: Permutation-null repetitions per regression.
        progress: Receives one message per regression setup, with its seconds.

    Returns:
        The metrics of `METRICS` (without the epoch and seed columns).
    """
    z = test.z
    aug = features.select_views(z, "augmented")
    if aug.shape[1] < 4:
        raise ValueError("the half-view setups need at least four augmented views")
    rng = numpy.random.default_rng(seed)
    idx = analysis.subsample(test.label.size, cfg.max_kernel_images, rng)
    x = aug[idx]
    run_cfg = dataclasses.replace(cfg, num_permutations=num_permutations)
    setups = controls.view_setups(x)[:3]  # same views and the two disjoint halves
    rows: list[Row] = []
    for setup in setups:
        start = time.perf_counter()
        rows.append(controls.run_setup(setup, run_cfg, rng))
        progress(f"setup {setup.name!r}: {time.perf_counter() - start:.1f} s")
    r2 = [float(r[controls.R2]) for r in rows]
    nulls = [float(r["null max"]) for r in rows if int(r["permutations"]) > 0]
    return {
        "kappa_z": metrics.kappa(aug),
        "kappa_z_center": metrics.kappa(features.select_views(z, "center")),
        "icc": metrics.icc1(features.log_radius(aug)).icc.value,
        "reliability": controls.split_half_reliability(x),
        "r2_same": r2[0],
        "r2_cross": float(numpy.mean(r2[1:])),
        "null_max": max(nulls) if nulls else float("nan"),
    }


def _settings(cfg: config.EvalConfig, seed: int, num_permutations: int) -> Row:
    return {
        "seed": seed,
        "max_kernel_images": cfg.max_kernel_images,
        "cv_folds": cfg.cv_folds,
        "num_permutations": num_permutations,
    }


def _read_shard(path: pathlib.Path, settings: Row) -> Row | None:
    """The row of a finished unit, or None if absent or computed differently."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:  # a torn file from a killed writer
        return None
    return payload["row"] if payload.get("settings") == settings else None


def run_unit(
    embed_dir: str | os.PathLike[str],
    run: str,
    epoch: int,
    cfg: config.EvalConfig,
    seed: int,
    num_permutations: int,
    progress: Progress = print_progress,
) -> Row:
    """Computes (or reuses) the shard of one (run, epoch) unit.

    Args:
        embed_dir: Embedding root holding the dumps and, written here, the shards.
        run: Run name.
        epoch: Epoch of the dump.
        cfg: Analysis settings.
        seed: See `epoch_row`.
        num_permutations: See `epoch_row`.
        progress: Receives progress messages.

    Returns:
        The row, with the epoch column first.

    Raises:
        FileNotFoundError: The unit has no `stl10_test` dump.
    """
    path = unit_path(embed_dir, run, epoch)
    settings = _settings(cfg, seed, num_permutations)
    done = _read_shard(path, settings)
    if done is not None:
        progress(f"{run} epoch {epoch}: shard exists, skipped")
        return done
    directory = dumps.set_dir(embed_dir, dump_name(run, epoch), dumps.STL10_TEST)
    shard = dumps.load_set(directory)
    if shard is None:
        raise FileNotFoundError(f"no stl10_test dump for {run} epoch {epoch}")
    progress(f"{run} epoch {epoch}: loaded {shard.z.shape}")
    start = time.perf_counter()
    row: Row = {
        "epoch": epoch,
        **epoch_row(shard, cfg, seed, num_permutations, progress),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"settings": settings, "row": row}))
    tmp.replace(path)
    progress(f"{run} epoch {epoch}: done in {time.perf_counter() - start:.1f} s")
    return row


def run_epoch_sweep(
    embed_dir: str | os.PathLike[str],
    run: str,
    cfg: config.EvalConfig,
    seed: int,
    num_permutations: int,
    progress: Progress = print_progress,
) -> list[Row]:
    """`run_unit` for every available epoch dump of `run`, sequentially.

    Raises:
        FileNotFoundError: `run` has no epoch dumps.
    """
    epochs = available_epochs(embed_dir, run)
    if not epochs:
        raise FileNotFoundError(f"no epoch dumps for {run} under {embed_dir}")
    return [
        run_unit(embed_dir, run, epoch, cfg, seed, num_permutations, progress)
        for epoch in epochs
    ]


def collect(
    embed_dir: str | os.PathLike[str],
    runs: collections.abc.Sequence[str],
    cfg: config.EvalConfig,
    seed: int,
    num_permutations: int,
    progress: Progress = print_progress,
) -> dict[str, list[Row]]:
    """Rows of the finished units of each run (ascending epochs), from the shards.

    Units with missing or differently-configured shards are reported through
    `progress` and left out (`curve` uses the epochs every run shares).

    Raises:
        FileNotFoundError: A run has no finished unit.
    """
    settings = _settings(cfg, seed, num_permutations)
    result: dict[str, list[Row]] = {}
    for run in runs:
        rows: list[Row] = []
        for epoch in available_epochs(embed_dir, run):
            row = _read_shard(unit_path(embed_dir, run, epoch), settings)
            if row is None:
                progress(f"{run} epoch {epoch}: no valid shard, left out")
            else:
                rows.append(row)
        if not rows:
            raise FileNotFoundError(f"no finished epoch units for {run}")
        result[run] = rows
    return result


@dataclasses.dataclass(frozen=True)
class Curve:
    """One metric over epochs, mean and sample std over runs (seeds).

    Attributes:
        epochs: Epochs present in every run, ascending.
        mean: Mean over runs per epoch.
        std: Sample std (ddof 1; 0 for a single run) per epoch.
    """

    epochs: list[int]
    mean: FloatArray
    std: FloatArray


def curve(runs: dict[str, list[Row]], key: str) -> Curve:
    """Mean and std over runs of `key`, on the epochs all runs share."""
    per_run = [{int(r["epoch"]): float(r[key]) for r in rows} for rows in runs.values()]
    shared: set[int] = set(per_run[0])
    for p in per_run[1:]:
        shared &= set(p)
    epochs = sorted(shared)
    values = numpy.array([[p[e] for e in epochs] for p in per_run])
    std = values.std(axis=0, ddof=1) if len(per_run) > 1 else numpy.zeros(len(epochs))
    return Curve(epochs, values.mean(axis=0), std)


def aggregate(runs: dict[str, list[Row]]) -> list[Row]:
    """Table by epoch: mean ± std over runs of every metric in `METRICS`.

    The permutation null is summarized by its largest value over runs.
    """
    value_keys = [k for k in METRICS if k != "null_max"]
    curves = {key: curve(runs, key) for key in value_keys}
    epochs = curves[value_keys[0]].epochs
    rows: list[Row] = []
    for i, epoch in enumerate(epochs):
        row: Row = {"epoch": epoch, "seeds": len(runs)}
        for key, c in curves.items():
            row[key] = (
                f"{c.mean[i]:.4g} ± {c.std[i]:.2g}"
                if len(runs) > 1
                else f"{c.mean[i]:.4g}"
            )
        row["null max (max over seeds)"] = max(
            float(r["null_max"])
            for rows_of_run in runs.values()
            for r in rows_of_run
            if int(r["epoch"]) == epoch
        )
        rows.append(row)
    return rows
