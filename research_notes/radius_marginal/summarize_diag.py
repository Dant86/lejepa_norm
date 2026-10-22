"""Joins the per-run marginal-radius diagnostics with the training logs of the sweep.

For each run `<prefix>_lam<lambda>_s<seed>` this reads

* the "Marginal radius (STL-10 test)" table that `apps/analyze.py` wrote into
  `<diag-dir>/<run>.md` (embedding of the final checkpoint on STL-10 test; the
  `augmented` rows use the global-view augmentation pipeline), and
* the training log (`metrics.jsonl`) under `<runs-dir>/<run>/`,

and writes two generated sections into the experiment note: a per-lambda table
(mean +- std over seeds) and a per-run table. Re-running updates them in place.

Usage:
    python research_notes/radius_marginal/summarize_diag.py \
        --diag-dir <dir with the per-run notes> --runs-dir <runs dir> \
        --note research_notes/radius_marginal/lambda_sweep.md
"""

import argparse
import math
import pathlib
import re

import numpy

from lejepa_norm import reporting
from lejepa_norm.eval import sweep

_SECTION = re.compile(
    r"<!-- BEGIN GENERATED: analyze\.marginal -->(.*?)"
    r"<!-- END GENERATED: analyze\.marginal -->",
    re.S,
)
KAPPA_Z_TRAIN = "kappa_z_epoch_mean"
KAPPA_H_TRAIN = "kappa_h_epoch_mean"


def parse_marginal(path: pathlib.Path) -> dict[tuple[str, str], dict[str, float]]:
    """Reads the marginal-radius table of an analyze note.

    Returns:
        `(space, views)` to the numeric columns of that row.

    Raises:
        ValueError: The note has no marginal-radius section.
    """
    match = _SECTION.search(path.read_text())
    if match is None:
        raise ValueError(f"{path}: no analyze.marginal section")
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in match.group(1).splitlines()
        if line.startswith("|")
    ]
    header, body = rows[0], rows[2:]  # rows[1] is the |---| separator
    parsed: dict[tuple[str, str], dict[str, float]] = {}
    for row in body:
        values = dict(zip(header, row, strict=True))
        parsed[(values["space"], values["views"])] = {
            key: float(value)
            for key, value in values.items()
            if key not in ("space", "views")
        }
    return parsed


def _mean_std(values: list[float]) -> str:
    array = numpy.asarray(values, dtype=numpy.float64)
    if len(array) < 2:
        return f"{array.mean():.4g}"
    return f"{array.mean():.4g} ± {array.std(ddof=1):.2g}"


def _last(run: sweep.Run, key: str) -> float:
    for epoch in sorted(run.epochs, reverse=True):
        if key in run.epochs[epoch]:
            return run.epochs[epoch][key]
    return math.nan


def main() -> int:
    """Writes the diagnostic sections into the note."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-dir", required=True, type=pathlib.Path)
    parser.add_argument("--runs-dir", required=True, type=pathlib.Path)
    parser.add_argument("--note", required=True, type=pathlib.Path)
    parser.add_argument("--prefix", default="radius_marginal")
    args = parser.parse_args()

    runs = sweep.load_runs(args.runs_dir, args.prefix)
    per_run: list[dict[str, str | int | float]] = []
    by_lambda: dict[float, list[dict[str, float]]] = {}
    missing: list[str] = []
    for run in runs:
        path = args.diag_dir / f"{run.name}.md"
        if not path.exists():
            missing.append(run.name)
            continue
        table = parse_marginal(path)
        z_center, z_aug = table[("z", "center")], table[("z", "augmented")]
        h_center, h_aug = table[("h", "center")], table[("h", "augmented")]
        record = {
            "kappa_z_train": _last(run, KAPPA_Z_TRAIN),
            "kappa_z_center": z_center["kappa"],
            "kappa_z_aug": z_aug["kappa"],
            "r2_over_d_center": z_center["mean_r_sq_over_d"],
            "mean_r_aug": z_aug["mean_r"],
            "std_r_aug": z_aug["std_r"],
            "rel_spread_aug": z_aug["std_r"] / z_aug["mean_r"],
            "kappa_h_train": _last(run, KAPPA_H_TRAIN),
            "kappa_h_center": h_center["kappa"],
            "kappa_h_aug": h_aug["kappa"],
        }
        by_lambda.setdefault(run.lam, []).append(record)
        per_run.append({"run": run.name, "lambda": run.lam, "seed": run.seed, **record})

    d = runs[0].embed_dim if runs else 256
    summary: list[dict[str, str | int]] = [
        {
            "lambda": "target (N(0, I))",
            "kappa_z_train": "1",
            "kappa_z_center": "1",
            "kappa_z_aug": "1",
            "r2_over_d_center": "1",
            "mean_r_aug": f"{math.sqrt(d - 0.5):.4g}",
            "std_r_aug": f"{math.sqrt(0.5):.4g}",
            "rel_spread_aug": f"{math.sqrt(0.5) / math.sqrt(d - 0.5):.3g}",
        }
    ]
    for lam, records in sorted(by_lambda.items()):
        row: dict[str, str | int] = {"lambda": f"{lam:g}", "seeds": len(records)}
        for key in records[0]:
            row[key] = _mean_std([r[key] for r in records])
        summary.append(row)

    note = (
        "" if not missing else (f"\n\nMissing diagnostics for: {', '.join(missing)}.")
    )
    reporting.update_section(
        args.note,
        "diag.summary",
        "### Test-time diagnostic of the final checkpoints, per lambda "
        "(mean ± std over seeds)\n\n"
        "`*_train`: pooled over all six views, mean over the last epoch's training "
        "steps. `*_center` / `*_aug`: embeddings of STL-10 test (8,000 images), "
        "un-augmented view and global-view augmentation. `r2_over_d_center` is "
        "E||z||^2 / d (1 for N(0, I)); `rel_spread_aug` is std(||z||) / mean(||z||)."
        + note
        + "\n\n"
        + reporting.markdown_table(summary),
    )
    reporting.update_section(
        args.note,
        "diag.runs",
        "### Test-time diagnostic, per run\n\n" + reporting.markdown_table(per_run),
    )
    print(f"wrote {args.note} ({len(per_run)} runs, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
