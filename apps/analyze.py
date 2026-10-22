"""Computes analyses and writes them into an experiment note.

Two modes. **Per-run** (`--run-name`): all analyses of one run, below. **Sweep**
(`--sweep PREFIX`): training-log summaries of every run `<runs_dir>/PREFIX_*`
grouped by lambda (resolved configs, not names): a per-lambda table (mean ± std
over seeds at the last epoch all seeds reached, with the N(0, I) targets), a
per-run progress table and, with Chrome, curves over epochs (mean with min-max
over seeds) of kappa_z, ||z|| mean and standard deviation, and k-NN accuracy; when
the logs hold per-view-type statistics (global vs local views), also their
columns and curves of kappa_z and ||z|| mean per view type. It needs no embedding
dumps and can be run while the sweep is still training.

With `--redundancy-controls` (per-run mode), only the controls of the redundancy
result are run (permutation null, cross-view regressions, F-test; see
`lejepa_norm.eval.controls`) on the `stl10_test` dump and written to their own
section; no other analysis runs and no figures are made.

Reads the embedding dumps of `--run-name` (under `LEJEPA_NORM_EMBED_DIR`, layout
in `lejepa_norm.eval.dumps`) and its training log (`metrics.jsonl` under
`LEJEPA_NORM_RUNS_DIR/<run-name>/`), then writes markdown tables into generated
sections of the note and PNG figures next to it. Re-running updates the
generated sections in place.

Examples:
    python apps/analyze.py --config configs/base.yaml --run-name a_l0.05_s0 \
        --note radius-conditional/variant_a.md --set eval.num_bootstrap=200
    python apps/analyze.py --config configs/base.yaml --run-name a_l0.05_s0 \
        --note radius-conditional/variant_a_controls.md --redundancy-controls
    python apps/analyze.py --sweep radius_marginal \
        --note radius_marginal/lambda_sweep.md
"""

import argparse
import collections.abc
import math
import pathlib
import sys
import typing

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import analysis
from lejepa_norm.eval import controls
from lejepa_norm.eval import dumps
from lejepa_norm.eval import runlog
from lejepa_norm.eval import sweep

_TITLES = {
    analysis.MARGINAL: "Marginal radius (STL-10 test)",
    analysis.ICC: "ICC(1) of the log-radius over augmented views",
    analysis.REDUNDANCY: "Redundancy: held-out R^2 of rho_bar",
    analysis.DIFFICULTY: "Difficulty correlations (Spearman, 95% bootstrap CI)",
    analysis.RADIUS_SHIFT: "Paired radius shift vs. corruption severity",
    analysis.DOWNSTREAM: "Downstream accuracy",
    analysis.SELECTIVE: "Selective classification (AURC, lower is better)",
    analysis.OOD: "OOD detection with the radius as score",
    analysis.RETRIEVAL: "DTD retrieval",
}


def _parse(argv: collections.abc.Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Writes all analyses of one run into an experiment note."
    )
    parser.add_argument("--config", type=pathlib.Path, help="Per-run mode only.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-name", help="Analyze one run (needs --config).")
    mode.add_argument(
        "--sweep", metavar="PREFIX", help="Summarize all runs named PREFIX_*."
    )
    parser.add_argument(
        "--note",
        required=True,
        type=pathlib.Path,
        help="Experiment note, relative to LEJEPA_NORM_NOTES_DIR.",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip PNG export (no Chrome)."
    )
    parser.add_argument(
        "--redundancy-controls",
        action="store_true",
        help="Per-run mode only: run just the controls of the redundancy result.",
    )
    args = parser.parse_args(argv)
    if args.run_name is not None and args.config is None:
        parser.error("--run-name requires --config")
    if args.redundancy_controls and args.run_name is None:
        parser.error("--redundancy-controls requires --run-name")
    return args


def _figures(
    result: analysis.AnalysisResult,
    records: list[dict[str, typing.Any]],
    note: pathlib.Path,
    chrome: pathlib.Path | None,
) -> list[str]:
    prefix = note.parent / note.stem
    links: list[str] = []

    def save(fig: plotting.go.Figure, name: str, alt: str) -> None:
        out = plotting.export_png(fig, f"{prefix}_{name}.png", chrome)
        links.append(reporting.image_link(out, note, alt))

    for space, data in result.figures.items():
        save(
            plotting.radius_histogram(
                data.radius,
                data.dim,
                f"Radius of {space} (augmented views)",
                fit_scale=space != "z",
            ),
            f"radius_hist_{space}",
            f"radius histogram of {space}",
        )
        if data.shift:
            series = {
                name: plotting.Series(
                    x=[r.severity for r in rows],
                    y=[r.mean.value for r in rows],
                    low=[r.mean.low for r in rows],
                    high=[r.mean.high for r in rows],
                )
                for name in dict.fromkeys(r.corruption for r in data.shift)
                for rows in [[r for r in data.shift if r.corruption == name]]
            }
            save(
                plotting.line_plot(
                    series,
                    f"Paired radius shift ({space})",
                    "severity",
                    "mean delta rho",
                    reference=0.0,
                ),
                f"radius_shift_{space}",
                f"paired radius shift vs severity ({space})",
            )
        if data.knn_margin is not None and data.residual is not None:
            save(
                plotting.scatter(
                    data.knn_margin,
                    data.residual,
                    f"Radius residual vs k-NN margin ({space})",
                    "k-NN margin",
                    "rho_bar residual",
                ),
                f"residual_vs_margin_{space}",
                f"radius residual vs k-NN margin ({space})",
            )
    kappa_series = {
        key: plotting.Series(*runlog.series(records, key)[1:])
        for key in runlog.numeric_keys(records, "kappa")
    }
    if kappa_series:
        x_name = runlog.series(records, next(iter(kappa_series)))[0]
        save(
            plotting.line_plot(
                kappa_series,
                "kappa during training",
                x_name,
                "kappa",
                reference=1.0,
            ),
            "kappa_training",
            "kappa during training",
        )
    return links


def _sweep_figures(
    runs: list[sweep.Run], note: pathlib.Path, chrome: pathlib.Path | None
) -> list[str]:
    prefix = note.parent / note.stem
    d = runs[0].embed_dim
    groups = sweep.by_lambda(runs)
    plots = (
        (sweep.KAPPA_Z, "kappa_z", "kappa_z during training", 1.0),
        (sweep.Z_NORM_MEAN, "z_norm_mean", "mean ||z|| during training", None),
        (sweep.Z_NORM_STD, "z_norm_std", "std of ||z|| during training", None),
        ("knn_acc_h", "knn_acc_h", "online k-NN accuracy (backbone h)", None),
    )
    present = {key for key, _ in sweep.summary_keys(runs)}
    plots += tuple(
        (key, name, f"{name} during training", reference)
        for key, name, reference in (
            (sweep.KAPPA_Z_GLOBAL, "kappa_z_global", 1.0),
            (sweep.KAPPA_Z_LOCAL, "kappa_z_local", 1.0),
            (sweep.Z_NORM_MEAN_GLOBAL, "z_norm_mean_global", None),
            (sweep.Z_NORM_MEAN_LOCAL, "z_norm_mean_local", None),
        )
        if key in present
    )
    references = {
        "z_norm_mean": math.sqrt(d - 0.5),
        "z_norm_std": math.sqrt(0.5),
        "z_norm_mean_global": math.sqrt(d - 0.5),
        "z_norm_mean_local": math.sqrt(d - 0.5),
    }
    links: list[str] = []
    for key, name, title, reference in plots:
        curves = {
            f"lambda={lam:g}": sweep.curve(group, key) for lam, group in groups.items()
        }
        series = {
            label: plotting.Series(c.epochs, c.mean, c.low, c.high)
            for label, c in curves.items()
            if c.epochs
        }
        if not series:
            continue
        fig = plotting.line_plot(
            series,
            f"{title} (mean over seeds, bars: min-max)",
            "epoch",
            name,
            reference=references.get(name, reference),
        )
        out = plotting.export_png(fig, f"{prefix}_curve_{name}.png", chrome)
        links.append(reporting.image_link(out, note, title))
    return links


def _run_sweep(args: argparse.Namespace) -> int:
    locations = paths.load_paths()
    runs = sweep.load_runs(locations.runs_dir, args.sweep)
    if not runs:
        print(f"no runs named {args.sweep}_* with a resolved config", file=sys.stderr)
        return 1
    note = locations.notes_dir / args.note
    reporting.update_section(
        note,
        "sweep.summary",
        f"### Sweep `{args.sweep}`: per lambda (mean ± std over seeds)\n\n"
        "Each metric is taken at the last epoch that every seed of that lambda has "
        "logged it; the `epoch` column is that epoch for `kappa_z`, and sparsely "
        "logged metrics (k-NN) may come from an earlier one.\n\n"
        + reporting.markdown_table(sweep.summary_rows(runs)),
    )
    reporting.update_section(
        note,
        "sweep.runs",
        "### Sweep runs\n\n" + reporting.markdown_table(sweep.run_rows(runs)),
    )
    if not args.no_figures:
        links = _sweep_figures(runs, note, locations.kaleido_chrome)
        reporting.update_section(
            note, "sweep.figures", "### Sweep figures\n\n" + "\n\n".join(links)
        )
    print(f"wrote {note}")
    return 0


def _run_redundancy_controls(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    test = dumps.load_set(
        dumps.set_dir(locations.embed_dir, args.run_name, dumps.STL10_TEST)
    )
    if test is None:
        print(f"no {dumps.STL10_TEST} dump for {args.run_name}", file=sys.stderr)
        return 1
    rows = controls.run_redundancy_controls(test, cfg.eval, cfg.seed)
    note = locations.notes_dir / args.note
    reporting.update_section(
        note,
        "analyze.redundancy_controls",
        f"### Redundancy controls (run `{args.run_name}`)\n\n"
        "Held-out R^2 of the radius from the direction (nested-CV kernel ridge, "
        "same images as the redundancy table) against a permutation null "
        f"({cfg.eval.num_permutations} shuffles of the radius through the identical "
        "pipeline; p = (1 + #{null >= observed}) / (1 + permutations)), with "
        "cross-view setups that share no view between input and target, and the "
        "classical OLS F-test (linear, in-sample). The reference row is the squared "
        "correlation of the two half-view means of rho_bar.\n\n"
        + reporting.markdown_table(rows),
    )
    print(f"wrote {note}")
    return 0


def main(argv: collections.abc.Sequence[str]) -> int:
    """Runs the analysis app; returns a process exit code."""
    args = _parse(argv)
    if args.sweep is not None:
        return _run_sweep(args)
    if args.redundancy_controls:
        return _run_redundancy_controls(args)
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    note = locations.notes_dir / args.note
    sets = dumps.load_run(locations.embed_dir, args.run_name)
    metrics_file = locations.runs_dir / args.run_name / runlog.METRICS_FILE
    records = runlog.read_metrics(metrics_file) if metrics_file.exists() else []

    result = analysis.run_analyses(sets, cfg.eval, cfg.seed)

    inputs = [{"set": name, "images": s.label.size} for name, s in sets.items()]
    reporting.update_section(
        note,
        "analyze.inputs",
        f"### Inputs (run `{args.run_name}`)\n\n" + reporting.markdown_table(inputs),
    )
    for name in analysis.TABLES:
        reporting.update_section(
            note,
            f"analyze.{name}",
            f"### {_TITLES[name]}\n\n" + reporting.markdown_table(result.tables[name]),
        )
    if not args.no_figures:
        links = _figures(result, records, note, locations.kaleido_chrome)
        reporting.update_section(
            note, "analyze.figures", "### Figures\n\n" + "\n\n".join(links)
        )
    print(f"wrote {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
