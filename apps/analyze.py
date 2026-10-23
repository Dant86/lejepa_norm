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

With `--aggregate-controls NOTE...` the generated controls sections of several
per-run notes are read back and aggregated over seeds per lambda (mean ± std);
nothing is recomputed.

With `--redundancy-controls` (per-run mode), only the controls of the redundancy
result are run (permutation null, cross-view regressions, F-test; see
`lejepa_norm.eval.controls`) on the `stl10_test` dump and written to their own
section; no other analysis runs and no figures are made.

The epoch sweep of the redundancy of `z` (see `lejepa_norm.eval.dynamics`) works
on the per-epoch `stl10_test` dumps written by `apps/embed.py` under
`radius_dynamics/RUN/epoch_XXXX` and has two steps. `--epoch-unit RUN EPOCH`
computes the R^2, kappa_z, ICC and reliability of one (run, epoch) and writes one
JSON shard (resumable: a finished shard is reused; this is what each job of
`slurm/analyze_epochs.sbatch` runs; no note is written). `--epoch-sweep RUN...`
is the cheap aggregation: it reads the shards (computing none) and writes per-run
tables, a table by epoch (mean ± std over runs) and a figure into `--note`. The
permutation null uses `eval.num_permutations` repetitions.

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
    python apps/analyze.py --aggregate-controls a_l0.05_s0_controls.md \
        a_l0.05_s1_controls.md --note radius-conditional/controls_summary.md
    python apps/analyze.py --sweep radius_marginal \
        --note radius_marginal/lambda_sweep.md
    python apps/analyze.py --config configs/base.yaml --set eval.num_permutations=20 \
        --epoch-unit a_l0.05_s0 10
    python apps/analyze.py --config configs/base.yaml --set eval.num_permutations=20 \
        --epoch-sweep a_l0.05_s0 a_l0.05_s1 --note radius_dynamics/sweep.md
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
from lejepa_norm.eval import dynamics
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
    mode.add_argument(
        "--aggregate-controls",
        nargs="+",
        type=pathlib.Path,
        metavar="CONTROLS_NOTE",
        help="Aggregate per-run redundancy-controls notes (relative to the notes "
        "dir) over seeds, per lambda, into --note.",
    )
    mode.add_argument(
        "--epoch-sweep",
        nargs="+",
        metavar="RUN",
        help="Epoch sweep of the redundancy of z over the per-epoch dumps of "
        "these runs (needs --config), into --note.",
    )
    mode.add_argument(
        "--epoch-unit",
        nargs=2,
        metavar=("RUN", "EPOCH"),
        help="Compute and store the shard of one (run, epoch) of the epoch sweep "
        "(needs --config; writes no note).",
    )
    parser.add_argument(
        "--note",
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
    needs_config = (args.run_name, args.epoch_sweep, args.epoch_unit)
    if any(m is not None for m in needs_config) and args.config is None:
        parser.error("--run-name, --epoch-sweep and --epoch-unit require --config")
    if args.epoch_unit is None and args.note is None:
        parser.error("--note is required (except with --epoch-unit)")
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


def _run_aggregate_controls(args: argparse.Namespace) -> int:
    locations = paths.load_paths()
    by_lambda: dict[float, list[list[dict[str, str]]]] = {}
    names: list[str] = []
    for relative in args.aggregate_controls:
        section = reporting.read_section(
            locations.notes_dir / relative, "analyze.redundancy_controls"
        )
        name = controls.run_name(section)
        cfg = config.load_config(locations.runs_dir / name / "config.resolved.yaml")
        by_lambda.setdefault(cfg.loss.lambda_, []).append(
            reporting.parse_table(section)
        )
        names.append(name)
    rows = controls.aggregate_controls(by_lambda)
    note = locations.notes_dir / args.note
    reporting.update_section(
        note,
        "analyze.redundancy_controls_aggregate",
        "### Redundancy controls: mean ± std over seeds, per lambda\n\n"
        f"Runs: {', '.join(f'`{n}`' for n in sorted(names))}. Per-run tables are "
        "in the individual controls notes; this table only aggregates them. "
        "`R^2` is the held-out nested-CV kernel-ridge value; `null max`, "
        "`worst permutation p` and `min F` are the least favorable value over "
        "seeds (the permutation p-value cannot go below 1 / (1 + permutations)).\n\n"
        + reporting.markdown_table(rows),
    )
    print(f"wrote {note}")
    return 0


def _run_epoch_unit(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    run, epoch = args.epoch_unit
    dynamics.run_unit(
        locations.embed_dir,
        run,
        int(epoch),
        cfg.eval,
        cfg.seed,
        cfg.eval.num_permutations,
    )
    return 0


def _run_epoch_sweep(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    note = locations.notes_dir / args.note
    runs = dynamics.collect(
        locations.embed_dir,
        args.epoch_sweep,
        cfg.eval,
        cfg.seed,
        cfg.eval.num_permutations,
    )
    for run, rows in runs.items():
        reporting.update_section(
            note,
            f"dynamics.run.{run}",
            f"### Epoch sweep, run `{run}`\n\n" + reporting.markdown_table(rows),
        )
    reporting.update_section(
        note,
        "dynamics.table",
        "### Redundancy and kappa_z by epoch: mean ± std over seeds\n\n"
        f"Runs: {', '.join(f'`{r}`' for r in runs)}. `z` only, `stl10_test`, "
        f"{cfg.eval.max_kernel_images} images for the regressions (the same images "
        f"at every epoch), K = {cfg.eval.num_views} augmented views; `r2_same`: "
        "rho_bar ~ u_hat from all views; `r2_cross`: disjoint view halves (mean of "
        "both directions); `reliability`: split-half r^2 of rho_bar; `icc`: "
        "ICC(1) of the log-radius over views; `kappa_z` on the augmented views "
        f"of all test images (`kappa_z_center`: center view). Permutation null: "
        f"{cfg.eval.num_permutations} shuffles per regression.\n\n"
        + reporting.markdown_table(dynamics.aggregate(runs)),
    )
    if not args.no_figures:
        fig = _dynamics_figure(runs)
        out = plotting.export_png(
            fig, note.parent / f"{note.stem}_r2_kappa.png", locations.kaleido_chrome
        )
        reporting.update_section(
            note,
            "dynamics.figure",
            "### Figure\n\n"
            + reporting.image_link(out, note, "R^2 and kappa_z vs epoch"),
        )
    print(f"wrote {note}")
    return 0


def _dynamics_figure(runs: dict[str, list[dynamics.Row]]) -> plotting.go.Figure:
    def series(key: str) -> plotting.Series:
        c = dynamics.curve(runs, key)
        return plotting.Series(
            c.epochs, list(c.mean), list(c.mean - c.std), list(c.mean + c.std)
        )

    return plotting.dual_axis_line_plot(
        {
            "R^2 same views": series("r2_same"),
            "R^2 disjoint halves": series("r2_cross"),
            "split-half reliability": series("reliability"),
        },
        {"kappa_z (right axis)": series("kappa_z")},
        "Redundancy of the radius and kappa_z over training (mean ± std over seeds)",
        "epoch",
        "R^2 / reliability",
        "kappa_z",
        right_reference=1.0,
    )


def main(argv: collections.abc.Sequence[str]) -> int:
    """Runs the analysis app; returns a process exit code."""
    args = _parse(argv)
    if args.sweep is not None:
        return _run_sweep(args)
    if args.aggregate_controls is not None:
        return _run_aggregate_controls(args)
    if args.epoch_unit is not None:
        return _run_epoch_unit(args)
    if args.epoch_sweep is not None:
        return _run_epoch_sweep(args)
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
