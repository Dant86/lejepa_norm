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

With `--transfer` (per-run mode), only the clean-to-corrupted transfer of the
radius map (z only; see `lejepa_norm.eval.transfer`) is run on the `stl10_test` and
`stl10c` dumps and written to its own section, with two PNGs (raw shift and
residual per severity) unless `--no-figures`. `--aggregate-transfer NOTE...`
aggregates such sections over seeds per lambda. h is not analyzed in these modes.

With `--redundancy-controls` (per-run mode), only the controls of the redundancy
result are run (permutation null, cross-view regressions, F-test; see
`lejepa_norm.eval.controls`) on the `stl10_test` dump and written to their own
section; no other analysis runs and no figures are made.

With `--randomize-radius PREFIX`, the radius of z-space embeddings is randomized
before k-NN, linear-probe and MLP classification of the STL-10 labels (bank:
`stl10_train` dump, test set: `stl10_test` dump; z only; see
`lejepa_norm.eval.randomization`) for every run `PREFIX_*` that has both dumps;
runs missing a dump are listed in the note, not dropped silently. The note gets
paired-bootstrap differences against the unmodified baseline with an equivalence
verdict (margin `eval.equivalence_margin`), accuracy tables and figures.

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
    python apps/analyze.py --config configs/base.yaml --run-name a_l0.05_s0 \
        --note radius_transfer/a_l0.05_s0.md --transfer
    python apps/analyze.py --aggregate-transfer radius_transfer/a_l0.05_s0.md \
        radius_transfer/a_l0.05_s1.md --note radius_transfer/summary.md
    python apps/analyze.py --aggregate-controls a_l0.05_s0_controls.md \
        a_l0.05_s1_controls.md --note radius-conditional/controls_summary.md
    python apps/analyze.py --sweep radius_marginal \
        --note radius_marginal/lambda_sweep.md
    python apps/analyze.py --config configs/base.yaml \
        --randomize-radius radius_marginal \
        --note radius_randomization/radius_randomization.md
"""

import argparse
import collections.abc
import math
import pathlib
import sys
import typing

import numpy

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import analysis
from lejepa_norm.eval import controls
from lejepa_norm.eval import dumps
from lejepa_norm.eval import randomization
from lejepa_norm.eval import runlog
from lejepa_norm.eval import sweep
from lejepa_norm.eval import transfer

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
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="Per-run and --randomize-radius modes.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-name", help="Analyze one run (needs --config).")
    mode.add_argument(
        "--sweep", metavar="PREFIX", help="Summarize all runs named PREFIX_*."
    )
    mode.add_argument(
        "--randomize-radius",
        metavar="PREFIX",
        help="k-NN with randomized radius for runs PREFIX_* (needs --config).",
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
        "--aggregate-transfer",
        nargs="+",
        type=pathlib.Path,
        metavar="TRANSFER_NOTE",
        help="Aggregate per-run transfer notes (relative to the notes dir) over "
        "seeds, per lambda, into --note.",
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
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="Per-run mode only: run just the clean-to-corrupted radius transfer.",
    )
    args = parser.parse_args(argv)
    if args.transfer and args.run_name is None:
        parser.error("--transfer requires --run-name")
    if args.run_name is not None and args.config is None:
        parser.error("--run-name requires --config")
    if args.randomize_radius is not None and args.config is None:
        parser.error("--randomize-radius requires --config")
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


def _transfer_figures(
    result: transfer.TransferResult, note: pathlib.Path, chrome: pathlib.Path | None
) -> list[str]:
    prefix = note.parent / note.stem
    links: list[str] = []
    panels = (
        ("raw_shift", "Raw paired radius shift (z)", "mean delta rho"),
        (
            "residual_held",
            "Residual after the clean-fitted map (z, held-out twins)",
            "mean (observed - predicted) rho",
        ),
    )
    for attribute, title, y_title in panels:
        series: dict[str, plotting.Series] = {}
        for corruption in dict.fromkeys(c.corruption for c in result.cells):
            cells = [c for c in result.cells if c.corruption == corruption]
            est = [getattr(c, attribute) for c in cells]
            series[corruption] = plotting.Series(
                x=[c.severity for c in cells],
                y=[e.value for e in est],
                low=[e.low for e in est],
                high=[e.high for e in est],
            )
        out = plotting.export_png(
            plotting.line_plot(series, title, "severity", y_title, reference=0.0),
            f"{prefix}_transfer_{attribute}.png",
            chrome,
        )
        links.append(reporting.image_link(out, note, title))
    return links


def _run_transfer(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    test = dumps.load_set(
        dumps.set_dir(locations.embed_dir, args.run_name, dumps.STL10_TEST)
    )
    corrupt = dumps.load_set(
        dumps.set_dir(locations.embed_dir, args.run_name, dumps.STL10_C)
    )
    if test is None or corrupt is None:
        print(
            f"need {dumps.STL10_TEST} and {dumps.STL10_C} dumps for {args.run_name}",
            file=sys.stderr,
        )
        return 1
    result = transfer.run_transfer(test, corrupt, cfg.eval, cfg.seed)
    note = locations.notes_dir / args.note
    reporting.update_section(
        note,
        "analyze.radius_transfer",
        f"### Radius transfer (run `{args.run_name}`)\n\n"
        "h is not analyzed; LeJEPA's product is z. Kernel ridge `u_hat -> rho_bar` "
        f"fitted on {result.n_fit} clean `stl10_test` images (hyperparameters by "
        f"leave-one-out CV on them: tau={result.tau:.3g}, alpha={result.alpha:.3g}), "
        "applied to the corrupted versions of the other "
        f"{result.n_held} images (`held-out twins`) and, for comparison, of the fit "
        "images (`fit twins`). Shifts and residuals are in log-radius units (mean "
        "over images, 95% bootstrap CI over images with the map held fixed); "
        "`predicted shift` is the mean of `f(u_corrupted) - f(u_clean twin)`; "
        "`R^2` is the transferred `R^2` within the cell; `nn_cos` is the "
        "mean cosine to the nearest fit direction (support check). The last row is "
        "the same map on the clean held-out images.\n\n"
        + reporting.markdown_table(transfer.table_rows(result)),
    )
    if not args.no_figures:
        links = _transfer_figures(result, note, locations.kaleido_chrome)
        reporting.update_section(
            note,
            "analyze.radius_transfer_figures",
            "### Figures\n\n" + "\n\n".join(links),
        )
    print(f"wrote {note}")
    return 0


def _randomization_figures(
    comparisons: list[randomization.Comparison],
    note: pathlib.Path,
    chrome: pathlib.Path | None,
) -> list[str]:
    """One figure per reader: paired accuracy change vs lambda, with 95% CIs."""
    prefix = note.parent / note.stem
    links: list[str] = []
    for reader in randomization.READERS:
        series: dict[str, plotting.Series] = {}
        for protocol in randomization.PROTOCOLS:
            for condition in randomization.CONDITIONS[1:]:
                rows = [
                    c
                    for c in comparisons
                    if c.reader == reader
                    and c.protocol == protocol
                    and c.condition == condition
                ]
                series[f"{condition} ({protocol})"] = plotting.Series(
                    x=[c.lam for c in rows],
                    y=[100 * c.difference for c in rows],
                    low=[100 * c.ci95[0] for c in rows],
                    high=[100 * c.ci95[1] for c in rows],
                )
        fig = plotting.line_plot(
            series,
            f"Accuracy change vs baseline, {reader} (z, 95% paired bootstrap CI)",
            "lambda",
            "accuracy change (points)",
            reference=0.0,
        )
        slug = "".join(ch if ch.isalnum() else "_" for ch in reader.split(" (")[0])
        out = plotting.export_png(fig, f"{prefix}_change_{slug}.png", chrome)
        links.append(reporting.image_link(out, note, f"accuracy change, {reader}"))
    return links


def _run_randomize_radius(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    runs = sweep.load_runs(locations.runs_dir, args.randomize_radius)
    results: list[randomization.RunResult] = []
    diagnostics: list[dict[str, str | int | float]] = []
    skipped: list[dict[str, str | int | float]] = []
    for run in runs:
        bank = dumps.load_set(
            dumps.set_dir(locations.embed_dir, run.name, dumps.STL10_TRAIN)
        )
        query = dumps.load_set(
            dumps.set_dir(locations.embed_dir, run.name, dumps.STL10_TEST)
        )
        if bank is None or query is None:
            missing = [
                name
                for name, shard in (
                    (dumps.STL10_TRAIN, bank),
                    (dumps.STL10_TEST, query),
                )
                if shard is None
            ]
            skipped.append({"run": run.name, "missing dump": ", ".join(missing)})
            print(f"skipping {run.name}: no {', '.join(missing)} dump", file=sys.stderr)
            continue
        evaluation = randomization.evaluate_shards(bank, query, cfg.eval, cfg.seed)
        results.append(randomization.RunResult(run.name, run.lam, run.seed, evaluation))
        diagnostics += [
            {
                "run": run.name,
                "MLP weight decay": evaluation.weight_decay,
                **row,
            }
            for row in evaluation.model
        ]
        print(f"done {run.name}", file=sys.stderr)
    if not results:
        print(f"no run named {args.randomize_radius}_* has both dumps", file=sys.stderr)
        return 1
    margin = cfg.eval.equivalence_margin
    comparisons = randomization.compare(
        results,
        margin,
        cfg.eval.num_bootstrap,
        numpy.random.default_rng(cfg.seed),
    )
    note = locations.notes_dir / args.note
    reporting.update_section(
        note,
        "analyze.randomization_comparison",
        "### Paired accuracy change vs the baseline, per lambda\n\n"
        f"z-space, center views; bank = STL-10 train, test set = STL-10 test; the "
        "same transformation is applied to bank and test set independently. "
        "Differences are in accuracy points (condition minus baseline on the same "
        "test images), averaged over training seeds; intervals are paired "
        f"bootstrap percentile intervals ({cfg.eval.num_bootstrap} replicates, test "
        "images resampled within each seed). Verdict, with the equivalence margin "
        f"set to ±{100 * margin:g} points before looking at results: `equivalent` "
        "if the 90% interval (TOST at 5%) lies inside the margin, `different` if "
        "not equivalent and the 95% interval excludes 0, otherwise `inconclusive`. "
        f"Shuffle repeats ({cfg.eval.num_randomizations} per randomized condition) "
        "only average out shuffle noise; they are not part of the interval.\n\n"
        + reporting.markdown_table(randomization.comparison_rows(comparisons)),
    )
    reporting.update_section(
        note,
        "analyze.randomization_accuracy",
        "### Accuracy: mean ± std over training seeds, per lambda\n\n"
        + reporting.markdown_table(randomization.accuracy_rows(results)),
    )
    reporting.update_section(
        note,
        "analyze.randomization_runs",
        "### Per-run accuracies\n\n"
        + reporting.markdown_table(randomization.per_run_rows(results)),
    )
    reporting.update_section(
        note,
        "analyze.randomization_model",
        "### Direction -> log-radius regression used by the residual shuffle\n\n"
        + reporting.markdown_table(diagnostics),
    )
    reporting.update_section(
        note,
        "analyze.randomization_skipped",
        "### Runs without the required dumps\n\n"
        + (
            reporting.markdown_table(skipped)
            if skipped
            else "None: every run of the prefix has `stl10_train` and `stl10_test`."
        ),
    )
    if not args.no_figures:
        links = _randomization_figures(comparisons, note, locations.kaleido_chrome)
        reporting.update_section(
            note,
            "analyze.randomization_figures",
            "### Figures\n\n" + "\n\n".join(links),
        )
    print(f"wrote {note}")
    return 0


def _run_aggregate_transfer(args: argparse.Namespace) -> int:
    locations = paths.load_paths()
    by_lambda: dict[float, list[list[dict[str, str]]]] = {}
    names: list[str] = []
    for relative in args.aggregate_transfer:
        section = reporting.read_section(
            locations.notes_dir / relative, "analyze.radius_transfer"
        )
        name = controls.run_name(section)
        cfg = config.load_config(locations.runs_dir / name / "config.resolved.yaml")
        by_lambda.setdefault(cfg.loss.lambda_, []).append(
            reporting.parse_table(section)
        )
        names.append(name)
    rows = transfer.aggregate_transfer(by_lambda)
    note = locations.notes_dir / args.note
    reporting.update_section(
        note,
        "analyze.radius_transfer_aggregate",
        "### Radius transfer: mean ± std over seeds, per lambda\n\n"
        f"Runs: {', '.join(f'`{n}`' for n in sorted(names))}. h is not analyzed; "
        "LeJEPA's product is z. Aggregates the per-run tables (held-out twins); "
        "`unexplained fraction` is the seed-mean residual over the seed-mean raw "
        "shift.\n\n" + reporting.markdown_table(rows),
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


def main(argv: collections.abc.Sequence[str]) -> int:
    """Runs the analysis app; returns a process exit code."""
    args = _parse(argv)
    if args.sweep is not None:
        return _run_sweep(args)
    if args.randomize_radius is not None:
        return _run_randomize_radius(args)
    if args.aggregate_controls is not None:
        return _run_aggregate_controls(args)
    if args.aggregate_transfer is not None:
        return _run_aggregate_transfer(args)
    if args.transfer:
        return _run_transfer(args)
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
