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

With `--sigreg-power` (needs `--config`), the synthetic SIGReg power study of
`lejepa_norm.eval.sigreg_power` (sizes from the config) is split into independent
units (one per radial profile and kappa, plus one per real run `PREFIX_*` given by
`--runs-prefix`; projector output z only). `--power-list-units` prints them,
`--power-unit NAME` computes one into a shard under `<runs_dir>/sigreg_power` (on
the GPU if there is one; finished shards are skipped, so jobs can be requeued), and
without either the shards are aggregated into the note with power and gradient
tables and figures (`slurm/sigreg_power.sbatch` fans the units out over SLURM).

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
    python apps/analyze.py --aggregate-controls a_l0.05_s0_controls.md \
        a_l0.05_s1_controls.md --note radius-conditional/controls_summary.md
    python apps/analyze.py --sweep radius_marginal \
        --note radius_marginal/lambda_sweep.md
    python apps/analyze.py --sigreg-power --config configs/base.yaml \
        --runs-prefix radius_marginal --power-unit null
    python apps/analyze.py --sigreg-power --config configs/base.yaml \
        --runs-prefix radius_marginal --note sigreg_power/power.md
"""

import argparse
import collections.abc
import dataclasses
import logging
import math
import pathlib
import sys
import typing

import numpy
import torch

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import analysis
from lejepa_norm.eval import controls
from lejepa_norm.eval import dumps
from lejepa_norm.eval import runlog
from lejepa_norm.eval import sigreg_power
from lejepa_norm.eval import sweep

logger = logging.getLogger(__name__)

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
        "--sigreg-power",
        action="store_true",
        help="Synthetic SIGReg power study (needs --config).",
    )
    parser.add_argument(
        "--power-unit",
        help="With --sigreg-power: compute only this unit into its shard (see "
        "--power-list-units); without it, aggregate the shards into --note.",
    )
    parser.add_argument(
        "--power-list-units",
        action="store_true",
        help="With --sigreg-power: print the unit names, one per line, and exit.",
    )
    parser.add_argument(
        "--power-shards",
        default="default",
        help="With --sigreg-power: shard directory name under <runs_dir>/sigreg_power.",
    )
    parser.add_argument(
        "--runs-prefix",
        help="With --sigreg-power: also analyze the real z of runs PREFIX_*.",
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
    if args.run_name is not None and args.config is None:
        parser.error("--run-name requires --config")
    if args.sigreg_power and args.config is None:
        parser.error("--sigreg-power requires --config")
    if args.note is None and not (args.power_unit or args.power_list_units):
        parser.error("--note is required")
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


def _power_figures(
    study: sigreg_power.SyntheticStudy,
    real: list[dict[str, str | float]],
    note: pathlib.Path,
    chrome: pathlib.Path | None,
) -> list[str]:
    prefix = note.parent / note.stem
    links: list[str] = []

    def column(
        rows: list[dict[str, str | float]], where: dict[str, str | float], key: str
    ) -> list[float]:
        return [
            float(r[key]) for r in rows if all(r.get(k) == v for k, v in where.items())
        ]

    kappas = list(sigreg_power.KAPPAS)
    statistics = [f"SIGReg, {views} view(s)" for views in (1, study.num_views)] + [
        "KS vs chi_d"
    ]
    series = {
        f"{statistic} ({family})": plotting.Series(
            kappas,
            column(
                study.power_rows,
                {"family": family, "statistic": statistic},
                "power",
            ),
        )
        for family in sigreg_power.FAMILIES
        for statistic in statistics
    }
    out = plotting.export_png(
        plotting.line_plot(
            series,
            "Power at level 0.05 against N(0, I) (N per view, synthetic)",
            "kappa = Var(||z||^2) / (2 d)",
            "power",
            reference=sigreg_power.LEVEL,
        ),
        f"{prefix}_power.png",
        chrome,
    )
    links.append(reporting.image_link(out, note, "power against kappa"))

    value_series: dict[str, plotting.Series] = {}
    for family in sigreg_power.FAMILIES:
        where: dict[str, str | float] = {
            "family": family,
            "statistic": "SIGReg, 1 view(s)",
        }
        mean = column(study.power_rows, where, "mean")
        std = column(study.power_rows, where, "std")
        value_series[f"synthetic {family}"] = plotting.Series(
            kappas,
            mean,
            [m - s for m, s in zip(mean, std, strict=True)],
            [m + s for m, s in zip(mean, std, strict=True)],
        )
    if real:
        ordered = sorted(real, key=lambda r: float(r["kappa (center views)"]))
        value_series["real checkpoints (center view)"] = plotting.Series(
            [float(r["kappa (center views)"]) for r in ordered],
            [float(r["SIGReg, center view"]) for r in ordered],
        )
    out = plotting.export_png(
        plotting.line_plot(
            value_series,
            "Single-view SIGReg vs kappa (bars: 1 sd over repetitions)",
            "kappa",
            "SIGReg",
            reference=study.null_mean,
        ),
        f"{prefix}_value.png",
        chrome,
    )
    links.append(reporting.image_link(out, note, "SIGReg value against kappa"))
    return links


def _real_runs(args: argparse.Namespace, locations: paths.Paths) -> list[str]:
    """Runs `PREFIX_*` that have an `stl10_test` dump."""
    if args.runs_prefix is None:
        return []
    return sorted(
        p.name
        for p in locations.embed_dir.glob(f"{args.runs_prefix}_*")
        if dumps.set_dir(locations.embed_dir, p.name, dumps.STL10_TEST).is_dir()
    )


def _run_power_unit(
    unit: str,
    study_cfg: sigreg_power.StudyConfig,
    real_reps: int,
    shard_dir: pathlib.Path,
    locations: paths.Paths,
    device: torch.device,
) -> None:
    """Computes one unit of the power study into its shard (skips finished ones)."""
    if unit.startswith("real_"):
        name = unit.removeprefix("real_")
        if sigreg_power.load_real(shard_dir, unit, study_cfg, real_reps) is not None:
            logger.info("%s: shard exists, skipping", unit)
            return
        shard = sigreg_power.load_real_shard(str(locations.embed_dir), name)
        if shard is None:
            raise FileNotFoundError(f"no stl10_test dump for run {name}")
        metrics_file = locations.runs_dir / name / runlog.METRICS_FILE
        records = runlog.read_metrics(metrics_file) if metrics_file.exists() else []
        row = sigreg_power.real_row(
            name,
            shard,
            study_cfg,
            sigreg_power.last_training_dist(records),
            device,
            real_reps,
        )
        sigreg_power.save_real(shard_dir, unit, row, study_cfg, real_reps)
        return
    if unit not in sigreg_power.cell_units():
        raise ValueError(f"unknown synthetic unit {unit!r}")
    if sigreg_power.load_cell(shard_dir, unit, study_cfg) is not None:
        logger.info("%s: shard exists, skipping", unit)
        return
    sigreg_power.save_cell(
        shard_dir, sigreg_power.run_cell(unit, study_cfg, device), study_cfg
    )


def _write_power_note(
    args: argparse.Namespace,
    study_cfg: sigreg_power.StudyConfig,
    real_reps: int,
    shard_dir: pathlib.Path,
    locations: paths.Paths,
    real_names: list[str],
) -> None:
    """Aggregates the unit shards into the note (cheap, CPU only)."""
    cells: dict[str, sigreg_power.CellResult] = {}
    missing: list[str] = []
    for unit in sigreg_power.cell_units():
        cell = sigreg_power.load_cell(shard_dir, unit, study_cfg)
        if cell is None:
            missing.append(unit)
        else:
            cells[unit] = cell
    raw_real: list[dict[str, str | float]] = []
    for name in real_names:
        unit = sigreg_power.real_unit(name)
        row = sigreg_power.load_real(shard_dir, unit, study_cfg, real_reps)
        if row is None:
            missing.append(unit)
        else:
            raw_real.append(row)
    if missing:
        raise FileNotFoundError(
            f"missing or stale shards in {shard_dir}: {', '.join(missing)}"
        )
    note = locations.notes_dir / args.note
    rng = numpy.random.default_rng(study_cfg.seed + 1)
    study = sigreg_power.assemble(cells, study_cfg)
    real = [sigreg_power.standardize_real_row(row, study) for row in raw_real]
    reporting.update_section(
        note,
        "sigreg_power.setup",
        "### Synthetic setup\n\n"
        + reporting.markdown_table(
            [
                {"parameter": key, "value": str(value)}
                for key, value in dataclasses.asdict(study_cfg).items()
            ]
        ),
    )
    reporting.update_section(
        note,
        "sigreg_power.kurtosis",
        "### Projection kurtosis: claimed vs exact vs Monte Carlo\n\n"
        + reporting.markdown_table(sigreg_power.kurtosis_rows(study_cfg, rng)),
    )
    reporting.update_section(
        note,
        "sigreg_power.power",
        "### SIGReg value and power against N(0, I)\n\n"
        f"Null (exact Gaussians): single view {study.null_mean:.4g} ± "
        f"{study.null_std:.3g}; {study_cfg.num_views} independent views "
        f"{study.null_views[0]:.4g} ± {study.null_views[1]:.3g}. Power is the "
        "fraction of repetitions above the 95% null quantile of the same "
        "statistic.\n\n" + reporting.markdown_table(study.power_rows),
    )
    reporting.update_section(
        note,
        "sigreg_power.gradient",
        "### Gradient of single-view SIGReg: radial vs tangential part\n\n"
        + reporting.markdown_table(study.gradient_rows),
    )
    if args.runs_prefix is not None:
        reporting.update_section(
            note,
            "sigreg_power.real",
            "### Real z embeddings (STL-10 test, projector output)\n\n"
            + reporting.markdown_table(real),
        )
    if not args.no_figures:
        links = _power_figures(study, real, note, locations.kaleido_chrome)
        reporting.update_section(
            note, "sigreg_power.figures", "### Figures\n\n" + "\n\n".join(links)
        )
    print(f"wrote {note}")


def _run_sigreg_power(args: argparse.Namespace) -> int:
    """Power study: list units, run one unit (GPU if available) or aggregate."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr
    )
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    study_cfg = sigreg_power.study_config(cfg)
    real_reps = cfg.eval.power_real_reps
    shard_dir = locations.runs_dir / "sigreg_power" / args.power_shards
    real_names = _real_runs(args, locations)
    units = list(sigreg_power.cell_units()) + [
        sigreg_power.real_unit(name) for name in real_names
    ]
    if args.power_list_units:
        print("\n".join(units), flush=True)
        return 0
    if args.power_unit is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("unit %s on %s", args.power_unit, device)
        _run_power_unit(
            args.power_unit, study_cfg, real_reps, shard_dir, locations, device
        )
        return 0
    _write_power_note(args, study_cfg, real_reps, shard_dir, locations, real_names)
    return 0


def main(argv: collections.abc.Sequence[str]) -> int:
    """Runs the analysis app; returns a process exit code."""
    args = _parse(argv)
    if args.sweep is not None:
        return _run_sweep(args)
    if args.aggregate_controls is not None:
        return _run_aggregate_controls(args)
    if args.sigreg_power:
        return _run_sigreg_power(args)
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
