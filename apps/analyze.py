"""Computes all analyses for one run and writes them into an experiment note.

Reads the embedding dumps of `--run-name` (under `LEJEPA_NORM_EMBED_DIR`, layout
in `lejepa_norm.eval.dumps`) and its training log (`metrics.jsonl` under
`LEJEPA_NORM_RUNS_DIR/<run-name>/`), then writes markdown tables into generated
sections of the note and PNG figures next to it. Re-running updates the
generated sections in place.

Example:
    python apps/analyze.py --config configs/base.yaml --run-name a_l0.05_s0 \
        --note radius-conditional/variant_a.md --set eval.num_bootstrap=200
"""

import argparse
import collections.abc
import pathlib
import sys
import typing

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import analysis
from lejepa_norm.eval import dumps
from lejepa_norm.eval import runlog

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
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--run-name", required=True)
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
    return parser.parse_args(argv)


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


def main(argv: collections.abc.Sequence[str]) -> int:
    """Runs the analysis app; returns a process exit code."""
    args = _parse(argv)
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
