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

With `--lowlabel PREFIX` (needs `--config`), the E1 readers are repeated with a
small labeled bank (`lejepa_norm.eval.lowlabel`): `--lowlabel-run NAME` computes one
run into a shard under `<runs_dir>/lowlabel/<--lowlabel-shards>` (one SLURM job per
run, see `slurm/lowlabel.sbatch`); without it the shards of all runs `PREFIX_*` (or
only those of `--lowlabel-lambda`) are aggregated into `--note`.

The epoch sweep of the redundancy of `z` (see `lejepa_norm.eval.dynamics`) works
on the per-epoch `stl10_test` dumps written by `apps/embed.py` under
`radius_dynamics/RUN/epoch_XXXX` and has two steps. `--epoch-unit RUN EPOCH`
computes the R^2, kappa_z, ICC and reliability of one (run, epoch) and writes one
JSON shard (resumable: a finished shard is reused; this is what each job of
`slurm/analyze_epochs.sbatch` runs; no note is written). `--epoch-sweep RUN...`
is the cheap aggregation: it reads the shards (computing none) and writes per-run
tables, a table by epoch (mean ± std over runs) and a figure into `--note`. The
permutation null uses `eval.num_permutations` repetitions.

With `--randomize-radius PREFIX`, the radius of z-space embeddings is randomized
before k-NN, linear-probe and MLP classification of the STL-10 labels (bank:
`stl10_train` dump, test set: `stl10_test` dump; z only; see
`lejepa_norm.eval.randomization`) for every run `PREFIX_*` that has both dumps;
runs missing a dump are listed in the note, not dropped silently. The note gets
paired-bootstrap differences against the unmodified baseline with an equivalence
verdict (margin `eval.equivalence_margin`), accuracy tables and figures.

With `--shift-unit` (per-run mode, needs `--config`; no `--note`), one run's
use-under-shift outcomes are computed into `<runs_dir>/<run>/shift_cells.npz`
(corrupted STL-10-C test images, readers fitted on the clean bank, radius
conditions applied to the test embeddings) and `shift_ood.npz` (STL-10 test vs DTD
and SVHN scores); z only, center views, memory-light. With `--shift PREFIX` (needs
`--config`) the units of the runs `PREFIX_*` are aggregated per lambda (paired
bootstrap over images, mean over seeds) into `<note stem>_lam<lambda>.md` next to
`--note` (see `lejepa_norm.eval.usage_shift`, `shift`, `ood_radius`).

With `--view-signal PREFIX` (needs `--config`), the radius view-signal study (radial
vs tangential displacement under corruption, oracle clean radius, severity from the
radius residual, radius-weighted multi-view aggregation; z only) is run: with
`--view-signal-run RUN` one run's unit file is computed (STL-10-C with all views,
needs about 64 GB), otherwise the units of the runs `PREFIX_*` are aggregated per
lambda into `<note stem>_lam<lambda>.md` next to `--note`, which gets the
cross-lambda summary
(see `lejepa_norm.eval.view_signal`, `view_signal_stats`).

With `--confidence-unit` (per-run mode, needs `--config`; no `--note`) one run's
radius-vs-confidence statistics (error detection by the radius signals, nested
comparison against the readers' own confidence, STL-10-C and OOD) are computed into
`<runs_dir>/<run>/confidence_<space>.npz` (`--confidence-space z`, or `h` for the
E2 baselines only); `--confidence PREFIX` aggregates them per group into
`<note stem>_<space>_<group>.md` plus a summary section in `--note` (see
`lejepa_norm.eval.usage_confidence`, `confidence`).

With `--retrieval PREFIX`, Euclidean retrieval of z (class-level on the STL-10 center
views, instance-level view-to-view on the augmented global views; see
`lejepa_norm.eval.retrieval`) is evaluated under the radius conditions of the
randomization study. `--retrieval-run NAME` computes one run's per-query metrics
into a shard under `<runs_dir>/radius_usage_retrieval` (one SLURM job per run, GPU
if available); without it the shards of all runs `PREFIX_*` are aggregated into
the note (paired bootstrap + TOST per lambda, margin `eval.retrieval_margin`).

With `--frozen-linear PREFIX` (needs `--config`), the radius dependence of the
frozen linear probe and MLP found by `--randomize-radius` is dissected (exact
logit decomposition, probe variants, per-image flip diagnostics; see
`lejepa_norm.eval.frozen_linear`). Per run `PREFIX_*` with both STL-10 dumps one
unit is computed into a shard under `<runs_dir>/frozen_linear/<--frozen-shards>`
(`--frozen-list-runs` prints the runs, `--frozen-run NAME` computes one; finished
shards are skipped, so SLURM jobs can be requeued and run in parallel,
`slurm/frozen_linear.sbatch`); without either the shards are aggregated per
lambda into `--note`.

With `--baseline-controls PREFIX...` (needs `--config`; z only), the E2 controls of
the redundancy R^2 (`lejepa_norm.eval.baseline_controls`: redundancy controls with
bootstrap intervals, random smooth functions of the direction, learning curve in
N) run per run `PREFIX_*_s<seed>` that has the `stl10_train` and `stl10_test` dumps:
`--bc-list-runs` prints them, `--bc-run NAME` computes one into a shard under
`<runs_dir>/baseline_controls/<--bc-shards>` (`--bc-parts` selects parts; finished
shards are skipped; `slurm/baseline_controls.sbatch` runs one SLURM job per run);
without either the shards are aggregated per group (run name without `_s<seed>`)
into three notes next to `--note` (`<stem>_redundancy.md`, `_smooth.md`,
`_curve.md`), with `--bc-reference` as the group of the contrast table.

With `--compare-prefix PREFIX --baseline-prefix BASELINE` (needs `--config`), the
runs `PREFIX_*` are compared with the matching runs `BASELINE_*` (same lambda and
seed index; z only): clean and STL-10-C accuracy of the E1 readers, class and
instance retrieval and OOD detection, plus training-health curves (see
`lejepa_norm.eval.sphere_compare`; built for E3, the model trained with
`loss.radius_free`). `--compare-unit NAME` first computes one radius-free run's
unit files under `<runs_dir>/NAME` (needs the five embedding dumps; GPU if
available); without it the units are aggregated into `--note`.

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
    python apps/analyze.py --config configs/base.yaml --set eval.num_permutations=20 \
        --epoch-unit a_l0.05_s0 10
    python apps/analyze.py --config configs/base.yaml --set eval.num_permutations=20 \
        --epoch-sweep a_l0.05_s0 a_l0.05_s1 --note radius_dynamics/sweep.md
    python apps/analyze.py --sigreg-power --config configs/base.yaml \
        --runs-prefix radius_marginal --power-unit null
    python apps/analyze.py --sigreg-power --config configs/base.yaml \
        --runs-prefix radius_marginal --note sigreg_power/power.md
    python apps/analyze.py --config configs/base.yaml \
        --randomize-radius radius_marginal \
        --note radius_randomization/radius_randomization.md
    python apps/analyze.py --config configs/base.yaml \
        --run-name radius_marginal_lam0.05_s0 --shift-unit
    python apps/analyze.py --config configs/base.yaml --shift radius_marginal \
        --note radius_usage/shift.md
    python apps/analyze.py --config configs/base.yaml --svhn-control radius_marginal \
        --svhn-control-stats
    python apps/analyze.py --config configs/base.yaml --svhn-control radius_marginal \
        --svhn-control-run radius_marginal_lam0.05_s0
    python apps/analyze.py --config configs/base.yaml --svhn-control radius_marginal \
        --note radius_usage/svhn_control.md
    # the same with Places365 as the OOD set (add --control-study places to each)
    python apps/analyze.py --config configs/base.yaml --retrieval radius_marginal \
        --retrieval-run radius_marginal_lam0.05_s0
    python apps/analyze.py --config configs/base.yaml --retrieval radius_marginal \
        --note radius_usage/retrieval_summary.md
    python apps/analyze.py --config configs/base.yaml \
        --frozen-linear radius_marginal --frozen-run radius_marginal_lam0.05_s0
    python apps/analyze.py --config configs/base.yaml \
        --frozen-linear radius_marginal --note radius_usage/frozen_linear_summary.md
    python apps/analyze.py --config configs/base.yaml \
        --baseline-controls radius_marginal random_init \
        --bc-run radius_marginal_lam0.05_s0
    python apps/analyze.py --config configs/base.yaml \
        --baseline-controls radius_marginal random_init \
        --note radius_baselines/controls.md
    python apps/analyze.py --config configs/base.yaml \
        --compare-prefix sigreg_sphere --baseline-prefix radius_marginal \
        --compare-unit sigreg_sphere_lam0.05_s0
    python apps/analyze.py --config configs/base.yaml \
        --compare-prefix sigreg_sphere --baseline-prefix radius_marginal \
        --note sigreg_sphere/sphere_vs_radius.md
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
from lejepa_norm.eval import baseline_controls
from lejepa_norm.eval import controls
from lejepa_norm.eval import dumps
from lejepa_norm.eval import dynamics
from lejepa_norm.eval import frozen_linear
from lejepa_norm.eval import lowlabel
from lejepa_norm.eval import randomization
from lejepa_norm.eval import retrieval
from lejepa_norm.eval import runlog
from lejepa_norm.eval import sigreg_power
from lejepa_norm.eval import sphere_compare
from lejepa_norm.eval import svhn_control
from lejepa_norm.eval import sweep
from lejepa_norm.eval import transfer
from lejepa_norm.eval import usage_confidence
from lejepa_norm.eval import usage_shift
from lejepa_norm.eval import usage_view_signal

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
    parser.add_argument(
        "--space",
        choices=("z", "h"),
        default="z",
        help="--randomize-radius only: embedding to randomize (default z; h is an "
        "explicit exception for the E2 baselines).",
    )
    mode.add_argument(
        "--retrieval",
        metavar="PREFIX",
        help="Euclidean retrieval of z under the radius conditions for runs "
        "PREFIX_* (needs --config); with --retrieval-run computes one run's shard, "
        "without it aggregates the shards into --note.",
    )
    mode.add_argument(
        "--view-signal",
        metavar="PREFIX",
        help="Radius view-signal study (`lejepa_norm.eval.view_signal`; needs "
        "--config): with --view-signal-run compute one run's unit "
        "(`<runs_dir>/<run>/view_signal.npz`, STL-10-C with all views, ~64 GB), "
        "otherwise aggregate the units of runs PREFIX_* per lambda into "
        "`<note stem>_lam<lambda>.md` plus a summary in --note.",
    )
    mode.add_argument(
        "--compare-prefix",
        metavar="PREFIX",
        help="Compare runs PREFIX_* with --baseline-prefix runs (E3, needs "
        "--config); --compare-unit computes one run's units, otherwise aggregate "
        "into --note.",
    )
    mode.add_argument(
        "--lowlabel",
        metavar="PREFIX",
        help="E1 readers with a small labeled bank for runs PREFIX_* (needs "
        "--config): with --lowlabel-run compute one run's shard, otherwise "
        "aggregate the shards into --note.",
    )
    mode.add_argument(
        "--frozen-linear",
        metavar="PREFIX",
        help="Dissect the radius dependence of frozen probes for runs PREFIX_* "
        "(needs --config).",
    )
    mode.add_argument(
        "--baseline-controls",
        nargs="+",
        metavar="PREFIX",
        help="E2 controls of the redundancy R^2 for runs PREFIX_*_s<seed> (needs "
        "--config): with --bc-run compute one run's shard, with --bc-list-runs "
        "list the runs, otherwise aggregate the shards into --note.",
    )
    mode.add_argument(
        "--shift",
        metavar="PREFIX",
        help="Aggregate the --shift-unit outputs of runs PREFIX_* per lambda "
        "(needs --config).",
    )
    mode.add_argument(
        "--confidence",
        metavar="PREFIX",
        help="Aggregate the --confidence-unit outputs of runs PREFIX_* per group "
        "(lambda for LeJEPA, loss for the E2 baselines) into notes next to --note "
        "(see --confidence-space).",
    )
    mode.add_argument(
        "--svhn-control",
        metavar="PREFIX",
        help="Is the SVHN radius signal explained by low-level image statistics? "
        "For runs PREFIX_* (needs --config): with --svhn-control-stats compute the "
        "image statistics once, with --svhn-control-run one run's unit, otherwise "
        "aggregate the units per lambda into --note (see "
        "`lejepa_norm.eval.svhn_control`).",
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
    mode.add_argument(
        "--aggregate-transfer",
        nargs="+",
        type=pathlib.Path,
        metavar="TRANSFER_NOTE",
        help="Aggregate per-run transfer notes (relative to the notes dir) over "
        "seeds, per lambda, into --note.",
    )
    mode.add_argument(
        "--sigreg-power",
        action="store_true",
        help="Synthetic SIGReg power study (needs --config).",
    )
    parser.add_argument(
        "--baseline-prefix",
        help="With --compare-prefix: prefix of the runs compared against.",
    )
    parser.add_argument(
        "--compare-unit",
        metavar="RUN",
        help="With --compare-prefix: compute this radius-free run's unit files "
        "instead of aggregating.",
    )
    parser.add_argument(
        "--retrieval-run",
        help="With --retrieval: compute only this run's per-query shard "
        "(<runs_dir>/radius_usage_retrieval/<run>.npz; GPU if available).",
    )
    parser.add_argument(
        "--frozen-run",
        help="With --frozen-linear: compute only this run into its shard; without "
        "it (and --frozen-list-runs), aggregate the shards into --note.",
    )
    parser.add_argument(
        "--frozen-list-runs",
        action="store_true",
        help="With --frozen-linear: print the runs with both dumps and exit.",
    )
    parser.add_argument(
        "--frozen-shards",
        default="default",
        help="With --frozen-linear: shard directory name under "
        "<runs_dir>/frozen_linear.",
    )
    parser.add_argument(
        "--bc-run", help="With --baseline-controls: compute only this run's shard."
    )
    parser.add_argument(
        "--bc-list-runs",
        action="store_true",
        help="With --baseline-controls: print the runs with both dumps and exit.",
    )
    parser.add_argument(
        "--bc-parts",
        default=",".join(baseline_controls.PARTS),
        help="With --bc-run: comma-separated subset of "
        + ", ".join(baseline_controls.PARTS)
        + ".",
    )
    parser.add_argument(
        "--bc-shards",
        default="default",
        help="With --baseline-controls: shard directory name under "
        "<runs_dir>/baseline_controls.",
    )
    parser.add_argument(
        "--bc-reference",
        default="random_init_bncal",
        help="With --baseline-controls (aggregation): group the contrast table "
        "subtracts.",
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
        "--lowlabel-run",
        help="With --lowlabel: compute only this run into its shard.",
    )
    parser.add_argument(
        "--lowlabel-lambda",
        type=float,
        help="With --lowlabel: aggregate only the runs of this lambda.",
    )
    parser.add_argument(
        "--lowlabel-shards",
        default="default",
        help="With --lowlabel: shard directory name under <runs_dir>/lowlabel.",
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
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="Per-run mode only: run just the clean-to-corrupted radius transfer.",
    )
    parser.add_argument(
        "--control-study",
        choices=sorted(svhn_control.STUDIES),
        default="svhn",
        help="With --svhn-control: the decisive OOD set, `svhn` (default) or "
        "`places` (Places365, see `research_notes/radius_usage/places_control.md`).",
    )
    parser.add_argument(
        "--svhn-control-stats",
        action="store_true",
        help="With --svhn-control: compute the per-image low-level statistics.",
    )
    parser.add_argument(
        "--svhn-control-run",
        metavar="RUN",
        help="With --svhn-control: compute the unit file of this run.",
    )
    parser.add_argument(
        "--shift-unit",
        action="store_true",
        help="Per-run mode only: compute the use-under-shift outcomes of the run.",
    )
    parser.add_argument(
        "--view-signal-run",
        help="With --view-signal: compute only this run's unit file.",
    )
    parser.add_argument(
        "--confidence-unit",
        action="store_true",
        help="Per-run mode only: compute the radius-confidence statistics of the "
        "run into <runs_dir>/<run>/confidence_<space>.npz.",
    )
    parser.add_argument(
        "--confidence-part",
        choices=usage_confidence.PARTS,
        default="all",
        help="With --confidence-unit: `ood` recomputes only the OOD groups of an "
        "existing unit (e.g. when a new OOD dump becomes available).",
    )
    parser.add_argument(
        "--confidence-space",
        choices=usage_confidence.SPACES,
        default="z",
        help="With --confidence / --confidence-unit: z (default) or h. h only for "
        "the E2 baselines (their projector is discarded after training).",
    )
    parser.add_argument(
        "--shift-part",
        choices=usage_shift.PARTS,
        default="both",
        help="With --shift-unit: which part to compute.",
    )
    args = parser.parse_args(argv)
    needs_config = (args.run_name, args.epoch_sweep, args.epoch_unit)
    if any(m is not None for m in needs_config) and args.config is None:
        parser.error("--run-name, --epoch-sweep and --epoch-unit require --config")
    if args.transfer and args.run_name is None:
        parser.error("--transfer requires --run-name")
    if args.shift is not None and args.config is None:
        parser.error("--shift requires --config")
    if args.svhn_control is not None and args.config is None:
        parser.error("--svhn-control requires --config")
    if args.shift_unit and args.run_name is None:
        parser.error("--shift-unit requires --run-name")
    if args.view_signal is not None and args.config is None:
        parser.error("--view-signal requires --config")
    if args.confidence_unit and (args.run_name is None or args.config is None):
        parser.error("--confidence-unit requires --run-name and --config")
    if args.sigreg_power and args.config is None:
        parser.error("--sigreg-power requires --config")
    if args.retrieval is not None and args.config is None:
        parser.error("--retrieval requires --config")
    if args.compare_prefix is not None and (
        args.config is None or args.baseline_prefix is None
    ):
        parser.error("--compare-prefix requires --config and --baseline-prefix")
    if args.lowlabel is not None and args.config is None:
        parser.error("--lowlabel requires --config")
    if args.note is None and not (
        args.epoch_unit
        or args.power_unit
        or args.power_list_units
        or args.shift_unit
        or args.view_signal_run
        or args.confidence_unit
        or args.svhn_control_stats
        or args.svhn_control_run
        or args.retrieval_run
        or args.compare_unit
        or args.lowlabel_run
        or args.frozen_run
        or args.frozen_list_runs
        or args.bc_run
        or args.bc_list_runs
    ):
        parser.error("--note is required")
    if args.randomize_radius is not None and args.config is None:
        parser.error("--randomize-radius requires --config")
    if args.frozen_linear is not None and args.config is None:
        parser.error("--frozen-linear requires --config")
    if args.baseline_controls is not None and args.config is None:
        parser.error("--baseline-controls requires --config")
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
        evaluation = randomization.evaluate_shards(
            bank, query, cfg.eval, cfg.seed, args.space
        )
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
        f"{args.space}-space, center views; bank = STL-10 train, test set = STL-10 "
        "test; the same transformation is applied to bank and test set independently. "
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


def _lowlabel_figures(
    runs: list[lowlabel.RunBudgets],
    comparisons: list[lowlabel.BudgetComparison],
    lam: float,
    note: pathlib.Path,
    chrome: pathlib.Path | None,
) -> list[str]:
    """Per reader: accuracy vs labels (retrained protocol) and paired change."""
    prefix = note.parent / f"{note.stem}"
    links: list[str] = []
    for reader in lowlabel.READERS:
        slug = "".join(ch if ch.isalnum() else "_" for ch in reader.split(" (")[0])
        acc: dict[str, plotting.Series] = {}
        for condition in randomization.CONDITIONS:
            x, y, sd = lowlabel.accuracy_curve(
                runs, lam, reader, randomization.RETRAIN, condition
            )
            acc[condition] = plotting.Series(
                x=x,
                y=[100 * v for v in y],
                low=[100 * (v - s) for v, s in zip(y, sd, strict=True)],
                high=[100 * (v + s) for v, s in zip(y, sd, strict=True)],
            )
        fig = plotting.line_plot(
            acc,
            f"Accuracy vs labels, {reader}, lambda={lam:g}",
            "labeled images (log scale)",
            "test accuracy (%)",
            log_x=True,
        )
        out = plotting.export_png(fig, f"{prefix}_accuracy_{slug}.png", chrome)
        links.append(reporting.image_link(out, note, f"accuracy vs labels, {reader}"))
        change: dict[str, plotting.Series] = {}
        for protocol in randomization.PROTOCOLS:
            for condition in randomization.CONDITIONS[1:]:
                if (
                    protocol == randomization.FROZEN
                    and condition == randomization.CONSTANT
                ):
                    continue  # collapses to chance: a distribution-shift artifact
                rows = [
                    b
                    for b in comparisons
                    if b.comparison.lam == lam
                    and b.comparison.reader == reader
                    and b.comparison.protocol == protocol
                    and b.comparison.condition == condition
                ]
                change[f"{condition} ({protocol})"] = plotting.Series(
                    x=[b.num_labels for b in rows],
                    y=[100 * b.comparison.difference for b in rows],
                    low=[100 * b.comparison.ci95[0] for b in rows],
                    high=[100 * b.comparison.ci95[1] for b in rows],
                )
        fig = plotting.line_plot(
            change,
            f"Accuracy change vs baseline, {reader}, lambda={lam:g}",
            "labeled images (log scale)",
            "accuracy change (points)",
            reference=0.0,
            log_x=True,
        )
        out = plotting.export_png(fig, f"{prefix}_change_{slug}.png", chrome)
        links.append(reporting.image_link(out, note, f"accuracy change, {reader}"))
    return links


def _run_lowlabel(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    shard_dir = locations.runs_dir / "lowlabel" / args.lowlabel_shards
    if args.lowlabel_run is not None:
        name = args.lowlabel_run
        bank = dumps.load_set(
            dumps.set_dir(locations.embed_dir, name, dumps.STL10_TRAIN)
        )
        query = dumps.load_set(
            dumps.set_dir(locations.embed_dir, name, dumps.STL10_TEST)
        )
        if bank is None or query is None:
            raise FileNotFoundError(f"missing stl10_train/stl10_test dump for {name}")
        results = lowlabel.evaluate_run(
            randomization.center_embeddings(bank.z),
            bank.label,
            randomization.center_embeddings(query.z),
            query.label,
            k=cfg.eval.knn_k,
            num_repeats=cfg.eval.num_randomizations,
            cv_folds=cfg.eval.cv_folds,
            seed=cfg.seed,
            num_subsets=cfg.eval.lowlabel_subsets,
            log=lambda message: print(f"{name}: {message}", file=sys.stderr),
        )
        lowlabel.save_run(shard_dir / f"{name}.npz", results)
        print(f"wrote {shard_dir / f'{name}.npz'}")
        return 0
    loaded: list[lowlabel.RunBudgets] = []
    for run in sweep.load_runs(locations.runs_dir, args.lowlabel):
        if args.lowlabel_lambda is not None and run.lam != args.lowlabel_lambda:
            continue
        shard = shard_dir / f"{run.name}.npz"
        if not shard.exists():
            raise FileNotFoundError(f"missing shard {shard}")
        loaded.append(
            lowlabel.RunBudgets(run.name, run.lam, run.seed, lowlabel.load_run(shard))
        )
    if not loaded:
        print(f"no run named {args.lowlabel}_* selected", file=sys.stderr)
        return 1
    margin = cfg.eval.equivalence_margin
    comparisons = lowlabel.compare(
        loaded, margin, cfg.eval.num_bootstrap, numpy.random.default_rng(cfg.seed)
    )
    note = locations.notes_dir / args.note
    reporting.update_section(
        note,
        "analyze.lowlabel_comparison",
        "### Paired accuracy change vs the baseline, per lambda and label budget\n\n"
        "z-space, center views; labeled bank = a class-balanced random subset of "
        "STL-10 train, test set = STL-10 test; the same transformation is applied "
        "to bank and test set independently. Differences are in accuracy points "
        "(condition minus baseline on the same test images and the same subsets), "
        "averaged over training seeds and subsets. `90% CI, test images`: paired "
        f"bootstrap ({cfg.eval.num_bootstrap} replicates, test images resampled "
        "within each seed), conditional on the drawn subsets. `subset SE`: standard "
        "error of the mean paired difference over subsets. Verdicts, with the "
        f"equivalence margin set to ±{100 * margin:g} points before looking at "
        "results: `equivalent` if the 90% interval (TOST at 5%) lies inside the "
        "margin, `different` if not equivalent and the 95% interval excludes 0, "
        "otherwise `inconclusive`; `(with subsets)` adds the subset SE in "
        "quadrature to the bootstrap SE (normal approximation).\n\n"
        + reporting.markdown_table(lowlabel.comparison_rows(comparisons)),
    )
    reporting.update_section(
        note,
        "analyze.lowlabel_summary",
        "### Compact view\n\nAccuracy change in points, `difference [90% CI "
        "including subset noise] verdict(test images)/verdict(with subsets)` with "
        "E = equivalent (margin ±"
        f"{100 * margin:g} points), D = different, I = inconclusive. The frozen "
        "constant-radius cell (chance level, a distribution-shift artifact) is "
        "omitted.\n\n" + reporting.markdown_table(lowlabel.summary_rows(comparisons)),
    )
    reporting.update_section(
        note,
        "analyze.lowlabel_accuracy",
        "### Accuracy per label budget\n\n"
        + reporting.markdown_table(lowlabel.accuracy_rows(loaded)),
    )
    reporting.update_section(
        note,
        "analyze.lowlabel_hyper",
        "### Hyperparameters selected by cross-validation inside each subset\n\n"
        "Linear probe: inverse regularization C; MLP: weight decay.\n\n"
        + reporting.markdown_table(lowlabel.hyper_rows(loaded)),
    )
    if not args.no_figures:
        links: list[str] = []
        for lam in sorted({r.lam for r in loaded}):
            links += _lowlabel_figures(
                loaded, comparisons, lam, note, locations.kaleido_chrome
            )
        reporting.update_section(
            note,
            "analyze.lowlabel_figures",
            "### Figures\n\nPer reader and lambda: accuracy vs number of labeled "
            "images (retrained protocol; mean over seeds and subsets, bars = "
            "standard deviation over subsets), then the paired accuracy change vs "
            "the baseline (bars = 95% paired bootstrap interval over test images; "
            "the frozen constant-radius cell is omitted, it is at chance).\n\n"
            + "\n\n".join(links),
        )
    print(f"wrote {note}")
    return 0


_FROZEN_CURVE_PROBES = (
    frozen_linear.RAW,
    frozen_linear.AUG,
    frozen_linear.MLP_RAW,
    frozen_linear.MLP_AUG,
)


def _frozen_runs_with_dumps(
    args: argparse.Namespace, locations: paths.Paths
) -> list[sweep.Run]:
    """Runs `PREFIX_*` that have both the `stl10_train` and `stl10_test` dumps."""
    return [
        run
        for run in sweep.load_runs(locations.runs_dir, args.frozen_linear)
        if all(
            any(dumps.set_dir(locations.embed_dir, run.name, name).glob("*.npz"))
            for name in (dumps.STL10_TRAIN, dumps.STL10_TEST)
        )
    ]


def _frozen_figures(
    points: list[frozen_linear.CurvePoint],
    note: pathlib.Path,
    chrome: pathlib.Path | None,
) -> list[str]:
    """Accuracy against a common constant radius, one figure per lambda."""
    prefix = note.parent / note.stem
    links: list[str] = []
    for lam in sorted({p.lam for p in points}):
        series = {
            probe: plotting.Series(
                x=[p.exponent for p in points if p.lam == lam and p.probe == probe],
                y=[p.accuracy for p in points if p.lam == lam and p.probe == probe],
            )
            for probe in _FROZEN_CURVE_PROBES
        }
        fig = plotting.line_plot(
            series,
            f"Accuracy with the same radius for every test image, lambda = {lam:g}",
            "log2(radius / median test radius)",
            "test accuracy",
        )
        out = plotting.export_png(fig, f"{prefix}_curve_lam{lam:g}.png", chrome)
        links.append(reporting.image_link(out, note, f"constant-radius curve {lam:g}"))
    return links


def _write_frozen_note(
    args: argparse.Namespace,
    cfg: config.ExperimentConfig,
    results: list[frozen_linear.RunAnalysis],
    locations: paths.Paths,
) -> None:
    """Aggregates the per-run shards into the note (CPU only)."""
    note = locations.notes_dir / args.note
    margin, boots = cfg.eval.equivalence_margin, cfg.eval.num_bootstrap
    rng = numpy.random.default_rng(cfg.seed)
    ci_text = (
        "Differences are in accuracy points on the same test images, averaged over "
        f"training seeds; intervals are paired bootstrap percentile intervals "
        f"({boots} replicates, test images resampled within each seed). Verdict with "
        f"the margin fixed at ±{100 * margin:g} points before looking at results: "
        "`equivalent` if the 90% interval lies inside the margin, `different` if not "
        "equivalent and the 95% interval excludes 0, else `inconclusive`. "
    )
    reporting.update_section(
        note,
        "analyze.frozen_change",
        "### Frozen accuracy change under each radius condition\n\n"
        + ci_text
        + "Each probe is fitted on the clean bank and evaluated frozen; the change "
        "is against its own baseline. Radius-free probes must be exactly 0.\n\n"
        + reporting.markdown_table(
            frozen_linear.change_rows(results, margin, boots, rng)
        ),
    )
    reporting.update_section(
        note,
        "analyze.frozen_cost",
        "### Clean accuracy of the variants against the E1 probe\n\n"
        + ci_text
        + "\n\n"
        + reporting.markdown_table(
            frozen_linear.cost_rows(results, margin, boots, rng)
        ),
    )
    reporting.update_section(
        note,
        "analyze.frozen_flips",
        "### Flips of the frozen E1 probes under the radius permutation\n\n"
        "Per test image and shuffle repeat; points = percent of (image, repeat) "
        "pairs; value [95% CI]. `correct -> wrong` and `wrong -> correct` add up to "
        "the net accuracy change.\n\n"
        + reporting.markdown_table(frozen_linear.flip_rows(results, boots, rng)),
    )
    reporting.update_section(
        note,
        "analyze.frozen_correlations",
        "### What predicts the flips (Spearman, per image, mean over seeds)\n\n"
        "Per-image flip rate (over the shuffle repeats) and accuracy loss against "
        "candidate predictors; value [95% bootstrap CI over test images].\n\n"
        + reporting.markdown_table(frozen_linear.correlation_rows(results, boots, rng)),
    )
    reporting.update_section(
        note,
        "analyze.frozen_grid",
        "### Flip rate (percent) by baseline margin and |log radius ratio|\n\n"
        + reporting.markdown_table(frozen_linear.grid_rows(results)),
    )
    reporting.update_section(
        note,
        "analyze.frozen_decomposition",
        "### Exact logit decomposition of the E1 linear probe (z, test set)\n\n"
        + reporting.markdown_table(frozen_linear.decomposition_rows(results)),
    )
    points = frozen_linear.curve_points(results)
    reporting.update_section(
        note,
        "analyze.frozen_curves",
        "### Accuracy when every test image has the same radius\n\n"
        + reporting.markdown_table(
            frozen_linear.curve_rows(points, _FROZEN_CURVE_PROBES)
        ),
    )
    if not args.no_figures:
        links = _frozen_figures(points, note, locations.kaleido_chrome)
        reporting.update_section(
            note, "analyze.frozen_figures", "### Figures\n\n" + "\n\n".join(links)
        )
    print(f"wrote {note}")


def _run_frozen_linear(args: argparse.Namespace) -> int:
    """Frozen-probe dissection: list runs, compute one run's shard, or aggregate."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr
    )
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    shard_dir = locations.runs_dir / "frozen_linear" / args.frozen_shards
    runs = _frozen_runs_with_dumps(args, locations)
    if args.frozen_list_runs:
        print("\n".join(run.name for run in runs), flush=True)
        return 0
    if args.frozen_run is not None:
        target = [run for run in runs if run.name == args.frozen_run]
        if not target:
            raise FileNotFoundError(f"run {args.frozen_run} lacks a dump or the prefix")
        run = target[0]
        shard = shard_dir / f"{run.name}.npz"
        if shard.exists():
            logger.info("%s: shard exists, skipping", run.name)
            return 0
        bank = dumps.load_set(
            dumps.set_dir(locations.embed_dir, run.name, dumps.STL10_TRAIN)
        )
        query = dumps.load_set(
            dumps.set_dir(locations.embed_dir, run.name, dumps.STL10_TEST)
        )
        assert bank is not None and query is not None
        result = frozen_linear.analyze_shards(
            run.name, run.lam, run.seed, bank, query, cfg.eval, cfg.seed
        )
        frozen_linear.save_run(shard, result)
        logger.info("%s: wrote %s", run.name, shard)
        return 0
    missing = [run.name for run in runs if not (shard_dir / f"{run.name}.npz").exists()]
    if missing or not runs:
        raise FileNotFoundError(f"missing shards in {shard_dir}: {', '.join(missing)}")
    results = [frozen_linear.load_run(shard_dir / f"{run.name}.npz") for run in runs]
    _write_frozen_note(args, cfg, results, locations)
    return 0


def _run_baseline_controls(args: argparse.Namespace) -> int:
    """E2 controls: list runs, compute one run's shard, or aggregate the shards."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr
    )
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    shard_dir = locations.runs_dir / "baseline_controls" / args.bc_shards
    runs = baseline_controls.list_runs(locations.embed_dir, args.baseline_controls)
    if args.bc_list_runs:
        print("\n".join(runs), flush=True)
        return 0
    if args.bc_run is not None:
        if args.bc_run not in runs:
            raise FileNotFoundError(f"run {args.bc_run} lacks a dump or the prefix")
        shard = shard_dir / f"{args.bc_run}.npz"
        if shard.exists():
            logger.info("%s: shard exists, skipping", args.bc_run)
            return 0
        bank = dumps.load_set(
            dumps.set_dir(locations.embed_dir, args.bc_run, dumps.STL10_TRAIN)
        )
        test = dumps.load_set(
            dumps.set_dir(locations.embed_dir, args.bc_run, dumps.STL10_TEST)
        )
        assert bank is not None and test is not None
        parts = [p.strip() for p in args.bc_parts.split(",") if p.strip()]
        result = baseline_controls.run_unit(
            args.bc_run, bank, test, cfg.eval, cfg.seed, parts
        )
        baseline_controls.save_run(shard, result)
        logger.info("%s: wrote %s", args.bc_run, shard)
        return 0
    missing = [r for r in runs if not (shard_dir / f"{r}.npz").exists()]
    if missing or not runs:
        raise FileNotFoundError(f"missing shards in {shard_dir}: {', '.join(missing)}")
    results = [baseline_controls.load_run(shard_dir / f"{r}.npz") for r in runs]
    written = baseline_controls.write_notes(
        results,
        cfg.eval,
        locations.notes_dir / args.note,
        args.bc_reference,
        cfg.seed,
        locations.kaleido_chrome,
        figures=not args.no_figures,
    )
    for note in written:
        print(f"wrote {note}")
    return 0


RETRIEVAL_SHARDS = "radius_usage_retrieval"


def _retrieval_figures(
    comparisons: list[retrieval.Comparison],
    note: pathlib.Path,
    chrome: pathlib.Path | None,
) -> list[str]:
    """One figure per task (mAP or recall@1): paired change vs lambda, 95% CIs."""
    prefix = note.parent / note.stem
    links: list[str] = []
    for task in retrieval.TASKS:
        metric = retrieval.task_metrics(task)[0]
        series: dict[str, plotting.Series] = {}
        for condition in retrieval.task_conditions(task)[1:]:
            rows = [
                c
                for c in comparisons
                if c.task == task and c.metric == metric and c.condition == condition
            ]
            series[condition] = plotting.Series(
                x=[c.lam for c in rows],
                y=[100 * c.difference for c in rows],
                low=[100 * c.ci95[0] for c in rows],
                high=[100 * c.ci95[1] for c in rows],
            )
        fig = plotting.line_plot(
            series,
            f"{metric} change vs baseline, {retrieval.TASK_TITLES[task]} "
            "(z, 95% paired CI)",
            "lambda",
            f"{metric} change (points)",
            reference=0.0,
        )
        out = plotting.export_png(
            fig, f"{prefix}_{task}_{metric.replace('@', 'at')}.png", chrome
        )
        links.append(reporting.image_link(out, note, f"{metric} change, {task}"))
    return links


def _run_retrieval(args: argparse.Namespace) -> int:
    """Retrieval study: one run's shard (`--retrieval-run`) or the aggregate note."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr
    )
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    shard_dir = locations.runs_dir / RETRIEVAL_SHARDS
    runs = sweep.load_runs(locations.runs_dir, args.retrieval)
    if args.retrieval_run is not None:
        run = next((r for r in runs if r.name == args.retrieval_run), None)
        if run is None:
            raise ValueError(f"no run {args.retrieval_run!r} under {args.retrieval}_*")
        train = dumps.load_set(
            dumps.set_dir(locations.embed_dir, run.name, dumps.STL10_TRAIN)
        )
        test = dumps.load_set(
            dumps.set_dir(locations.embed_dir, run.name, dumps.STL10_TEST)
        )
        if train is None or test is None:
            raise FileNotFoundError(f"{run.name}: needs stl10_train and stl10_test")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("%s on %s", run.name, device)
        result = retrieval.evaluate_shards(
            run.name, run.lam, cfg.seed, train, test, cfg.eval, device
        )
        logger.info("wrote %s", retrieval.save_result(shard_dir, result))
        return 0
    results = [
        # lambda and training seed come from the run's resolved config.
        dataclasses.replace(
            retrieval.load_result(shard_dir / f"{r.name}.npz"), lam=r.lam, seed=r.seed
        )
        for r in runs
        if (shard_dir / f"{r.name}.npz").exists()
    ]
    missing = [r.name for r in runs if not (shard_dir / f"{r.name}.npz").exists()]
    if not results:
        print(f"no retrieval shards for {args.retrieval}_*", file=sys.stderr)
        return 1
    margin = cfg.eval.retrieval_margin
    comparisons = retrieval.compare(
        results, margin, cfg.eval.num_bootstrap, numpy.random.default_rng(cfg.seed)
    )
    note = locations.notes_dir / args.note
    reporting.update_section(
        note,
        "analyze.retrieval_comparison",
        "### Paired change vs the baseline, per lambda\n\n"
        "z-space, Euclidean distance; differences in points (condition minus "
        "baseline on the same queries, mAP / precision@k / recall@k in %), averaged "
        "over training seeds; intervals are paired bootstrap percentile intervals "
        f"({cfg.eval.num_bootstrap} replicates, queries resampled within each "
        f"seed). Verdict with the margin fixed before the results at ±{100 * margin:g}"
        " points: `equivalent` if the 90% interval lies inside the margin, "
        "`different` if not and the 95% interval excludes 0, else `inconclusive`. "
        f"Shuffle repeats ({cfg.eval.num_randomizations} per randomized condition) "
        "are averaged per query and are not part of the interval.\n\n"
        + reporting.markdown_table(retrieval.comparison_rows(comparisons)),
    )
    values = "".join(
        f"#### {retrieval.TASK_TITLES[task]}\n\n"
        + reporting.markdown_table(retrieval.value_rows(results, task))
        + "\n\n"
        for task in retrieval.TASKS
    )
    reporting.update_section(
        note,
        "analyze.retrieval_values",
        "### Metric values (%), mean ± std over training seeds\n\n" + values.rstrip(),
    )
    reporting.update_section(
        note,
        "analyze.retrieval_r2",
        "### Held-out R^2 of log-radius from direction (test center views)\n\n"
        + reporting.markdown_table(
            [
                {"run": r.name, "lambda": r.lam, "seed": r.seed, "R^2": r.r2_test}
                for r in sorted(results, key=lambda r: (r.lam, r.seed))
            ]
        )
        + "\n\nRuns without a shard: "
        + (", ".join(missing) if missing else "none")
        + ".",
    )
    if not args.no_figures:
        links = _retrieval_figures(comparisons, note, locations.kaleido_chrome)
        reporting.update_section(
            note, "analyze.retrieval_figures", "### Figures\n\n" + "\n\n".join(links)
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


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _run_shift_unit(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    for out in usage_shift.run_unit(
        cfg, locations, args.run_name, args.shift_part, _stderr
    ):
        print(f"wrote {out}")
    return 0


def _run_view_signal(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    if args.view_signal_run is not None:
        out = usage_view_signal.run_unit(cfg, locations, args.view_signal_run, _stderr)
        print(f"wrote {out}")
        return 0
    written = usage_view_signal.write_notes(
        cfg, locations, args.view_signal, args.note, not args.no_figures
    )
    if not written:
        print(f"no unit files for runs {args.view_signal}_*", file=sys.stderr)
        return 1
    for out in written:
        print(f"wrote {out}")
    return 0


def _run_svhn_control(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    study = svhn_control.STUDIES[args.control_study]
    if args.svhn_control_stats:
        print(f"wrote {svhn_control.compute_image_stats(locations, _stderr, study)}")
    elif args.svhn_control_run:
        out = svhn_control.run_unit(
            cfg, locations, args.svhn_control_run, _stderr, study
        )
        print(f"wrote {out}")
    else:
        for out in svhn_control.write_notes(
            cfg, locations, args.svhn_control, args.note, not args.no_figures, study
        ):
            print(f"wrote {out}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    """E3: one radius-free run's units (`--compare-unit`) or the aggregate note."""
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    if args.compare_unit is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for out in sphere_compare.run_unit(
            cfg, locations, args.compare_unit, _stderr, device
        ):
            print(f"wrote {out}")
        return 0
    out = sphere_compare.write_note(
        cfg,
        locations,
        args.compare_prefix,
        args.baseline_prefix,
        args.note,
        not args.no_figures,
    )
    print(f"wrote {out}")
    return 0


def _run_shift(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    for out in usage_shift.write_notes(
        cfg, locations, args.shift, args.note, not args.no_figures
    ):
        print(f"wrote {out}")
    return 0


def _run_confidence_unit(args: argparse.Namespace) -> int:
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths()
    out = usage_confidence.run_unit(
        cfg,
        locations,
        args.run_name,
        args.confidence_space,
        _stderr,
        args.confidence_part,
    )
    print(f"wrote {out}")
    return 0


def _run_confidence(args: argparse.Namespace) -> int:
    locations = paths.load_paths()
    for out in usage_confidence.write_notes(
        locations,
        args.confidence,
        args.confidence_space,
        args.note,
        not args.no_figures,
    ):
        print(f"wrote {out}")
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
    if args.randomize_radius is not None:
        return _run_randomize_radius(args)
    if args.retrieval is not None:
        return _run_retrieval(args)
    if args.lowlabel is not None:
        return _run_lowlabel(args)
    if args.frozen_linear is not None:
        return _run_frozen_linear(args)
    if args.view_signal is not None:
        return _run_view_signal(args)
    if args.svhn_control is not None:
        return _run_svhn_control(args)
    if args.baseline_controls is not None:
        return _run_baseline_controls(args)
    if args.compare_prefix is not None:
        return _run_compare(args)
    if args.shift is not None:
        return _run_shift(args)
    if args.shift_unit:
        return _run_shift_unit(args)
    if args.confidence is not None:
        return _run_confidence(args)
    if args.confidence_unit:
        return _run_confidence_unit(args)
    if args.aggregate_controls is not None:
        return _run_aggregate_controls(args)
    if args.epoch_unit is not None:
        return _run_epoch_unit(args)
    if args.epoch_sweep is not None:
        return _run_epoch_sweep(args)
    if args.aggregate_transfer is not None:
        return _run_aggregate_transfer(args)
    if args.transfer:
        return _run_transfer(args)
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
