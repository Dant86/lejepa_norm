"""Driver of the "is the radius used under shift?" study (`shift`, `ood_radius`).

`run_unit` computes the per-image outcomes of one run from its embedding dumps and
saves them next to the run (`shift_cells.npz`, `shift_ood.npz`); `write_notes`
aggregates the saved units of a run prefix per lambda into generated note sections.
Only the center-view z is ever read (STL-10-C would not fit in memory otherwise).
"""

import collections.abc
import pathlib

import numpy

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import dumps
from lejepa_norm.eval import ood_radius
from lejepa_norm.eval import shards
from lejepa_norm.eval import shift
from lejepa_norm.eval import sweep

CELLS_FILE = "shift_cells.npz"
OOD_FILE = "shift_ood.npz"
PARTS = ("classification", "ood", "both")

Row = dict[str, str | int | float]


def run_unit(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    run_name: str,
    part: str,
    log: collections.abc.Callable[[str], None],
) -> list[pathlib.Path]:
    """Computes and saves the unit files of one run; returns the written paths.

    Raises:
        FileNotFoundError: a required embedding dump is missing.
    """
    run_dir = locations.runs_dir / run_name
    resolved = config.load_config(run_dir / "config.resolved.yaml")

    def center(name: str) -> shards.CenterShard:
        loaded = dumps.load_center_set(
            dumps.set_dir(locations.embed_dir, run_name, name)
        )
        if loaded is None:
            raise FileNotFoundError(f"{run_name}: no {name} embedding dump")
        return loaded

    bank, test = center(dumps.STL10_TRAIN), center(dumps.STL10_TEST)
    written: list[pathlib.Path] = []
    if part in ("classification", "both"):
        corrupted = center(dumps.STL10_C)
        log(f"{run_name}: {corrupted.z.shape[0]} corrupted center embeddings loaded")
        correct, image_id, radius_shift, residual = shift.evaluate_cells(
            bank.z.astype(numpy.float64),
            bank.label,
            test,
            corrupted,
            k=cfg.eval.knn_k,
            num_repeats=cfg.eval.num_randomizations,
            cv_folds=cfg.eval.cv_folds,
            seed=cfg.seed,
            log=log,
        )
        out = run_dir / CELLS_FILE
        shift.save_unit(
            out,
            shift.ShiftUnit(
                run_name,
                resolved.loss.lambda_,
                resolved.seed,
                correct,
                image_id,
                radius_shift,
                residual,
            ),
        )
        written.append(out)
    if part in ("ood", "both"):
        scores = {
            name: ood_radius.compute_scores(
                bank.z.astype(numpy.float64),
                test.z.astype(numpy.float64),
                center(name).z.astype(numpy.float64),
                cfg.eval.knn_k,
                cfg.seed,
            )
            for name in dumps.OOD_SETS
        }
        out = run_dir / OOD_FILE
        ood_radius.save_unit(
            out,
            ood_radius.OodUnit(
                run_name,
                resolved.loss.lambda_,
                resolved.seed,
                test.z.shape[0],
                scores,
            ),
        )
        written.append(out)
    return written


def _classification_sections(
    cell_units: list[shift.ShiftUnit],
    comparisons: list[shift.ShiftComparison],
    lam: float,
    cfg: config.ExperimentConfig,
) -> dict[str, str]:
    margin = cfg.eval.equivalence_margin
    pairs = (
        "Differences are in accuracy points (condition minus baseline on the same "
        "corrupted test images), averaged over training seeds; intervals are paired "
        f"bootstrap percentile intervals ({cfg.eval.num_bootstrap} replicates, test "
        "images resampled within each seed, the same resample for every cell). "
        f"Verdict, with the equivalence margin set to ±{100 * margin:g} points before "
        "looking at results: `equivalent` if the 90% interval lies inside the margin, "
        "`different` if not equivalent and the 95% interval excludes 0, otherwise "
        "`inconclusive`. Bank / training set: clean STL-10 train."
    )
    mine = [u for u in cell_units if u.lam == lam]
    correlations = [r for r in shift.shift_correlations(mine) if r["lambda"] == lam]
    return {
        "analyze.shift_context": (
            "### Radius shift and baseline accuracy per corruption\n\n"
            "Averaged over the 5 severities and the seeds. `log-radius shift` = mean "
            "paired change of the center-view log-radius vs the clean twin; "
            "`clean-map residual` = mean of `log r - f(u)` with the clean-fitted "
            "direction -> log-radius map `f`; accuracies are those of the frozen "
            "readers on the unmodified corrupted embeddings.\n\n"
            + reporting.markdown_table(shift.context_rows(mine, comparisons))
        ),
        "analyze.shift_corruption": (
            "### Accuracy change per corruption (severities averaged)\n\n"
            + pairs
            + "\n\n"
            + reporting.markdown_table(shift.corruption_rows(comparisons, lam))
        ),
        "analyze.shift_correlation": (
            "### Does the radius shift predict the accuracy change?\n\n"
            "Spearman correlation over the 50 (corruption, severity) cells of the "
            "seed-mean radius shift and the seed-mean accuracy change of the frozen "
            "protocol. Descriptive: cells share images and corruption families.\n\n"
            + reporting.markdown_table(correlations)
        ),
        "analyze.shift_cells": (
            "### Accuracy change per corruption and severity\n\n"
            + pairs
            + "\n\n"
            + reporting.markdown_table(shift.cell_rows(comparisons, lam))
        ),
    }


def _ood_sections(
    results: list[ood_radius.OodResult], cfg: config.ExperimentConfig
) -> dict[str, str]:
    return {
        "analyze.ood_scores": (
            "### OOD detection: AUROC and FPR@95TPR\n\n"
            "STL-10 test (ID) vs OOD, z center views, bank = STL-10 train; higher "
            "score = more OOD; AUROC = P(OOD score > ID score); FPR@95TPR = fraction "
            "of OOD images kept when 95% of the ID images are kept. Seed means with "
            f"95% paired bootstrap intervals ({cfg.eval.num_bootstrap} replicates, ID "
            "and OOD images resampled separately). Definitions and pre-registered "
            "criteria: docstring of `lejepa_norm.eval.ood_radius`.\n\n"
            + reporting.markdown_table(ood_radius.score_rows(results))
        ),
        "analyze.ood_differences": (
            "### Does the radius add information? Pre-registered comparisons\n\n"
            f"Margin {ood_radius.OOD_AUROC_MARGIN:g} AUROC; `{ood_radius.ADDS}` iff "
            "the lower 95% bound of the difference exceeds the margin.\n\n"
            + reporting.markdown_table(ood_radius.difference_rows(results))
        ),
        "analyze.ood_oracle": (
            "### Weight picked by the optimistic combination\n\n"
            + reporting.markdown_table(ood_radius.oracle_rows(results))
        ),
    }


def write_notes(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    prefix: str,
    note: pathlib.Path,
    figures: bool,
) -> list[pathlib.Path]:
    """Aggregates the unit files of runs `prefix_*` into one note per lambda.

    The notes are `<note stem>_lam<lambda>.md` next to `note` (relative to the
    notes directory); returns their paths.
    """
    if cfg.eval.equivalence_margin != shift.EQUIVALENCE_MARGIN:
        raise ValueError("eval.equivalence_margin differs from the pre-registered one")
    cell_units: list[shift.ShiftUnit] = []
    ood_units: list[ood_radius.OodUnit] = []
    skipped: list[Row] = []
    for run in sweep.load_runs(locations.runs_dir, prefix):
        run_dir = locations.runs_dir / run.name
        if (run_dir / CELLS_FILE).is_file():
            cell_units.append(shift.load_unit(run_dir / CELLS_FILE))
        else:
            skipped.append({"run": run.name, "lambda": run.lam, "missing": CELLS_FILE})
        if (run_dir / OOD_FILE).is_file():
            ood_units.append(ood_radius.load_unit(run_dir / OOD_FILE))
        else:
            skipped.append({"run": run.name, "lambda": run.lam, "missing": OOD_FILE})
    rng = numpy.random.default_rng(cfg.seed)
    margin = cfg.eval.equivalence_margin
    comparisons = (
        shift.compare(cell_units, margin, cfg.eval.num_bootstrap, rng)
        if cell_units
        else []
    )
    ood_results = (
        ood_radius.evaluate(ood_units, cfg.eval.num_bootstrap, rng) if ood_units else []
    )
    base = locations.notes_dir / note
    written: list[pathlib.Path] = []
    for lam in sorted({u.lam for u in cell_units} | {u.lam for u in ood_units}):
        path = base.with_name(f"{base.stem}_lam{lam:g}{base.suffix}")
        mine = [r for r in skipped if r["lambda"] == lam]
        reporting.update_section(
            path,
            "analyze.shift_inputs",
            "### Inputs\n\nRuns aggregated: "
            + ", ".join(
                f"`{n}`"
                for n in sorted(
                    {u.name for u in [*cell_units, *ood_units] if u.lam == lam}
                )
            )
            + ".\n\nUnit files missing for this lambda:\n\n"
            + (reporting.markdown_table(mine) if mine else "None."),
        )
        sections: dict[str, str] = {}
        if any(u.lam == lam for u in cell_units):
            sections |= _classification_sections(cell_units, comparisons, lam, cfg)
        if any(r.lam == lam for r in ood_results):
            sections |= _ood_sections([r for r in ood_results if r.lam == lam], cfg)
        for name, content in sections.items():
            reporting.update_section(path, name, content)
        if figures and any(u.lam == lam for u in cell_units):
            links: list[str] = []
            for stem, fig, alt in shift.figures(comparisons, lam):
                out = plotting.export_png(
                    fig, path.parent / f"shift_{stem}.png", locations.kaleido_chrome
                )
                links.append(reporting.image_link(out, path, alt))
            reporting.update_section(
                path, "analyze.shift_figures", "### Figures\n\n" + "\n\n".join(links)
            )
        written.append(path)
    return written
