"""Driver of the radius view-signal study (`view_signal`, `view_signal_stats`).

`run_unit` computes one run's per-image outcomes from its embedding dumps and saves
them next to the run (`view_signal.npz`); `write_notes` aggregates the units of a
run prefix per lambda into generated note sections plus a cross-lambda summary.
STL-10-C is read with all views (z only), about 4 GB, so jobs need ~64 GB.
"""

import collections.abc
import pathlib

import numpy

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import dumps
from lejepa_norm.eval import shift
from lejepa_norm.eval import sweep
from lejepa_norm.eval import view_signal
from lejepa_norm.eval import view_signal_stats as stats

UNIT_FILE = "view_signal.npz"

Row = dict[str, str | int | float]

_ACCURACY_NOTE = (
    "Differences are in accuracy points (condition minus reference on the same "
    "images), averaged over training seeds; intervals are paired bootstrap "
    "percentile intervals ({b} replicates, test images resampled within each seed, "
    "the same resample for every comparison of a seed). Verdict, with the "
    "equivalence margin set to ±{m:g} points before looking at results: "
    "`equivalent` if the 90% interval lies inside the margin, `different` if not "
    "equivalent and the 95% interval excludes 0, otherwise `inconclusive`. "
    "Readers are fitted on the clean STL-10 train bank only."
)


def run_unit(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    run_name: str,
    log: collections.abc.Callable[[str], None],
) -> pathlib.Path:
    """Computes and saves the unit file of one run; returns its path.

    Raises:
        FileNotFoundError: a required embedding dump is missing.
    """
    run_dir = locations.runs_dir / run_name
    resolved = config.load_config(run_dir / "config.resolved.yaml")
    bank = dumps.load_center_set(
        dumps.set_dir(locations.embed_dir, run_name, dumps.STL10_TRAIN)
    )
    clean = dumps.load_view_set(
        dumps.set_dir(locations.embed_dir, run_name, dumps.STL10_TEST)
    )
    corrupted = dumps.load_view_set(
        dumps.set_dir(locations.embed_dir, run_name, dumps.STL10_C)
    )
    if bank is None or clean is None or corrupted is None:
        raise FileNotFoundError(f"{run_name}: missing stl10_train/test/stl10c dump")
    log(f"{run_name}: {corrupted.z.shape[0]} corrupted images, all views loaded")
    (
        image_id,
        geometry,
        null_geometry,
        correct,
        severity_rows,
        effective,
        residual_sd,
    ) = view_signal.evaluate_unit(
        bank.z.astype(numpy.float64),
        bank.label,
        clean,
        corrupted,
        k=cfg.eval.knn_k,
        cv_folds=cfg.eval.cv_folds,
        seed=cfg.seed,
        log=log,
    )
    out = run_dir / UNIT_FILE
    view_signal.save_unit(
        out,
        view_signal.ViewSignalUnit(
            run_name,
            resolved.loss.lambda_,
            resolved.seed,
            image_id,
            geometry,
            null_geometry,
            correct,
            severity_rows,
            effective,
            residual_sd,
        ),
    )
    return out


def _sections(
    lam: float,
    cfg: config.ExperimentConfig,
    oracle: list[stats.AccuracyComparison],
    aggregation: list[stats.AccuracyComparison],
    rows: dict[str, list[Row]],
) -> dict[str, str]:
    """Generated sections of one lambda's note."""
    margin = cfg.eval.equivalence_margin
    text = _ACCURACY_NOTE.format(b=cfg.eval.num_bootstrap, m=100 * margin)

    def mine(table: list[Row]) -> list[Row]:
        return [r for r in table if r["lambda"] == lam]

    return {
        "analyze.view_geometry": (
            "### A. Radial vs tangential share of the displacement\n\n"
            "Clean twin -> corrupted image. "
            "`F_rad = sum a^2 / sum (a^2 + t^2)` with `a = <z_x - z_c, u_c>` (radial, "
            "along the clean direction) and `t` the orthogonal remainder; ratio of "
            "sums over images and the level's cells (so strong severities weigh "
            "more), seed mean, 95% bootstrap interval. A displacement in a random "
            "direction has `F_rad` about 1/256. The null row is the displacement "
            "from a clean image to a different clean image.\n\n"
            + reporting.markdown_table(mine(rows["geometry"]))
        ),
        "analyze.view_geometry_cells": (
            "### A. Radial share per cell\n\n"
            + reporting.markdown_table(mine(rows["geometry_cells"]))
        ),
        "analyze.view_oracle": (
            "### B. Oracle radius: accuracy of frozen readers\n\n"
            + text
            + "\n\n"
            + reporting.markdown_table(
                stats.accuracy_rows([c for c in oracle if c.lam == lam])
            )
        ),
        "analyze.view_recovery": (
            "### B. Share of the corruption loss recovered by the oracle radius\n\n"
            "`(oracle radius - baseline) / (clean twin - baseline)`, only where the "
            f"mean loss exceeds {100 * stats.MIN_LOSS:g} points; 95% bootstrap "
            "interval of the ratio of seed-mean paired differences.\n\n"
            + reporting.markdown_table(mine(rows["recovery"]))
        ),
        "analyze.view_severity": (
            "### C. Corruption severity from the radius residual\n\n"
            "Per corruption, 6 x 8000 rows (clean twins = severity 0). "
            "`res = log r - f(u)` with the clean-fitted direction -> log-radius map. "
            "Spearman values: seed mean [95% bootstrap interval, images resampled "
            "with their 6 rows]. Held-out predictions: ridge (RidgeCV) on the "
            "standardized direction, on `[res, |res|, log r]` and on both, 5 folds "
            "grouped by image. Verdict on `direction + radius - direction` with the "
            f"margin {stats.SEVERITY_MARGIN:g} Spearman fixed in advance: "
            f"`{stats.DOES_NOT_ADD}` if the 90% interval is inside the margin, "
            f"`{stats.ADDS}` if the lower 95% bound exceeds it, `{stats.WORSE}` if "
            "the upper 95% bound is below minus the margin.\n\n"
            + reporting.markdown_table(mine(rows["severity"]))
        ),
        "analyze.view_aggregation": (
            "### D. Radius-weighted multi-view aggregation\n\n"
            "Image embedding = unit vector of the weighted sum of the K = 8 augmented "
            "view directions (scheme weights in the pre-registration), scaled to the "
            "bank-median radius; readers fitted on the bank directions at that scale. "
            + text
            + "\n\n"
            + reporting.markdown_table(
                stats.accuracy_rows([c for c in aggregation if c.lam == lam])
            )
        ),
        "analyze.view_effective": (
            "### D. Mean effective number of views per scheme\n\n"
            "`1 / sum p^2` with `p` the weights normalized to sum 1 (8 = uniform, "
            "1 = a single view); shows how strong each weighting is.\n\n"
            + reporting.markdown_table(mine(rows["effective"]))
        ),
        "analyze.view_accuracy": (
            "### Mean accuracies (seed means)\n\n"
            + reporting.markdown_table(mine(rows["accuracy"]))
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
    notes directory), and `note` itself gets the cross-lambda summary tables;
    returns their paths.
    """
    if cfg.eval.equivalence_margin != shift.EQUIVALENCE_MARGIN:
        raise ValueError("eval.equivalence_margin differs from the pre-registered one")
    units: list[view_signal.ViewSignalUnit] = []
    missing: list[Row] = []
    for run in sweep.load_runs(locations.runs_dir, prefix):
        path = locations.runs_dir / run.name / UNIT_FILE
        if path.is_file():
            units.append(view_signal.load_unit(path))
        else:
            missing.append({"run": run.name, "lambda": run.lam, "missing": UNIT_FILE})
    if not units:
        return []
    rng = numpy.random.default_rng(cfg.seed)
    margin = cfg.eval.equivalence_margin
    num = cfg.eval.num_bootstrap
    oracle = stats.compare_accuracy(
        units, stats.ORACLE_CONTRASTS, False, margin, num, rng
    )
    aggregation = stats.compare_accuracy(
        units, stats.AGGREGATION_CONTRASTS, True, margin, num, rng
    )
    rows: dict[str, list[Row]] = {
        "geometry": stats.geometry_rows(units, num, rng),
        "geometry_cells": stats.geometry_cell_rows(units),
        "recovery": stats.recovery_rows(units, num, rng),
        "severity": stats.severity_rows(units, num, rng),
        "effective": stats.effective_view_rows(units),
        "accuracy": stats.accuracy_summary_rows(units),
    }
    base = locations.notes_dir / note
    written: list[pathlib.Path] = []
    for lam in sorted({u.lam for u in units}):
        path = base.with_name(f"{base.stem}_lam{lam:g}{base.suffix}")
        runs = ", ".join(f"`{u.name}`" for u in units if u.lam == lam)
        absent = [r for r in missing if r["lambda"] == lam]
        reporting.update_section(
            path,
            "analyze.view_inputs",
            "### Inputs\n\nRuns aggregated: "
            + runs
            + ".\n\nUnit files missing for this lambda:\n\n"
            + (reporting.markdown_table(absent) if absent else "None."),
        )
        for name, content in _sections(lam, cfg, oracle, aggregation, rows).items():
            reporting.update_section(path, name, content)
        if figures:
            links: list[str] = []
            for stem, fig in stats.figures(rows["geometry_cells"], oracle, lam).items():
                out = plotting.export_png(
                    fig,
                    path.parent / f"{base.stem}_lam{lam:g}_{stem}.png",
                    locations.kaleido_chrome,
                )
                links.append(reporting.image_link(out, path, stem))
            reporting.update_section(
                path, "analyze.view_figures", "### Figures\n\n" + "\n\n".join(links)
            )
        written.append(path)
    overall = [stats.OVERALL, stats.CLEAN_TEST]
    reporting.update_section(
        base,
        "analyze.view_summary",
        "### Summary across lambda (overall rows)\n\n"
        "**B. Oracle radius** (all corruptions)\n\n"
        + reporting.markdown_table(
            stats.accuracy_rows([c for c in oracle if c.level == stats.OVERALL])
        )
        + "\n\n**D. Aggregation** (all corruptions and clean test)\n\n"
        + reporting.markdown_table(
            stats.accuracy_rows([c for c in aggregation if c.level in overall])
        )
        + "\n\n**A. Geometry** (all corruptions and null)\n\n"
        + reporting.markdown_table(
            [
                r
                for r in rows["geometry"]
                if r["level"] == stats.OVERALL or str(r["level"]).startswith("null")
            ]
        )
        + "\n\n**C. Severity** (mean over corruptions)\n\n"
        + reporting.markdown_table(
            [r for r in rows["severity"] if str(r["corruption"]).startswith("mean")]
        ),
    )
    written.append(base)
    return written
