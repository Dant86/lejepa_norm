"""Driver of the augmentation-parameter regression study (`aug_regression`).

`run_unit` reads the re-embedded `stl10_test` dump of one run (views with their
augmentation parameters, run name `<dump prefix>_<source run>`), fits a linear probe
on the clean `stl10_train` center views of the source run, and saves the results of
parts A to C as JSON next to the dump's log directory. `write_notes` turns the unit
files of a source-run prefix into one note with per-checkpoint tables (no pooling over
seeds: sign agreement across seeds is reported instead).
"""

import collections.abc
import dataclasses
import json
import pathlib

import numpy
import plotly.graph_objects as go

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import aug_regression
from lejepa_norm.eval import downstream
from lejepa_norm.eval import dumps
from lejepa_norm.eval import sweep

UNIT_FILE = "aug_regression.json"
DUMP_PREFIX = "augreg"
NUM_CLASSES = 10
# Pre-registered thresholds (preregistration.md, P2 and reading rules).
P2_MIN_R2 = 0.30
P2_MIN_CHECKPOINTS = 7

Row = dict[str, str | int | float]
Interval = list[float]  # [point, low, high]


@dataclasses.dataclass
class Unit:
    """Saved results of one checkpoint (JSON-serializable)."""

    run: str
    lam: float
    seed: int
    view_accuracy: float
    columns: list[str]
    beta: list[float]
    beta_interval: list[list[float]]
    within_r2: Interval
    cv_r2_linear: float
    cv_r2_trees: float
    within_share: float
    block_drop: dict[str, Interval]
    curve_grid: list[float]
    curve: list[float]
    slope_sd: float
    slope_alpha_corr: float
    correct_auc: dict[str, Interval]
    both_weights: list[float]
    retains: bool
    angle_r2: dict[str, Interval]
    angle_beta: list[float]


def dump_name(run_name: str) -> str:
    """Name of the re-embedded dump of the source run `run_name`."""
    return f"{DUMP_PREFIX}_{run_name}"


def run_unit(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    run_name: str,
    log: collections.abc.Callable[[str], None],
) -> pathlib.Path:
    """Computes and saves the unit file of one source run; returns its path.

    Raises:
        FileNotFoundError: a dump is missing, or the dump has no augmentation
            parameters.
    """
    resolved = config.load_config(
        locations.runs_dir / run_name / "config.resolved.yaml"
    )
    dump = dumps.load_set(
        dumps.set_dir(locations.embed_dir, dump_name(run_name), dumps.STL10_TEST)
    )
    bank = dumps.load_center_set(
        dumps.set_dir(locations.embed_dir, run_name, dumps.STL10_TRAIN)
    )
    if dump is None or bank is None:
        raise FileNotFoundError(f"{run_name}: missing augreg stl10_test or stl10_train")
    if dump.aug is None:
        raise FileNotFoundError(f"{dump_name(run_name)}: dump has no `aug` array")
    z = dump.z[:, 1:].astype(numpy.float64)  # augmented views only
    n, k, d = z.shape
    log(f"{run_name}: {n} images x {k} views")
    # The crops are drawn on the source image (STL-10 is 96 x 96).
    design = aug_regression.build_design(dump.aug, 96, 96)
    y = numpy.log(numpy.linalg.norm(z, axis=-1))
    rng = numpy.random.default_rng(cfg.seed)
    num_bootstrap = cfg.eval.num_bootstrap
    folds = 5
    regression = aug_regression.analyze_regression(
        y, design, num_bootstrap, folds, rng, cfg.seed
    )

    probe = downstream.LinearProbe(
        bank.z.astype(numpy.float64), bank.label, NUM_CLASSES
    )
    pred = probe.predict(z.reshape(n * k, d)).reshape(n, k)
    correct = pred == dump.label[:, None]
    angle = aug_regression.angle_to_mean_direction(z)
    quality = aug_regression.analyze_view_quality(
        aug_regression.demean(y),
        aug_regression.demean(design.x),
        correct,
        angle,
        num_bootstrap,
        folds,
        rng,
    )
    out = locations.runs_dir / dump_name(run_name) / UNIT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    unit = Unit(
        run=run_name,
        lam=resolved.loss.lambda_,
        seed=resolved.seed,
        view_accuracy=float(correct.mean()),
        columns=list(aug_regression.COLUMN_NAMES),
        beta=regression.beta.tolist(),
        beta_interval=regression.beta_interval.tolist(),
        within_r2=list(regression.within_r2),
        cv_r2_linear=regression.cv_r2_linear,
        cv_r2_trees=regression.cv_r2_trees,
        within_share=regression.within_share,
        block_drop={k_: list(v) for k_, v in regression.block_drop.items()},
        curve_grid=regression.curve_grid.tolist(),
        curve=regression.curve.tolist(),
        slope_sd=regression.slope_sd,
        slope_alpha_corr=regression.slope_alpha_corr,
        correct_auc={k_: list(v) for k_, v in quality.correct_auc.items()},
        both_weights=quality.both_weights.tolist(),
        retains=quality.retains,
        angle_r2={k_: list(v) for k_, v in quality.angle_r2.items()},
        angle_beta=quality.angle_beta.tolist(),
    )
    out.write_text(json.dumps(dataclasses.asdict(unit), indent=1) + "\n")
    return out


def _load_units(locations: paths.Paths, prefix: str) -> tuple[list[Unit], list[Row]]:
    units: list[Unit] = []
    missing: list[Row] = []
    for run in sweep.load_runs(locations.runs_dir, prefix):
        path = locations.runs_dir / dump_name(run.name) / UNIT_FILE
        if path.is_file():
            units.append(Unit(**json.loads(path.read_text())))
        else:
            missing.append({"run": run.name, "lambda": run.lam, "missing": UNIT_FILE})
    return units, missing


def _interval(v: Interval) -> str:
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def tables(units: list[Unit]) -> dict[str, list[Row]]:
    """Per-checkpoint rows of the generated sections, keyed by section name."""
    out: dict[str, list[Row]] = {
        k: [] for k in ("r2", "coef", "blocks", "quality", "angle")
    }
    for u in units:
        base: Row = {"lambda": u.lam, "seed": u.seed}
        out["r2"].append(
            {
                **base,
                "within R2 [95%]": _interval(u.within_r2),
                "CV R2 linear": u.cv_r2_linear,
                "CV R2 trees": u.cv_r2_trees,
                "within share of var": u.within_share,
                "slope sd": u.slope_sd,
                "corr(slope, mean)": u.slope_alpha_corr,
                "view acc": u.view_accuracy,
            }
        )
        coef: Row = dict(base)
        for name, b, (low, high) in zip(
            u.columns, u.beta, u.beta_interval, strict=True
        ):
            coef[name] = f"{b:+.3f} [{low:+.3f}, {high:+.3f}]"
        out["coef"].append(coef)
        out["blocks"].append(
            {**base, **{k: _interval(v) for k, v in u.block_drop.items()}}
        )
        out["quality"].append(
            {
                **base,
                **{f"AUC {k}": _interval(v) for k, v in u.correct_auc.items()},
                "r retains": u.retains,
            }
        )
        out["angle"].append(
            {**base, **{f"R2 {k}": _interval(v) for k, v in u.angle_r2.items()}}
        )
    return out


def _sign_agreement(group: list[Unit]) -> bool:
    """P4: the crop-area, blur-gate and sigma effects have one sign over seeds."""
    columns = group[0].columns

    def signs(value: collections.abc.Callable[[Unit], float]) -> set[float]:
        return {float(numpy.sign(value(u))) for u in group}

    checks = [
        signs(lambda u: u.curve[-1] - u.curve[0]),
        signs(lambda u: u.beta[columns.index("blur")]),
        signs(lambda u: u.beta[columns.index("sigma")]),
    ]
    return all(len(s) == 1 for s in checks)


def verdicts(units: list[Unit]) -> list[Row]:
    """Pre-registered checks P1 to P4 and the overall reading."""
    n = len(units)
    p2 = sum(u.cv_r2_linear >= P2_MIN_R2 for u in units)
    trees_ok = sum(u.cv_r2_trees - u.cv_r2_linear <= 0.10 for u in units)
    rising = sum(u.curve[-1] > u.curve[0] for u in units)
    by_lam: dict[float, list[Unit]] = {}
    for u in units:
        by_lam.setdefault(u.lam, []).append(u)
    agree = sum(_sign_agreement(g) for g in by_lam.values())
    retains = sum(u.retains for u in units)
    if p2 < P2_MIN_CHECKPOINTS:
        reading = "rejected (P2 fails)"
    elif rising == n and retains == 0:
        reading = "supported (P1, P2, P3 hold)"
    elif rising == n:
        reading = "partly supported (P1, P2 hold; P3 fails)"
    else:
        reading = "not supported by the pre-registered rule"
    return [
        {"check": "P1 area effect rises", "checkpoints": f"{rising}/{n}"},
        {
            "check": f"P2 CV R2 linear >= {P2_MIN_R2} ({P2_MIN_CHECKPOINTS} needed)",
            "checkpoints": f"{p2}/{n}",
        },
        {"check": "P2 trees add <= 0.10 over linear", "checkpoints": f"{trees_ok}/{n}"},
        {
            "check": "P3 `r` retains the AUC (0 means P3 holds)",
            "checkpoints": f"{retains}/{n}",
        },
        {
            "check": "P4 signs agree over seeds (lambdas)",
            "checkpoints": f"{agree}/{len(by_lam)}",
        },
        {"check": "Reading of the pre-registration", "checkpoints": reading},
    ]


def write_notes(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    prefix: str,
    note: pathlib.Path,
    figures: bool,
) -> list[pathlib.Path]:
    """Writes the generated sections of the note from the unit files of `prefix_*`.

    Returns the written paths (empty if no unit file exists).
    """
    del cfg
    units, missing = _load_units(locations, prefix)
    if not units:
        return []
    path = locations.notes_dir / note
    rows = tables(units)
    reporting.update_section(
        path,
        "analyze.aug_inputs",
        "### Inputs\n\nRuns: "
        + ", ".join(f"`{u.run}`" for u in units)
        + ".\n\nUnit files missing:\n\n"
        + (reporting.markdown_table(missing) if missing else "None."),
    )
    reporting.update_section(
        path,
        "analyze.aug_verdicts",
        "### Pre-registered checks\n\n" + reporting.markdown_table(verdicts(units)),
    )
    reporting.update_section(
        path,
        "analyze.aug_r2",
        "### Within-image R2 of the augmentation parameters\n\n"
        "`within R2`: share of the within-image variance of `log ||z||` explained "
        "by the parameters (in-sample, with an image-bootstrap 95% interval); `CV` "
        "columns are held out over image folds; `within share of var` is the share "
        "of the total variance that is within-image (the rest is the image effect); "
        "`slope sd` is the estimated sd of the image-specific log-area slope "
        "(two-stage estimate with the sampling variance removed; deviation from the "
        "pre-registered mixed model) and `corr(slope, mean)` its correlation with "
        "the image mean (attenuated by noise); `view acc` is the frozen linear "
        "probe's accuracy on the augmented views.\n\n"
        + reporting.markdown_table(rows["r2"]),
    )
    reporting.update_section(
        path,
        "analyze.aug_blocks",
        "### Block partial R2 per augmentation type\n\n"
        "Drop in within-image R2 when the block (and the area x blur interaction for "
        "crop and blur) is removed; 95% image-bootstrap intervals. Shares overlap "
        "(parameters are correlated) and do not sum to the total.\n\n"
        + reporting.markdown_table(rows["blocks"]),
    )
    reporting.update_section(
        path,
        "analyze.aug_coef",
        "### Coefficients per parameter (standardized)\n\n"
        "Change of `log ||z||` per standard deviation of the parameter, other "
        "parameters held fixed, image fixed effects; the three log-area columns "
        "are spline terms (see the effect curve). 95% image-bootstrap intervals.\n\n"
        + reporting.markdown_table(rows["coef"]),
    )
    reporting.update_section(
        path,
        "analyze.aug_quality",
        "### Does the radius residual carry view information? (part C-a)\n\n"
        "Within-image AUC for the frozen linear probe being correct on a view, using "
        "`yhat` (radius predicted from the parameters, cross-fitted), `r` (the "
        "remaining radius) and the fixed-effects combination `both`. `r retains` is "
        "true if `r` is more than half way from 0.5 to the AUC of `both`.\n\n"
        + reporting.markdown_table(rows["quality"]),
    )
    reporting.update_section(
        path,
        "analyze.aug_angle",
        "### Angle to the image's mean direction (part C-b)\n\n"
        "Within-image R2 of the angle (view direction to the mean of the image's "
        "augmented directions) on `yhat`, `r` and both; `r_increment` is the gain "
        "of `r` over `yhat`.\n\n" + reporting.markdown_table(rows["angle"]),
    )
    if figures:
        links = [
            reporting.image_link(
                plotting.export_png(
                    fig,
                    path.with_name(f"{path.stem}_{stem}.png"),
                    locations.kaleido_chrome,
                ),
                path,
                stem,
            )
            for stem, fig in _figures(units).items()
        ]
        reporting.update_section(
            path, "analyze.aug_figures", "### Figures\n\n" + "\n\n".join(links)
        )
    return [path]


def _figures(units: list[Unit]) -> dict[str, go.Figure]:
    """Effect curve of log crop area, one line per checkpoint."""
    series = {
        f"lam {u.lam:g} s{u.seed}": plotting.Series(u.curve_grid, u.curve)
        for u in units
    }
    return {
        "area_curve": plotting.line_plot(
            series,
            "Effect of crop area on the radius",
            "log crop area",
            "change in log ||z||",
        )
    }
