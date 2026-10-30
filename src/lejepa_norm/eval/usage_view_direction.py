"""Driver of the view-direction study (`view_direction`).

`run_unit` reads the `augreg_<run>` stl10_test dump written for the augmentation
regression (views with their saved parameters), the STL-10 test pixels (for crop
content) and the clean `stl10_train` bank (for the linear probe), and saves one JSON
per checkpoint. `write_notes` turns the unit files of a source-run prefix into the
generated sections of one note, with the pre-registered checks V1 to V4
(`research_notes/radius_view_direction/preregistration.md`).
"""

import collections.abc
import dataclasses
import json
import pathlib

import numpy

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import reporting
from lejepa_norm.data import layout
from lejepa_norm.eval import aug_regression
from lejepa_norm.eval import downstream
from lejepa_norm.eval import dumps
from lejepa_norm.eval import sweep
from lejepa_norm.eval import usage_aug_regression
from lejepa_norm.eval import view_direction

UNIT_FILE = "view_direction.json"
# Pre-registered thresholds (preregistration.md). The count is the same 7 of 9 that
# V1 and V3 name; V2 and V4 do not name one, so the same count is used.
V1_MIN_GAIN = 0.20
V2_MAX_GAIN = 0.03
V3_MAX_AUC = 0.53
MIN_CHECKPOINTS = 7

Row = dict[str, str | int | float]
Interval = list[float]


@dataclasses.dataclass
class Unit:
    """Saved results of one checkpoint (JSON-serializable)."""

    run: str
    lam: float
    seed: int
    view_accuracy: float
    spread: dict[str, float]
    ridge_penalty: float
    ridge_cv_r2: float
    r2: dict[str, list[float]]
    increments: dict[str, list[float]]
    resid_auc: dict[str, Interval]


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
    name = usage_aug_regression.dump_name(run_name)
    dump = dumps.load_set(dumps.set_dir(locations.embed_dir, name, dumps.STL10_TEST))
    bank = dumps.load_center_set(
        dumps.set_dir(locations.embed_dir, run_name, dumps.STL10_TRAIN)
    )
    if dump is None or bank is None:
        raise FileNotFoundError(f"{run_name}: missing augreg stl10_test or stl10_train")
    if dump.aug is None:
        raise FileNotFoundError(f"{name}: dump has no `aug` array")
    z = dump.z[:, 1:].astype(numpy.float64)
    n, k, d = z.shape
    log(f"{run_name}: {n} images x {k} views")
    images = numpy.load(layout.images_path(locations.data_dir, "stl10", "test"))
    aug = dump.aug.astype(numpy.float64)
    log("crop content")
    content = view_direction.crop_content(images[dump.image_id], aug)
    design = aug_regression.build_design(aug, images.shape[1], images.shape[2])
    y = numpy.log(numpy.linalg.norm(z, axis=-1))
    probe = downstream.LinearProbe(
        bank.z.astype(numpy.float64), bank.label, usage_aug_regression.NUM_CLASSES
    )
    correct = probe.predict(z.reshape(n * k, d)).reshape(n, k) == dump.label[:, None]
    rng = numpy.random.default_rng(cfg.seed)
    log("regression")
    result = view_direction.analyze_direction(
        y, design, z, content, correct, cfg.eval.num_bootstrap, 5, rng
    )
    spread = view_direction.direction_spread(z, rng)
    out = locations.runs_dir / name / UNIT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    unit = Unit(
        run=run_name,
        lam=resolved.loss.lambda_,
        seed=resolved.seed,
        view_accuracy=float(correct.mean()),
        spread=dataclasses.asdict(spread),
        ridge_penalty=result.ridge_penalty,
        ridge_cv_r2=result.ridge_cv_r2,
        r2=result.r2,
        increments=result.increments,
        resid_auc={k_: list(v) for k_, v in result.resid_auc.items()},
    )
    out.write_text(json.dumps(dataclasses.asdict(unit), indent=1) + "\n")
    return out


def _load_units(locations: paths.Paths, prefix: str) -> tuple[list[Unit], list[Row]]:
    units: list[Unit] = []
    missing: list[Row] = []
    for run in sweep.load_runs(locations.runs_dir, prefix):
        path = locations.runs_dir / usage_aug_regression.dump_name(run.name) / UNIT_FILE
        if path.is_file():
            units.append(Unit(**json.loads(path.read_text())))
        else:
            missing.append({"run": run.name, "lambda": run.lam, "missing": UNIT_FILE})
    return units, missing


def _interval(v: Interval) -> str:
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def verdicts(units: list[Unit]) -> list[Row]:
    """Pre-registered checks V1 to V4 and the reading."""
    n = len(units)
    direction = "direction | params"
    v1 = sum(u.increments[direction][0] >= V1_MIN_GAIN for u in units)
    v2 = sum(
        u.increments["content | params+direction"][0] <= V2_MAX_GAIN for u in units
    )
    v3 = sum(u.resid_auc["params+direction+content"][0] < V3_MAX_AUC for u in units)
    by_lam: dict[float, list[float]] = {}
    for u in units:
        by_lam.setdefault(u.lam, []).append(u.increments[direction][0])
    means = [float(numpy.mean(by_lam[lam])) for lam in sorted(by_lam)]
    v4 = all(a < b for a, b in zip(means, means[1:], strict=False)) and all(
        g > 0 for gains in by_lam.values() for g in gains
    )
    if v1 < MIN_CHECKPOINTS:
        reading = (
            "V1 fails: the within-image radius is largely independent of direction"
        )
        if v3 >= MIN_CHECKPOINTS:
            reading += " (but V3 holds: not covered by the pre-registered rule)"
    elif v3 >= MIN_CHECKPOINTS:
        reading = "V1 and V3 hold: the within-image radius is mostly view direction"
    else:
        reading = (
            "V1 holds, V3 fails: direction explains the radius, which still carries "
            "correctness information beyond direction and content"
        )
    return [
        {
            "check": f"V1 direction adds >= {V1_MIN_GAIN} CV R2 over params",
            "checkpoints": f"{v1}/{n}",
        },
        {
            "check": f"V2 content adds <= {V2_MAX_GAIN} over params+direction",
            "checkpoints": f"{v2}/{n}",
        },
        {
            "check": f"V3 residual AUC < {V3_MAX_AUC} after params+direction+content",
            "checkpoints": f"{v3}/{n}",
        },
        {
            "check": "V4 direction gain rises with lambda, positive in every seed",
            "checkpoints": "holds" if v4 else "fails",
        },
        {"check": "Reading of the pre-registration", "checkpoints": reading},
    ]


def tables(units: list[Unit]) -> dict[str, list[Row]]:
    """Per-checkpoint rows of the generated sections, keyed by section name."""
    out: dict[str, list[Row]] = {k: [] for k in ("spread", "r2", "increments", "auc")}
    for u in units:
        base: Row = {"lambda": u.lam, "seed": u.seed}
        out["spread"].append(
            {
                **base,
                "within-image angle (deg)": u.spread["within_mean"],
                "p90": u.spread["within_p90"],
                "to image mean": u.spread["to_mean"],
                "different images": u.spread["across_mean"],
            }
        )
        out["r2"].append(
            {
                **base,
                **{f"CV {k}": v[3] for k, v in u.r2.items()},
                "ridge CV R2": u.ridge_cv_r2,
            }
        )
        out["increments"].append(
            {
                **base,
                **{
                    k: f"{v[0]:+.3f} (in-sample {v[1]:+.3f} [{v[2]:+.3f}, {v[3]:+.3f}])"
                    for k, v in u.increments.items()
                },
            }
        )
        out["auc"].append({**base, **{k: _interval(v) for k, v in u.resid_auc.items()}})
    return out


def write_notes(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    prefix: str,
    note: pathlib.Path,
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
        "analyze.dir_inputs",
        "### Inputs\n\nRuns: "
        + ", ".join(f"`{u.run}`" for u in units)
        + ".\n\nUnit files missing:\n\n"
        + (reporting.markdown_table(missing) if missing else "None."),
    )
    reporting.update_section(
        path,
        "analyze.dir_verdicts",
        "### Pre-registered checks\n\n" + reporting.markdown_table(verdicts(units)),
    )
    reporting.update_section(
        path,
        "analyze.dir_spread",
        "### How far apart are the views of one image?\n\n"
        "Angles between view directions (8 global views per test image). `different "
        "images` pairs each view with the same view slot of a random other image.\n\n"
        + reporting.markdown_table(rows["spread"]),
    )
    reporting.update_section(
        path,
        "analyze.dir_r2",
        "### Held-out within-image R2 of the nested models\n\n"
        "Image-grouped 5-fold cross-validation of the within-image regression of "
        "`log ||z||`. `ridge CV R2` is the held-out R2 of the direction ridge "
        "prediction alone (the `direction_fit` covariate is cross-fitted on the same "
        "folds, so the second stage has the usual mild stacking optimism).\n\n"
        + reporting.markdown_table(rows["r2"]),
    )
    reporting.update_section(
        path,
        "analyze.dir_increments",
        "### Gain of a block over the model without it\n\n"
        "Held-out gain in R2, then the in-sample gain with a 95% image-bootstrap "
        "interval.\n\n" + reporting.markdown_table(rows["increments"]),
    )
    reporting.update_section(
        path,
        "analyze.dir_auc",
        "### Does the leftover radius still predict correctness?\n\n"
        "Within-image AUC (95% image-bootstrap interval) of the radius left over "
        "after each model (cross-fitted) for the frozen linear probe being correct "
        "on the view; 0.5 is chance.\n\n" + reporting.markdown_table(rows["auc"]),
    )
    return [path]
