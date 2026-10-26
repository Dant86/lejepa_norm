"""Driver of the "does the radius track confidence?" study (`confidence`).

`run_unit` computes the statistics of one run from its embedding dumps and saves
them next to the run (`confidence_<space>.npz`); `write_notes` aggregates the saved
units of a run prefix per group (one note per lambda for LeJEPA, one per loss for
the E2 baselines) into generated note sections, plus a summary section in the main
note. Only the center view is ever read (STL-10-C would not fit otherwise). z is
the analyzed space except for the E2 baselines, whose projector is discarded after
training (explicit exception, see the pre-registration note): they are read in
`h` and only on the clean `stl10_test` set unless their other dumps exist.
"""

import collections.abc
import pathlib

import numpy
import numpy.typing
import yaml

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import confidence
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import shards
from lejepa_norm.eval import shift

Row = dict[str, str | int | float]
FloatArray = numpy.typing.NDArray[numpy.float64]

# SVHN is deliberately excluded (upsampled 32 -> 96, confounds the low-level
# statistics; `research_notes/radius_usage/svhn_control.md`). Places365 `val`
# (downsampled to 96 x 96) is embedded by the `radius-usage-places-control` branch;
# its dump is used when present, the set name is fixed here to avoid a dependency.
PLACES_TEST = "places_test"
PARTS = ("all", "ood")
OOD_NAMES = {dumps.DTD_TEST: "dtd", PLACES_TEST: "places"}
SPACES = ("z", "h")


def unit_file(space: str) -> str:
    """File name of a run's unit for `space`."""
    return f"confidence_{space}.npz"


def _read_identity(run_dir: pathlib.Path) -> tuple[str, int]:
    """`(group, seed)` of a run from its resolved config (not parsed from names).

    Read as plain YAML so that runs of the E2 baselines load without their loss
    variants being part of this checkout's config schema.
    """
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    loss = resolved.get("loss", {})
    variant = str(loss.get("variant", "A"))
    seed = int(resolved.get("seed", 0))
    if variant in ("A", "B", "C"):
        return f"lam{float(loss.get('lambda', 0.05)):g}", seed
    return variant, seed


def _subset(
    query: confidence.Query, rows: numpy.typing.NDArray[numpy.intp]
) -> confidence.Query:
    return confidence.Query(
        rho=query.rho[rows],
        resid=query.resid[rows],
        label=query.label[rows],
        readers={
            name: confidence.ReaderOutput(o.p1[rows], o.margin[rows], o.pred[rows])
            for name, o in query.readers.items()
        },
    )


def _load(
    locations: paths.Paths, run_name: str, name: str, space: str
) -> shards.CenterShard | None:
    return dumps.load_center_set(
        dumps.set_dir(locations.embed_dir, run_name, name), space
    )


def run_unit(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    run_name: str,
    space: str,
    log: collections.abc.Callable[[str], None],
    part: str = "all",
) -> pathlib.Path:
    """Computes and saves the unit file of one run; returns its path.

    `part="ood"` recomputes only the OOD groups and keeps the rest of the existing
    unit (used when a new OOD dump, e.g. Places365, becomes available).

    Raises:
        FileNotFoundError: the `stl10_train` or `stl10_test` dump is missing.
        ValueError: `space` is not `"z"` or `"h"`.
    """
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    run_dir = locations.runs_dir / run_name
    group, seed = _read_identity(run_dir)
    bank = _load(locations, run_name, dumps.STL10_TRAIN, space)
    test = _load(locations, run_name, dumps.STL10_TEST, space)
    if bank is None or test is None:
        raise FileNotFoundError(f"{run_name}: stl10_train / stl10_test dump missing")
    corrupted = (
        _load(locations, run_name, dumps.STL10_C, space) if part == "all" else None
    )
    ood = {
        OOD_NAMES[name]: shard
        for name in OOD_NAMES
        if (shard := _load(locations, run_name, name, space)) is not None
    }
    log(f"{run_name} ({group}, seed {seed}, {space}): sets loaded")
    seeds = numpy.random.SeedSequence([cfg.seed, seed])
    s_clean, s_cell, s_id, s_ood, s_fold = (int(x) for x in seeds.generate_state(5))
    bank_x = bank.z.astype(numpy.float64)
    labels = bank.label
    num_classes = int(max(labels.max(), test.label.max())) + 1
    readers = confidence.Readers(bank_x, labels, cfg.eval.knn_k, num_classes)
    rho_bank = features.log_radius(bank_x)
    class_rho = numpy.array([rho_bank[labels == c].mean() for c in range(num_classes)])

    order = numpy.argsort(test.image_id, kind="stable")
    clean_x = test.z[order].astype(numpy.float64)
    clean_ids = test.image_id[order]
    clean_label = test.label[order]
    inputs: dict[str, tuple[FloatArray, numpy.typing.NDArray[numpy.int64]]] = {
        confidence.CLEAN: (clean_x, clean_label)
    }
    cell_rows: dict[shift.CellId, numpy.typing.NDArray[numpy.intp]] = {}
    if corrupted is not None:
        cell_rows = shift.cell_indices(corrupted)
        for cell, rows in cell_rows.items():
            inputs[f"cell:{cell[0]}:{cell[1]}"] = (
                corrupted.z[rows].astype(numpy.float64),
                corrupted.label[rows],
            )
    for name, shard in ood.items():
        inputs[f"ood:{name}"] = (shard.z.astype(numpy.float64), shard.label)
    keys = list(inputs)
    resid_pred = confidence.predict_log_radius(
        bank_x, [inputs[k][0] for k in keys], cfg.seed
    )
    log("direction -> log-radius map applied to every query set")
    queries: dict[str, confidence.Query] = {}
    for key, pred in zip(keys, resid_pred, strict=True):
        x, label = inputs[key]
        rho = features.log_radius(x)
        queries[key] = confidence.Query(rho, rho - pred, label, readers(x))
    log("readers applied to every query set")

    stats: dict[tuple[str, str], FloatArray] = {}
    if part == "ood":
        previous = confidence.load_unit(run_dir / unit_file(space))
        stats = {k: v for k, v in previous.stats.items() if not k[0].startswith("ood:")}
    booter = confidence.Booter(clean_x.shape[0], cfg.eval.num_bootstrap, s_clean)
    clean = queries[confidence.CLEAN]
    for metric, value in (
        confidence.set_statistics(clean, class_rho, booter, s_fold).items()
        if part == "all"
        else ()
    ):
        stats[confidence.CLEAN, metric] = value
    if cell_rows:
        cell_booter = confidence.Booter(
            clean_x.shape[0], confidence.CORRUPTED_BOOTSTRAP, s_cell
        )
        by_corruption: dict[str, list[dict[str, FloatArray]]] = {}
        for index, cell in enumerate(cell_rows):
            ids = corrupted.image_id[cell_rows[cell]] if corrupted else clean_ids
            twin = _subset(clean, numpy.searchsorted(clean_ids, ids))
            result = confidence.set_statistics(
                queries[f"cell:{cell[0]}:{cell[1]}"],
                class_rho,
                cell_booter,
                s_fold,
                twin,
            )
            by_corruption.setdefault(cell[0], []).append(result)
            for metric, value in result.items():
                if not metric.split("|")[-1].startswith("curve"):
                    stats[f"cell:{cell[0]}:{cell[1]}", metric] = value[:1]
            log(f"cell {index + 1}/{len(cell_rows)}: {cell[0]} {cell[1]}")
        for name, members in by_corruption.items():
            for metric, value in confidence.mean_groups(members).items():
                stats[f"corr:{name}", metric] = value
        everything = [r for members in by_corruption.values() for r in members]
        for metric, value in confidence.mean_groups(everything).items():
            stats[confidence.ALL_CORRUPTIONS, metric] = value
    for name in ood:
        id_booter = confidence.Booter(clean_x.shape[0], cfg.eval.num_bootstrap, s_id)
        ood_booter = confidence.Booter(
            len(queries[f"ood:{name}"].rho), cfg.eval.num_bootstrap, s_ood
        )
        for metric, value in confidence.ood_statistics(
            clean, queries[f"ood:{name}"], id_booter, ood_booter, s_fold
        ).items():
            stats[f"ood:{name}", metric] = value
        log(f"ood set {name} done")
    out = run_dir / unit_file(space)
    confidence.save_unit(out, confidence.Unit(run_name, group, seed, space, stats))
    return out


def load_units(
    locations: paths.Paths, prefix: str, space: str
) -> dict[str, list[confidence.Unit]]:
    """Units of the runs `prefix_*` (those that have one), by group."""
    out: dict[str, list[confidence.Unit]] = {}
    for directory in sorted(locations.runs_dir.glob(f"{prefix}_*")):
        file = directory / unit_file(space)
        if file.is_file():
            unit = confidence.load_unit(file)
            out.setdefault(unit.group, []).append(unit)
    return out


def _corruption_names(units: list[confidence.Unit]) -> list[str]:
    return sorted(
        {
            g.split(":", 1)[1]
            for u in units
            for (g, _m) in u.stats
            if g.startswith("corr:")
        }
    )


def corruption_rows(units: list[confidence.Unit]) -> list[Row]:
    """Per corruption (severities averaged): shift, error AUROC, radius additions."""
    rows: list[Row] = []
    for name in _corruption_names(units):
        group = f"corr:{name}"
        shift_rho = confidence.aggregate(units, group, "shift_rho")
        for reader in confidence.READERS:
            e = {
                m: confidence.aggregate(units, group, f"{reader}|{m}")
                for m in (
                    "accuracy",
                    "auroc[-rho]",
                    "auroc[-resid]",
                    "d_auroc[M1-M0]",
                    "d_auroc[M2-M0]",
                    "spearman[d_rho,d_p1]",
                )
            }
            rows.append(
                {
                    "corruption": name,
                    "reader": reader,
                    "mean shift of log-radius": round(shift_rho.value, 4),
                    "accuracy": round(e["accuracy"].value, 4),
                    "AUROC(-rho)": round(e["auroc[-rho]"].value, 4),
                    "verdict -rho": confidence.signal_verdict(e["auroc[-rho]"]),
                    "AUROC(-resid)": round(e["auroc[-resid]"].value, 4),
                    "M1 - M0": round(e["d_auroc[M1-M0]"].value, 4),
                    "verdict M1": confidence.difference_verdict(
                        e["d_auroc[M1-M0]"], confidence.AUROC_MARGIN
                    ),
                    "M2 - M0": round(e["d_auroc[M2-M0]"].value, 4),
                    "verdict M2": confidence.difference_verdict(
                        e["d_auroc[M2-M0]"], confidence.AUROC_MARGIN
                    ),
                    "Spearman(d_rho, d_p1)": round(e["spearman[d_rho,d_p1]"].value, 3),
                }
            )
    return rows


def severity_rows(units: list[confidence.Unit]) -> list[Row]:
    """Mean paired log-radius shift per corruption and severity (seed mean)."""
    rows: list[Row] = []
    for name in _corruption_names(units):
        row: Row = {"corruption": name}
        values: list[float] = []
        for severity in range(1, 6):
            e = confidence.aggregate(units, f"cell:{name}:{severity}", "shift_rho")
            row[f"severity {severity}"] = round(e.value, 4)
            values.append(e.value)
        row["non-increasing"] = str(
            all(b <= a + 1e-12 for a, b in zip(values, values[1:], strict=False))
        )
        rows.append(row)
    return rows


def type_rows(units: list[confidence.Unit], clean_accuracy: float) -> list[Row]:
    """Prediction-4 diagnostics: signs and the rank link between shift and loss."""
    from_shift: list[float] = []
    loss: list[float] = []
    for name in _corruption_names(units):
        group = f"corr:{name}"
        from_shift.append(confidence.aggregate(units, group, "shift_rho").value)
        loss.append(
            clean_accuracy
            - confidence.aggregate(units, group, f"{confidence.PROBE_Z}|accuracy").value
        )
    if len(from_shift) < 3:
        return []
    rank = _spearman(numpy.array(from_shift), numpy.array(loss))
    return [
        {
            "corruptions with negative mean shift": (
                f"{sum(v < 0 for v in from_shift)} of {len(from_shift)}"
            ),
            "Spearman(shift, accuracy loss of probe_z) across corruption types": round(
                rank, 3
            ),
        }
    ]


def _spearman(a: FloatArray, b: FloatArray) -> float:
    ra = numpy.argsort(numpy.argsort(a)).astype(numpy.float64)
    rb = numpy.argsort(numpy.argsort(b)).astype(numpy.float64)
    return float(numpy.corrcoef(ra, rb)[0, 1])


def cell_rows(units: list[confidence.Unit], reader: str) -> list[Row]:
    """Error AUROC of `-rho` per corruption and severity for `reader`."""
    rows: list[Row] = []
    for name in _corruption_names(units):
        row: Row = {"corruption": name}
        for severity in range(1, 6):
            e = confidence.aggregate(
                units, f"cell:{name}:{severity}", f"{reader}|auroc[-rho]"
            )
            row[f"severity {severity}"] = round(e.value, 3)
        rows.append(row)
    return rows


def _sections(units: list[confidence.Unit], group: str, space: str) -> dict[str, str]:
    cfg_note = (
        "Seed means; intervals are paired bootstrap percentile intervals over "
        "images (resampled within each seed). Definitions, margins and verdict "
        "rules: `research_notes/radius_confidence/confidence.md` "
        "(pre-registration)."
    )
    sections: dict[str, str] = {
        "analyze.confidence_inputs": (
            f"### Inputs\n\nRuns aggregated ({space} space, center views): "
            + ", ".join(f"`{u.name}`" for u in units)
            + "."
        ),
        "analyze.confidence_clean": (
            "### Clean STL-10 test: error detection by the radius signals\n\n"
            + cfg_note
            + "\n\n"
            + reporting.markdown_table(
                confidence.signal_rows(units, confidence.CLEAN, "clean test")
            )
            + "\n\n"
            + reporting.markdown_table(
                confidence.correlation_rows(units, confidence.CLEAN, "clean test")
            )
        ),
        "analyze.confidence_nested": (
            "### Beyond the reader's own confidence (clean test)\n\n"
            "`M0 = [logit p1, margin]`, `M1 = M0 + rho`, `M2 = M0 + resid`, "
            "5-fold cross-fitted; AUROC differences have margin "
            f"{confidence.AUROC_MARGIN:g}, accuracy-at-coverage differences "
            f"{100 * confidence.COVERAGE_MARGIN:g} points.\n\n"
            + reporting.markdown_table(
                confidence.nested_rows(units, confidence.CLEAN, "clean test")
            )
        ),
    }
    if any(g == confidence.ALL_CORRUPTIONS for (g, _m) in units[0].stats):
        label = "all 50 corrupted cells (mean over cells)"
        clean_acc = confidence.aggregate(
            units, confidence.CLEAN, f"{confidence.PROBE_Z}|accuracy"
        ).value
        sections |= {
            "analyze.confidence_corrupted": (
                "### STL-10-C: error detection and additions, all cells pooled\n\n"
                + cfg_note
                + f" ({confidence.CORRUPTED_BOOTSTRAP} replicates for the corrupted "
                "cells.)\n\n"
                + reporting.markdown_table(
                    confidence.signal_rows(units, confidence.ALL_CORRUPTIONS, label)
                )
                + "\n\n"
                + reporting.markdown_table(
                    confidence.correlation_rows(
                        units, confidence.ALL_CORRUPTIONS, label
                    )
                )
                + "\n\n"
                + reporting.markdown_table(
                    confidence.nested_rows(units, confidence.ALL_CORRUPTIONS, label)
                )
            ),
            "analyze.confidence_per_corruption": (
                "### Per corruption (severities averaged)\n\n"
                + reporting.markdown_table(corruption_rows(units))
            ),
            "analyze.confidence_shift": (
                "### Sign and monotonicity of the radius shift\n\n"
                "Mean paired change of the center-view log-radius against the clean "
                "twin of each image.\n\n"
                + reporting.markdown_table(severity_rows(units))
                + "\n\n"
                + reporting.markdown_table(type_rows(units, clean_acc))
            ),
            "analyze.confidence_cells": (
                "### AUROC of `-rho` for the errors of `probe_u`, per cell\n\n"
                + reporting.markdown_table(cell_rows(units, confidence.PROBE_U))
                + "\n\n### AUROC of `-rho` for the errors of `knn_u`, per cell\n\n"
                + reporting.markdown_table(cell_rows(units, confidence.KNN_U))
            ),
        }
    ood_groups = sorted(
        {g for u in units for (g, _m) in u.stats if g.startswith("ood:")}
    )
    if ood_groups:
        sections["analyze.confidence_ood"] = (
            "### ID (STL-10 test) vs OOD: the radius against the readers' "
            "confidence\n\nOOD is the positive class; `-rho`, `-p1` are oriented "
            "so that high = more OOD. SVHN is deliberately excluded: it is upsampled "
            "from 32 x 32 to 96 x 96, which confounds the low-level statistics "
            "(`research_notes/radius_usage/svhn_control.md`, branch "
            "`radius-usage-svhn-control`). DTD is native high resolution "
            "(downsampled); Places365 `val` is downsampled to 96 x 96.\n\n"
            + "\n\n".join(
                reporting.markdown_table(confidence.ood_rows(units, g))
                for g in ood_groups
            )
        )
    return sections


def _figures(
    units: list[confidence.Unit], group: str
) -> list[tuple[str, plotting.go.Figure, str]]:
    out: list[tuple[str, plotting.go.Figure, str]] = []
    curve: dict[str, plotting.Series] = {}
    for name in ("p1", "M0", "M2", "rho", "resid"):
        ys = [
            confidence.aggregate(
                units, confidence.CLEAN, f"{confidence.PROBE_U}|curve[{name}]@{c:g}"
            ).value
            for c in confidence.CURVE_COVERAGES
        ]
        curve[name] = plotting.Series(x=list(confidence.CURVE_COVERAGES), y=ys)
    out.append(
        (
            f"{group}_risk_coverage",
            plotting.line_plot(
                curve,
                f"Selective classification, clean test, probe_u ({group})",
                "coverage",
                "accuracy of the kept images",
            ),
            f"risk-coverage curves of probe_u on the clean test set, {group}",
        )
    )
    names = _corruption_names(units)
    if names:
        shifts = {
            name: plotting.Series(
                x=list(range(1, 6)),
                y=[
                    confidence.aggregate(units, f"cell:{name}:{s}", "shift_rho").value
                    for s in range(1, 6)
                ],
            )
            for name in names
        }
        out.append(
            (
                f"{group}_shift",
                plotting.line_plot(
                    shifts,
                    f"Mean paired log-radius shift by severity ({group})",
                    "severity",
                    "mean change of log-radius vs clean twin",
                    reference=0.0,
                ),
                f"log-radius shift per corruption and severity, {group}",
            )
        )
        aurocs = {
            name: plotting.Series(
                x=list(range(1, 6)),
                y=[
                    confidence.aggregate(
                        units, f"cell:{name}:{s}", f"{confidence.PROBE_U}|auroc[-rho]"
                    ).value
                    for s in range(1, 6)
                ],
            )
            for name in names
        }
        out.append(
            (
                f"{group}_auroc_by_severity",
                plotting.line_plot(
                    aurocs,
                    f"Error detection by -rho, probe_u ({group})",
                    "severity",
                    "AUROC",
                    reference=0.5,
                ),
                f"error-detection AUROC of the radius by severity, {group}",
            )
        )
    return out


def summary_rows(by_group: dict[str, list[confidence.Unit]]) -> list[Row]:
    """One row per group and direction-only reader: the headline numbers."""
    rows: list[Row] = []
    for group, units in sorted(by_group.items()):
        has_c = any(g == confidence.ALL_CORRUPTIONS for (g, _m) in units[0].stats)
        for reader in confidence.DIRECTION_READERS:
            row: Row = {"group": group, "seeds": len(units), "reader": reader}
            auroc = confidence.aggregate(
                units, confidence.CLEAN, f"{reader}|auroc[-rho]"
            )
            row["clean AUROC(-rho)"] = round(auroc.value, 4)
            row["clean verdict"] = confidence.signal_verdict(auroc)
            for model in ("M1", "M2", "M3"):
                d = confidence.aggregate(
                    units, confidence.CLEAN, f"{reader}|d_auroc[{model}-M0]"
                )
                row[f"clean {model}-M0"] = round(d.value, 4)
                row[f"clean {model} verdict"] = confidence.difference_verdict(
                    d, confidence.AUROC_MARGIN
                )
            if has_c:
                auroc = confidence.aggregate(
                    units, confidence.ALL_CORRUPTIONS, f"{reader}|auroc[-rho]"
                )
                row["corrupted AUROC(-rho)"] = round(auroc.value, 4)
                for model in ("M1", "M2", "M3"):
                    d = confidence.aggregate(
                        units,
                        confidence.ALL_CORRUPTIONS,
                        f"{reader}|d_auroc[{model}-M0]",
                    )
                    row[f"corrupted {model}-M0"] = round(d.value, 4)
                    row[f"corrupted {model} verdict"] = confidence.difference_verdict(
                        d, confidence.AUROC_MARGIN
                    )
            rows.append(row)
    return rows


def write_notes(
    locations: paths.Paths,
    prefix: str,
    space: str,
    note: pathlib.Path,
    figures: bool,
) -> list[pathlib.Path]:
    """Aggregates the units of runs `prefix_*` into one note per group.

    The notes are `<note stem>_<group>.md` next to `note` (relative to the notes
    directory); the summary table goes into `note` itself. Returns the paths.
    """
    by_group = load_units(locations, prefix, space)
    if not by_group:
        raise FileNotFoundError(f"no {unit_file(space)} found for runs {prefix}_*")
    base = locations.notes_dir / note
    written: list[pathlib.Path] = []
    for group, units in sorted(by_group.items()):
        path = base.with_name(f"{base.stem}_{space}_{group}{base.suffix}")
        for name, content in _sections(units, group, space).items():
            reporting.update_section(path, name, content)
        if figures:
            links: list[str] = []
            for stem, fig, alt in _figures(units, group):
                out = plotting.export_png(
                    fig,
                    path.parent / f"{base.stem}_{space}_{stem}.png",
                    locations.kaleido_chrome,
                )
                links.append(reporting.image_link(out, path, alt))
            reporting.update_section(
                path,
                "analyze.confidence_figures",
                "### Figures\n\n" + "\n\n".join(links),
            )
        written.append(path)
    reporting.update_section(
        base,
        f"analyze.confidence_summary_{space}",
        f"### Summary ({space} space, direction-only readers)\n\n"
        "M3 is a post hoc addition (only the direction-predicted part of the "
        "log-radius); M1, M2 are pre-registered.\n\n"
        + reporting.markdown_table(summary_rows(by_group)),
    )
    return written
