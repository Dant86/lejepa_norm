"""Aggregation of `view_signal` units: paired comparisons, tables and figures.

Per lambda, the three training seeds are combined as in the other usage studies:
test images are resampled with replacement independently within each seed (one
resample per seed shared by every comparison of that seed, so all comparisons are
paired), the per-seed statistic is averaged over seeds, and the 90% / 95% percentile
intervals of the replicates are reported. Accuracy comparisons use
`randomization.verdict` with the margin fixed in advance (0.5 points); the severity
comparison uses the margin `SEVERITY_MARGIN` (0.01 Spearman, fixed in advance, the
analogue of the 0.01 AUROC margin of `ood_radius`). Seed variability beyond three
seeds, and the dependence among the severities of one corruption, are not modeled.

Levels of the accuracy comparisons: one per corruption (mean over its 5 severities,
equal weights), `OVERALL` (mean over the 10 corruptions' 50 cells), one per severity
over all corruptions and, for part D, `CLEAN_TEST` (the uncorrupted images).
"""

import dataclasses
import statistics

import numpy
import numpy.typing

from lejepa_norm import plotting
from lejepa_norm.eval import metrics
from lejepa_norm.eval import randomization
from lejepa_norm.eval import shift
from lejepa_norm.eval import view_signal

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
Row = dict[str, str | int | float]
CellId = shift.CellId

OVERALL = "all corruptions"
CLEAN_TEST = "clean test (no corruption)"
# Fixed in advance (Spearman units).
SEVERITY_MARGIN = 0.01
# Part B: the loss must exceed this (accuracy units) for a recovery ratio.
MIN_LOSS = 0.01

ADDS = "adds"
DOES_NOT_ADD = "does not add"
WORSE = "worse"
INCONCLUSIVE = randomization.INCONCLUSIVE


@dataclasses.dataclass(frozen=True)
class Contrast:
    """`condition - reference` accuracy, paired by image.

    Attributes:
        name: Row label.
        condition: Condition (or scheme) compared.
        reference: Condition (or scheme) it is compared with.
        clean_twin: If True the reference is the *clean twin's* accuracy under the
            `reference` condition (cell `(CLEAN, 0)`), otherwise the same cell's.
    """

    name: str
    condition: str
    reference: str
    clean_twin: bool = False


ORACLE_CONTRASTS = (
    Contrast(
        "oracle radius - baseline", view_signal.ORACLE_RADIUS, view_signal.BASELINE
    ),
    Contrast("scaled constant - baseline", view_signal.SCALED, view_signal.BASELINE),
    Contrast(
        "oracle direction - clean twin",
        view_signal.ORACLE_DIRECTION,
        view_signal.BASELINE,
        clean_twin=True,
    ),
    Contrast(
        "baseline - clean twin (total corruption loss)",
        view_signal.BASELINE,
        view_signal.BASELINE,
        clean_twin=True,
    ),
)
AGGREGATION_CONTRASTS = (
    *(
        Contrast(f"{s} - mean", s, view_signal.AGG_MEAN)
        for s in view_signal.AGGREGATIONS
        if s != view_signal.AGG_MEAN
    ),
    Contrast(
        "top-radius view - bottom-radius view",
        view_signal.AGG_TOP,
        view_signal.AGG_BOTTOM,
    ),
    Contrast(
        "top-radius view - random single view (post hoc)",
        view_signal.AGG_TOP,
        view_signal.AGG_FIRST,
    ),
    Contrast(
        "bottom-radius view - random single view (post hoc)",
        view_signal.AGG_BOTTOM,
        view_signal.AGG_FIRST,
    ),
)


def _group(units: list[view_signal.ViewSignalUnit], lam: float):
    group = sorted((u for u in units if u.lam == lam), key=lambda u: u.seed)
    ref = group[0]
    for unit in group[1:]:
        if unit.image_id.keys() != ref.image_id.keys() or any(
            not numpy.array_equal(unit.image_id[c], ref.image_id[c])
            for c in ref.image_id
        ):
            raise ValueError(f"{unit.name} and {ref.name} have different test images")
    return group


def levels(unit: view_signal.ViewSignalUnit) -> dict[str, list[CellId]]:
    """Corrupted-cell levels: per corruption, per severity, overall."""
    cells = unit.cells()
    out: dict[str, list[CellId]] = {
        c: [x for x in cells if x[0] == c] for c in unit.corruptions()
    }
    for severity in sorted({c[1] for c in cells}):
        out[f"severity {severity} ({OVERALL})"] = [c for c in cells if c[1] == severity]
    out[OVERALL] = cells
    return out


def _value(
    unit: view_signal.ViewSignalUnit, reader: str, condition: str, cells: list[CellId]
) -> FloatArray:
    """Per-image correctness averaged over `cells`, `(N,)`."""
    return numpy.mean(
        [unit.correct[reader, condition, c[0], c[1]] for c in cells], axis=0
    )


def _interval(ci: tuple[float, float]) -> str:
    return f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"


@dataclasses.dataclass(frozen=True)
class AccuracyComparison:
    """One paired accuracy comparison (see `compare_accuracy`)."""

    lam: float
    reader: str
    level: str
    contrast: str
    condition_accuracy: float
    reference_accuracy: float
    difference: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    verdict: str


def compare_accuracy(
    units: list[view_signal.ViewSignalUnit],
    contrasts: tuple[Contrast, ...],
    include_clean_level: bool,
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[AccuracyComparison]:
    """Paired bootstrap differences of `contrasts`, per lambda, reader and level."""
    out: list[AccuracyComparison] = []
    for lam in sorted({u.lam for u in units}):
        group = _group(units, lam)
        num_images = len(next(iter(group[0].image_id.values())))
        draws = randomization.bootstrap_draws(
            num_images, len(group), num_bootstrap, rng
        )
        lv = levels(group[0])
        if include_clean_level:
            lv[CLEAN_TEST] = [(view_signal.CLEAN, 0)]
        for reader in view_signal.READERS:
            for level, cells in lv.items():
                for contrast in contrasts:
                    a = [_value(u, reader, contrast.condition, cells) for u in group]
                    ref_cells = (
                        [(view_signal.CLEAN, 0)] if contrast.clean_twin else cells
                    )
                    b = [
                        _value(u, reader, contrast.reference, ref_cells) for u in group
                    ]
                    diffs = [x - y for x, y in zip(a, b, strict=True)]
                    ci90, ci95 = randomization.intervals(
                        randomization.bootstrap_seed_mean(
                            draws, lambda s, idx, d=diffs: d[s][idx].mean(axis=1)
                        )
                    )
                    acc = statistics.fmean(float(x.mean()) for x in a)
                    ref = statistics.fmean(float(y.mean()) for y in b)
                    out.append(
                        AccuracyComparison(
                            lam,
                            reader,
                            level,
                            contrast.name,
                            acc,
                            ref,
                            acc - ref,
                            ci90,
                            ci95,
                            randomization.verdict(ci90, ci95, margin),
                        )
                    )
    return out


def accuracy_rows(comparisons: list[AccuracyComparison]) -> list[Row]:
    """Comparisons as table rows (differences in accuracy points)."""
    return [
        {
            "lambda": c.lam,
            "level": c.level,
            "reader": c.reader,
            "contrast": c.contrast,
            "accuracy": f"{c.condition_accuracy:.4f}",
            "reference accuracy": f"{c.reference_accuracy:.4f}",
            "difference (points)": f"{100 * c.difference:+.2f}",
            "90% CI": _interval(c.ci90),
            "95% CI": _interval(c.ci95),
            "verdict": c.verdict,
        }
        for c in comparisons
    ]


def recovery_rows(
    units: list[view_signal.ViewSignalUnit],
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Row]:
    """Share of the corruption loss that the oracle radius recovers.

    `(oracle radius - baseline) / (clean twin - baseline)` per lambda, reader and
    level (corruptions and overall), ratio of seed-mean paired differences, 95%
    interval from the same kind of image resamples; only where the mean loss
    exceeds `MIN_LOSS`.
    """
    rows: list[Row] = []
    clean = [(view_signal.CLEAN, 0)]
    for lam in sorted({u.lam for u in units}):
        group = _group(units, lam)
        num_images = len(next(iter(group[0].image_id.values())))
        draws = randomization.bootstrap_draws(
            num_images, len(group), num_bootstrap, rng
        )
        for reader in view_signal.READERS:
            for level, cells in levels(group[0]).items():
                if level.startswith("severity"):
                    continue
                gain = [
                    _value(u, reader, view_signal.ORACLE_RADIUS, cells)
                    - _value(u, reader, view_signal.BASELINE, cells)
                    for u in group
                ]
                loss = [
                    _value(u, reader, view_signal.BASELINE, clean)
                    - _value(u, reader, view_signal.BASELINE, cells)
                    for u in group
                ]
                mean_loss = statistics.fmean(float(x.mean()) for x in loss)
                if mean_loss <= MIN_LOSS:
                    continue
                num = randomization.bootstrap_seed_mean(
                    draws, lambda s, idx, g=gain: g[s][idx].mean(axis=1)
                )
                den = randomization.bootstrap_seed_mean(
                    draws, lambda s, idx, g=loss: g[s][idx].mean(axis=1)
                )
                _, (lo, hi) = randomization.intervals(num / den)
                point = statistics.fmean(float(x.mean()) for x in gain) / mean_loss
                rows.append(
                    {
                        "lambda": lam,
                        "level": level,
                        "reader": reader,
                        "corruption loss (points)": f"{100 * mean_loss:.2f}",
                        "recovered by oracle radius": f"{point:+.3f}",
                        "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                    }
                )
    return rows


def geometry_rows(
    units: list[view_signal.ViewSignalUnit],
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Row]:
    """Radial share of the clean -> corrupted displacement (part A).

    `F_rad = sum a^2 / sum (a^2 + t^2)` over images and the level's cells (ratio of
    sums), seed mean, 95% bootstrap interval over images; the median `r_x / r_c`;
    and the null row (clean image -> a different clean image).
    """
    rows: list[Row] = []
    for lam in sorted({u.lam for u in units}):
        group = _group(units, lam)
        num_images = len(next(iter(group[0].image_id.values())))
        draws = randomization.bootstrap_draws(
            num_images, len(group), num_bootstrap, rng
        )

        def share(
            radial: list[FloatArray],
            total: list[FloatArray],
            draws: list[IntArray] = draws,
        ) -> tuple[str, str]:
            boot = randomization.bootstrap_seed_mean(
                draws,
                lambda s, idx: radial[s][idx].sum(axis=1) / total[s][idx].sum(axis=1),
            )
            _, (lo, hi) = randomization.intervals(boot)
            point = statistics.fmean(
                float(r.sum() / t.sum()) for r, t in zip(radial, total, strict=True)
            )
            return f"{point:.4f}", f"[{lo:.4f}, {hi:.4f}]"

        lv = levels(group[0])
        for level, cells in lv.items():
            radial = [
                numpy.sum([u.geometry[c][0] for c in cells], axis=0) for u in group
            ]
            total = [
                numpy.sum([u.geometry[c][:2].sum(axis=0) for c in cells], axis=0)
                for u in group
            ]
            point, ci = share(radial, total)
            ratio = statistics.fmean(
                float(numpy.exp(numpy.median([u.geometry[c][2] for c in cells])))
                for u in group
            )
            rows.append(
                {
                    "lambda": lam,
                    "level": level,
                    "radial share F_rad": point,
                    "95% CI": ci,
                    "median r_x / r_c": f"{ratio:.3f}",
                }
            )
        point, ci = share(
            [u.null_geometry[0] for u in group],
            [u.null_geometry[0] + u.null_geometry[1] for u in group],
        )
        rows.append(
            {
                "lambda": lam,
                "level": "null: clean image -> other clean image",
                "radial share F_rad": point,
                "95% CI": ci,
                "median r_x / r_c": "",
            }
        )
    return rows


def geometry_cell_rows(units: list[view_signal.ViewSignalUnit]) -> list[Row]:
    """Radial share per cell, seed mean (point estimates only)."""
    rows: list[Row] = []
    for lam in sorted({u.lam for u in units}):
        group = _group(units, lam)
        for cell in group[0].cells():
            share = statistics.fmean(
                float(u.geometry[cell][0].sum() / (u.geometry[cell][:2].sum()))
                for u in group
            )
            ratio = statistics.fmean(
                float(numpy.exp(numpy.median(u.geometry[cell][2]))) for u in group
            )
            rows.append(
                {
                    "lambda": lam,
                    "corruption": cell[0],
                    "severity": cell[1],
                    "radial share F_rad": f"{share:.4f}",
                    "median r_x / r_c": f"{ratio:.3f}",
                }
            )
    return rows


def _rank(x: FloatArray) -> FloatArray:
    order = numpy.argsort(x, kind="stable")
    ranks = numpy.empty(x.size, dtype=numpy.float64)
    ranks[order] = numpy.arange(x.size, dtype=numpy.float64)
    return ranks


def _spearman_fixed(pred: FloatArray, severity_ranks: FloatArray) -> float:
    """Pearson correlation of `rank(pred)` with fixed severity ranks."""
    a = _rank(pred) - (pred.size - 1) / 2.0
    b = severity_ranks - severity_ranks.mean()
    return float(a @ b / numpy.sqrt((a @ a) * (b @ b)))


def severity_verdict(
    ci90: tuple[float, float], ci95: tuple[float, float], margin: float
) -> str:
    """`does not add` / `adds` / `worse` / `inconclusive` for a Spearman difference."""
    if ci90[0] > -margin and ci90[1] < margin:
        return DOES_NOT_ADD
    if ci95[0] > margin:
        return ADDS
    if ci95[1] < -margin:
        return WORSE
    return INCONCLUSIVE


SEVERITY_SIGNALS = (
    ("Spearman(res, severity), signed", "res"),
    ("held-out Spearman: direction", view_signal.MODEL_DIRECTION),
    ("held-out Spearman: radius", view_signal.MODEL_RADIUS),
    ("held-out Spearman: direction + radius", view_signal.MODEL_BOTH),
)
_DELTA = "delta"


def _signals(
    rows: view_signal.SeverityRows, index: IntArray, ranks: FloatArray
) -> dict[str, float]:
    """Spearman of each signal on the rows of the resampled images `index`."""
    values = {
        "res": rows.res[index],
        **{m: rows.prediction[m][index] for m in view_signal.SEVERITY_MODELS},
    }
    out = {name: _spearman_fixed(v, ranks) for name, v in values.items()}
    out[_DELTA] = out[view_signal.MODEL_BOTH] - out[view_signal.MODEL_DIRECTION]
    return out


def severity_rows(
    units: list[view_signal.ViewSignalUnit],
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Row]:
    """Part C: severity from the radius residual and from direction (+ radius).

    Per lambda and corruption: the Spearman correlation of the signed residual
    with severity, the held-out Spearman of the three models, and the paired
    difference `direction + radius - direction` with its verdict (margin
    `SEVERITY_MARGIN`). Seed means; images (all their 6 rows) are resampled.
    """
    rows: list[Row] = []
    for lam in sorted({u.lam for u in units}):
        group = _group(units, lam)
        num_images = len(next(iter(group[0].image_id.values())))
        draws = randomization.bootstrap_draws(
            num_images, len(group), num_bootstrap, rng
        )
        means: dict[str, list[float]] = {}
        for corruption in group[0].corruptions():
            members = [u.severity_rows[corruption] for u in group]
            ranks = metrics.rankdata(members[0].severity)
            offsets = numpy.arange(ranks.size // num_images) * num_images
            names = ["res", *view_signal.SEVERITY_MODELS, _DELTA]
            cache: dict[int, FloatArray] = {}

            def replicates(
                s: int,
                idx: IntArray,
                m: list[view_signal.SeverityRows] = members,
                r: FloatArray = ranks,
                off: IntArray = offsets,
                memo: dict[int, FloatArray] = cache,
                columns: list[str] = names,
            ) -> FloatArray:
                if s not in memo:
                    table: list[list[float]] = []
                    for b in idx:
                        sig = _signals(m[s], (b[None, :] + off[:, None]).ravel(), r)
                        table.append([sig[n] for n in columns])
                    memo[s] = numpy.asarray(table)
                return memo[s]

            point = {
                n: statistics.fmean(
                    _signals(m, numpy.arange(ranks.size), ranks)[n] for m in members
                )
                for n in names
            }
            ci: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
            for j, n in enumerate(names):
                ci[n] = randomization.intervals(
                    randomization.bootstrap_seed_mean(
                        draws, lambda s, idx, j=j: replicates(s, idx)[:, j]
                    )
                )
            row: Row = {"lambda": lam, "corruption": corruption}
            for label, which in SEVERITY_SIGNALS:
                _, c95 = ci[which]
                row[label] = f"{point[which]:+.3f} [{c95[0]:+.3f}, {c95[1]:+.3f}]"
            c90, c95 = ci[_DELTA]
            row["direction + radius - direction"] = (
                f"{point[_DELTA]:+.3f} [{c95[0]:+.3f}, {c95[1]:+.3f}]"
            )
            row["verdict"] = severity_verdict(c90, c95, SEVERITY_MARGIN)
            rows.append(row)
            for n in names:
                means.setdefault(n, []).append(point[n])
        mean_row: Row = {"lambda": lam, "corruption": "mean over corruptions"}
        for label, which in SEVERITY_SIGNALS:
            mean_row[label] = f"{statistics.fmean(means[which]):+.3f}"
        mean_row["direction + radius - direction"] = (
            f"{statistics.fmean(means[_DELTA]):+.3f}"
        )
        mean_row["verdict"] = ""
        rows.append(mean_row)
    return rows


def effective_view_rows(units: list[view_signal.ViewSignalUnit]) -> list[Row]:
    """Mean effective number of views of each scheme (power of the weighting)."""
    rows: list[Row] = []
    for lam in sorted({u.lam for u in units}):
        group = [u for u in units if u.lam == lam]
        row: Row = {"lambda": lam}
        for scheme in view_signal.AGGREGATIONS:
            row[scheme] = (
                f"{statistics.fmean(u.effective_views[scheme] for u in group):.2f}"
            )
        rows.append(row)
    return rows


def accuracy_summary_rows(units: list[view_signal.ViewSignalUnit]) -> list[Row]:
    """Mean accuracy per reader, condition / scheme and level (seed means)."""
    rows: list[Row] = []
    for lam in sorted({u.lam for u in units}):
        group = _group(units, lam)
        lv = {**levels(group[0]), CLEAN_TEST: [(view_signal.CLEAN, 0)]}
        for reader in view_signal.READERS:
            row: Row = {"lambda": lam, "reader": reader}
            for condition in (
                *view_signal.ORACLE_CONDITIONS,
                *view_signal.AGGREGATIONS,
            ):
                for level in (CLEAN_TEST, OVERALL):
                    key = (reader, condition, *lv[level][0])
                    if key not in group[0].correct:
                        continue
                    accuracy = statistics.fmean(
                        float(_value(u, reader, condition, lv[level]).mean())
                        for u in group
                    )
                    row[f"{condition}: {level}"] = f"{accuracy:.4f}"
            rows.append(row)
    return rows


def figures(
    geometry_cells: list[Row],
    comparisons: list[AccuracyComparison],
    lam: float,
) -> dict[str, plotting.go.Figure]:
    """Radial share vs severity per corruption; oracle-radius change vs severity."""
    series: dict[str, plotting.Series] = {}
    for corruption in sorted({str(r["corruption"]) for r in geometry_cells}):
        mine = [
            r
            for r in geometry_cells
            if r["corruption"] == corruption and r["lambda"] == lam
        ]
        series[corruption] = plotting.Series(
            [int(r["severity"]) for r in mine],
            [float(r["radial share F_rad"]) for r in mine],
        )
    out = {
        "radial_share": plotting.line_plot(
            series,
            f"Radial share of the clean-to-corrupted displacement (lambda={lam:g})",
            "severity",
            "F_rad",
        )
    }
    series2: dict[str, plotting.Series] = {}
    for reader in view_signal.READERS:
        mine = sorted(
            (
                c
                for c in comparisons
                if c.lam == lam
                and c.reader == reader
                and c.contrast == "oracle radius - baseline"
                and c.level.startswith("severity")
            ),
            key=lambda c: c.level,
        )
        series2[reader] = plotting.Series(
            list(range(1, len(mine) + 1)),
            [100 * c.difference for c in mine],
            [100 * c.ci95[0] for c in mine],
            [100 * c.ci95[1] for c in mine],
        )
    out["oracle_radius"] = plotting.line_plot(
        series2,
        f"Accuracy change from restoring the clean radius (lambda={lam:g})",
        "severity (all corruptions)",
        "oracle radius - baseline (points)",
        reference=0.0,
    )
    return out
