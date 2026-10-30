"""Is the radius used under distribution shift? Corrupted STL-10-C test images.

The randomization study (`randomization`) asks whether classifiers on *clean*
STL-10 embeddings use the radius. Here the readers are the same (Euclidean k-NN,
linear probe, MLP; only z, center views) and the protocol is the same, but the
*test* images are corrupted (STL-10-C: the 10 vendored corruptions x 5 severities
of the STL-10 `test` split). The bank / training set is always the clean STL-10
`train` embeddings. Under corruption the radius shifts systematically (E6,
`radius_transfer`), so the question is whether that shifted radius hurts or helps
the readers, and whether removing it (normalizing the corrupted embedding)
recovers accuracy.

Conditions are those of `randomization.CONDITIONS`, applied to the corrupted test
embeddings of each (corruption, severity) cell, with the same two protocols:

* `frozen`: the reader is fitted on the clean, unmodified bank and applied to the
  modified corrupted test embeddings. This is the primary protocol: the reader
  never sees a modified embedding in training.
* `retrain`: the reader is fitted on the bank with the same *type* of
  modification (constant radius on both; permuted / residual-shuffled radius in
  the bank) and applied to the modified corrupted test embeddings. Only the bank
  changes, so each retrained reader is fitted once and reused for all 50 cells.

Radius permutations are within a cell (the 8000 corrupted images of one
corruption and severity). The direction -> log-radius map of the `residual`
condition is the clean-fitted map (out-of-fold on the bank, fitted on the whole
bank for the test embeddings); applied to corrupted directions it is exactly the
map whose transfer failure E6 documents, so the residual shuffle keeps the part
of the radius that the *clean* map predicts and permutes the rest (which, for
corrupted images, includes the systematic shift). The `frozen` `constant`
condition feeds unit-norm test embeddings to a reader fitted on raw-radius
features; for Euclidean k-NN this mixes the bank radii into the ranking, as in
the E1 frozen protocol.

A fourth treatment, `SCALED_CONSTANT` (every embedding rescaled to the median bank
radius), was added post hoc after the first results showed the unit-norm
`constant` condition collapsing frozen readers to chance (the scale mismatch E1
already documented on clean images, about a factor 16, not a statement about
information in the radius). The unit-norm results are kept; the scale-matched
variant is the one that answers "does normalizing recover accuracy".

Statistics: per-image correctness of every condition (averaged over shuffle
repeats) is compared with the baseline's on the same corrupted test images.
`compare` resamples test images within each training seed (the same resample for
every cell, reader and condition of that seed, so the severity averages are also
paired), averages the per-seed mean differences over seeds and reports 90% / 95%
percentile intervals with the TOST-style verdict of `randomization.verdict`. The
equivalence margin is `EQUIVALENCE_MARGIN` = 0.5 accuracy points, fixed before
looking at any result (same as E1). Seed variability beyond averaging three
seeds, and the dependence between the severities of one corruption (they share
the images), are not modeled by the intervals; severity averages are paired on
the images.
"""

import dataclasses
import os
import statistics
import typing

import numpy
import numpy.typing

from lejepa_norm import plotting
from lejepa_norm.eval import features
from lejepa_norm.eval import metrics
from lejepa_norm.eval import randomization
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
Row = dict[str, str | int | float]

# Fixed in advance (accuracy units, 0.005 = 0.5 points); same as E1.
EQUIVALENCE_MARGIN = 0.005

SHIFT_READERS = (
    randomization.KNN_EUCLIDEAN,
    randomization.LINEAR,
    randomization.MLP,
)
# Conditions compared with the baseline.
# Added after the first results (post hoc): the unit-norm `constant` condition feeds
# frozen readers features about 16x smaller than any they were fitted on (E1 found
# the same artifact on clean images), so it measures scale mismatch. This variant
# puts every embedding at the median bank radius instead (same direction, scale
# matched); for the retrained protocol it is equivalent to the unit-norm one up to
# the reader's scale sensitivity.
SCALED_CONSTANT = "constant radius (bank median)"
TREATMENTS = (*randomization.CONDITIONS[1:], SCALED_CONSTANT)

CLEAN = "clean"
OVERALL = "all corruptions"
# Rows at the corruption / overall level use severity 0 ("averaged over severities").
AVERAGED = 0

# Max test embeddings per call of the direction -> log-radius regression (each
# call refits the kernel ridge on the bank).
_RADIUS_BATCH = 90_000

CellId = tuple[str, int]
Key = tuple[str, str, str, str, int]
SEP = "|"


@dataclasses.dataclass
class ShiftUnit:
    """The per-image outcomes of one run.

    Attributes:
        name: Run name.
        lam: SIGReg weight.
        seed: Training seed.
        correct: Per (reader, protocol, condition, corruption, severity), the
            per-image correctness (fraction of shuffle repeats), float32 `(N,)`.
            The clean test set is the cell `(CLEAN, 0)`.
        image_id: Per cell, the `image_id` of every row (alignment check).
        radius_shift: Per corrupted cell, mean paired change of the log-radius
            versus the clean twin of each image.
        clean_map_residual: Per cell, mean of `log r - f(u)` with `f` the
            clean-fitted direction -> log-radius map.
    """

    name: str
    lam: float
    seed: int
    correct: dict[Key, FloatArray]
    image_id: dict[CellId, IntArray]
    radius_shift: dict[CellId, float]
    clean_map_residual: dict[CellId, float]

    def cells(self) -> list[CellId]:
        """Cells present, clean reference first, then corruptions and severities."""
        found = {(k[3], k[4]) for k in self.correct}
        return sorted(found, key=lambda c: (c[0] != CLEAN, c[0], c[1]))


def save_unit(path: str | os.PathLike[str], unit: ShiftUnit) -> None:
    """Writes a unit as an uncompressed `.npz` (atomically)."""
    arrays: dict[str, numpy.typing.NDArray[typing.Any]] = {
        "name": numpy.asarray(unit.name),
        "lam": numpy.asarray(unit.lam),
        "seed": numpy.asarray(unit.seed),
    }
    for key, correct in unit.correct.items():
        arrays["c" + SEP + SEP.join(str(p) for p in key)] = correct.astype(
            numpy.float32
        )
    for (corruption, severity), ids in unit.image_id.items():
        arrays[f"i{SEP}{corruption}{SEP}{severity}"] = ids
    for cell, value in unit.radius_shift.items():
        arrays[f"s{SEP}{cell[0]}{SEP}{cell[1]}"] = numpy.asarray(value)
    for cell, value in unit.clean_map_residual.items():
        arrays[f"r{SEP}{cell[0]}{SEP}{cell[1]}"] = numpy.asarray(value)
    partial = str(path) + ".partial"
    with open(partial, "wb") as f:
        numpy.savez(f, **arrays)  # pyright: ignore[reportArgumentType]
    os.replace(partial, path)


def load_unit(path: str | os.PathLike[str]) -> ShiftUnit:
    """Reads a unit written by `save_unit`."""
    correct: dict[Key, FloatArray] = {}
    image_id: dict[CellId, IntArray] = {}
    shift: dict[CellId, float] = {}
    residual: dict[CellId, float] = {}
    with numpy.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            parts = name.split(SEP)
            if parts[0] == "c":
                key = (parts[1], parts[2], parts[3], parts[4], int(parts[5]))
                correct[key] = arrays[name].astype(numpy.float64)
            elif parts[0] == "i":
                image_id[parts[1], int(parts[2])] = arrays[name]
            elif parts[0] == "s":
                shift[parts[1], int(parts[2])] = float(arrays[name])
            elif parts[0] == "r":
                residual[parts[1], int(parts[2])] = float(arrays[name])
        return ShiftUnit(
            name=str(arrays["name"]),
            lam=float(arrays["lam"]),
            seed=int(arrays["seed"]),
            correct=correct,
            image_id=image_id,
            radius_shift=shift,
            clean_map_residual=residual,
        )


def cell_indices(
    corrupted: shards.CenterShard | shards.ViewShard,
) -> dict[CellId, numpy.typing.NDArray[numpy.intp]]:
    """Row indices of each (corruption, severity) cell, ordered by `image_id`."""
    out: dict[CellId, numpy.typing.NDArray[numpy.intp]] = {}
    names = numpy.unique(corrupted.corruption)
    for corruption in names:
        in_corruption = corrupted.corruption == corruption
        for severity in numpy.unique(corrupted.severity[in_corruption]):
            rows = numpy.flatnonzero(in_corruption & (corrupted.severity == severity))
            out[str(corruption), int(severity)] = rows[
                numpy.argsort(corrupted.image_id[rows], kind="stable")
            ]
    return out


def evaluate_cells(
    bank: FloatArray,
    bank_labels: numpy.typing.ArrayLike,
    clean_test: shards.CenterShard,
    corrupted: shards.CenterShard,
    k: int,
    num_repeats: int,
    cv_folds: int,
    seed: int,
    readers: tuple[str, ...] = SHIFT_READERS,
    log: typing.Callable[[str], None] = lambda _msg: None,
) -> tuple[
    dict[Key, FloatArray],
    dict[CellId, IntArray],
    dict[CellId, float],
    dict[CellId, float],
]:
    """Runs every reader, protocol and condition on the clean and corrupted cells.

    Args:
        bank: `(M, d)` clean labeled embeddings (STL-10 `train`, center view).
        bank_labels: `(M,)` integer labels.
        clean_test: STL-10 `test` center z (the clean twins and a reference cell).
        corrupted: STL-10-C center z; `corruption`, `severity` and `image_id`
            (index into `clean_test`) define the cells.
        k: Neighbours of the k-NN reader.
        num_repeats: Shuffles per randomized condition.
        cv_folds: Outer folds of the bank's out-of-fold radius regression.
        seed: Seed of every random draw.
        readers: Subset of `SHIFT_READERS`.
        log: Progress callback.

    Returns:
        `(correct, image_id, radius_shift, clean_map_residual)` as in `ShiftUnit`.
    """
    by = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    num_classes = int(max(by.max(), corrupted.label.max(), clean_test.label.max())) + 1
    rho_bank = features.log_radius(bank)
    u_bank = features.direction(bank)
    oof = redundancy.NestedKernelRidge(
        u_bank, "exp_dot", cv_folds, numpy.random.default_rng(seed)
    ).fit(rho_bank)
    weight_decay = (
        randomization.select_weight_decay(bank, by, num_classes, seed)
        if randomization.MLP in readers
        else 0.0
    )
    # Cells: the clean test set first, then every corrupted cell.
    clean_order = numpy.argsort(clean_test.image_id, kind="stable")
    clean_sorted = clean_test.z[clean_order]
    clean_ids = clean_test.image_id[clean_order]
    cells: dict[CellId, tuple[FloatArray, IntArray, IntArray]] = {
        (CLEAN, 0): (
            clean_sorted.astype(numpy.float64),
            clean_test.label[clean_order],
            clean_ids,
        )
    }
    for cell, rows in cell_indices(corrupted).items():
        cells[cell] = (
            corrupted.z[rows].astype(numpy.float64),
            corrupted.label[rows],
            corrupted.image_id[rows],
        )
    # Held-out direction -> log-radius predictions of the clean-fitted map.
    predicted: dict[CellId, FloatArray] = {}
    pending: list[CellId] = []
    pending_rows = 0
    for cell in [*cells, None]:
        if cell is not None and pending_rows + len(cells[cell][1]) <= _RADIUS_BATCH:
            pending.append(cell)
            pending_rows += len(cells[cell][1])
            continue
        if pending:
            stacked = numpy.concatenate(
                [features.direction(cells[c][0]) for c in pending]
            )
            pred = redundancy.fit_predict(
                u_bank, rho_bank, stacked, rng=numpy.random.default_rng(seed)
            )
            offset = 0
            for c in pending:
                n = len(cells[c][1])
                predicted[c] = pred[offset : offset + n]
                offset += n
            log(f"radius map applied to {len(predicted)}/{len(cells)} cells")
        pending, pending_rows = ([cell], len(cells[cell][1])) if cell else ([], 0)

    # Readers: fitted once, on the bank (they do not depend on the test cell).
    transforms: dict[str, randomization.Transform] = {
        randomization.PERMUTE: lambda x, _p, rng: randomization.permute_radius(x, rng),
        randomization.RESIDUAL: randomization.residual_shuffle,
    }
    unit_bank = randomization.constant_radius(bank)
    median_radius = float(numpy.median(features.radius(bank)))
    scaled_bank = randomization.constant_radius(bank, median_radius)
    clean_fit: dict[str, randomization.Predictor] = {}
    constant_fit: dict[str, randomization.Predictor] = {}
    scaled_fit: dict[str, randomization.Predictor] = {}
    shuffled_fit: dict[tuple[str, str, int], randomization.Predictor] = {}
    for reader in readers:
        fit = randomization.make_fit(reader, by, k, num_classes, seed, weight_decay)
        clean_fit[reader] = fit(bank)
        constant_fit[reader] = fit(unit_bank)
        scaled_fit[reader] = fit(scaled_bank)
        for condition, transform in transforms.items():
            for repeat in range(num_repeats):
                rng_bank = numpy.random.default_rng([seed, repeat, 0])
                shuffled_fit[reader, condition, repeat] = fit(
                    transform(bank, oof.predictions, rng_bank)
                )
        log(f"readers fitted: {reader}")

    correct: dict[Key, FloatArray] = {}
    radius_shift: dict[CellId, float] = {}
    residual: dict[CellId, float] = {}
    for index, (cell, (x, qy, ids)) in enumerate(cells.items()):
        corruption, severity = cell
        rho = features.log_radius(x)
        residual[cell] = float(numpy.mean(rho - predicted[cell]))
        if corruption != CLEAN:
            twin = features.log_radius(
                clean_sorted[numpy.searchsorted(clean_ids, ids)].astype(numpy.float64)
            )
            radius_shift[cell] = float(numpy.mean(rho - twin))
        unit_x = randomization.constant_radius(x)
        scaled_x = randomization.constant_radius(x, median_radius)
        modified: dict[tuple[str, int], FloatArray] = {}
        for condition, transform in transforms.items():
            for repeat in range(num_repeats):
                rng_query = numpy.random.default_rng([seed, repeat, 1, index])
                modified[condition, repeat] = transform(x, predicted[cell], rng_query)

        def store(
            reader: str,
            protocol: str,
            condition: str,
            preds: list[IntArray],
            qy: IntArray = qy,
            corruption: str = corruption,
            severity: int = severity,
        ) -> None:
            hits = numpy.stack([p == qy for p in preds]).astype(numpy.float64)
            correct[reader, protocol, condition, corruption, severity] = hits.mean(0)

        for reader in readers:
            base = clean_fit[reader](x)
            for protocol in randomization.PROTOCOLS:
                store(reader, protocol, randomization.BASELINE, [base])
            store(
                reader,
                randomization.FROZEN,
                randomization.CONSTANT,
                [clean_fit[reader](unit_x)],
            )
            store(
                reader,
                randomization.RETRAIN,
                randomization.CONSTANT,
                [constant_fit[reader](unit_x)],
            )
            store(
                reader,
                randomization.FROZEN,
                SCALED_CONSTANT,
                [clean_fit[reader](scaled_x)],
            )
            store(
                reader,
                randomization.RETRAIN,
                SCALED_CONSTANT,
                [scaled_fit[reader](scaled_x)],
            )
            for condition in transforms:
                xs = [modified[condition, r] for r in range(num_repeats)]
                store(
                    reader,
                    randomization.FROZEN,
                    condition,
                    [clean_fit[reader](q) for q in xs],
                )
                store(
                    reader,
                    randomization.RETRAIN,
                    condition,
                    [shuffled_fit[reader, condition, r](q) for r, q in enumerate(xs)],
                )
        log(f"cell {index + 1}/{len(cells)}: {corruption} {severity}")
    return correct, {c: v[2] for c, v in cells.items()}, radius_shift, residual


@dataclasses.dataclass(frozen=True)
class ShiftComparison:
    """Paired accuracy change of one condition versus the baseline, one lambda.

    Attributes:
        lam: SIGReg weight.
        reader: Reader name.
        protocol: Protocol name.
        condition: The compared (non-baseline) condition.
        corruption: Corruption name, or `OVERALL` (mean over the corruptions).
        severity: 1..5, or `AVERAGED` (0) for the mean over severities.
        seeds: Training seeds averaged.
        baseline: Mean baseline accuracy over seeds (and averaged cells).
        accuracy: Mean accuracy of the condition.
        difference: `accuracy - baseline` in accuracy (0-1) units.
        ci90: 90% paired bootstrap interval of the difference.
        ci95: 95% paired bootstrap interval of the difference.
        verdict: `randomization.EQUIVALENT`, `DIFFERENT` or `INCONCLUSIVE`.
    """

    lam: float
    reader: str
    protocol: str
    condition: str
    corruption: str
    severity: int
    seeds: int
    baseline: float
    accuracy: float
    difference: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    verdict: str


def _resample_counts(
    num_images: int, num_bootstrap: int, rng: numpy.random.Generator
) -> FloatArray:
    """`(B, N)` multiplicities of `B` resamples of `N` images with replacement."""
    idx = rng.integers(0, num_images, size=(num_bootstrap, num_images))
    idx += numpy.arange(num_bootstrap)[:, None] * num_images
    counts = numpy.bincount(idx.ravel(), minlength=num_bootstrap * num_images)
    return counts.reshape(num_bootstrap, num_images).astype(numpy.float64)


def _check_aligned(group: list[ShiftUnit]) -> None:
    ref = group[0]
    for unit in group[1:]:
        if unit.image_id.keys() != ref.image_id.keys() or any(
            not numpy.array_equal(unit.image_id[c], ref.image_id[c])
            for c in ref.image_id
        ):
            raise ValueError(f"{unit.name} and {ref.name} have different test images")


def _readers(units: list[ShiftUnit]) -> tuple[str, ...]:
    """The readers of `SHIFT_READERS` that the units contain."""
    present = {key[0] for u in units for key in u.correct}
    return tuple(r for r in SHIFT_READERS if r in present)


def compare(
    units: list[ShiftUnit],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[ShiftComparison]:
    """Paired bootstrap differences vs. the baseline, per lambda (see module doc).

    Returns:
        Per lambda, reader, protocol and non-baseline condition: one comparison per
        corrupted cell (severity 1..5), one per corruption (`severity = AVERAGED`)
        and one over all corruptions (`corruption = OVERALL`).
    """
    out: list[ShiftComparison] = []
    for lam in sorted({u.lam for u in units}):
        group = sorted((u for u in units if u.lam == lam), key=lambda u: u.seed)
        _check_aligned(group)
        cells = [c for c in group[0].cells() if c[0] != CLEAN]
        combos = [
            (reader, protocol, condition)
            for reader in _readers(units)
            for protocol in randomization.PROTOCOLS
            for condition in TREATMENTS
        ]
        columns = [(*combo, *cell) for combo in combos for cell in cells]
        num_images = len(
            group[0].correct[columns[0][0], columns[0][1], columns[0][2], *cells[0]]
        )
        diff = numpy.empty((len(group), num_images, len(columns)))
        base_acc = numpy.empty((len(group), len(columns)))
        for s, unit in enumerate(group):
            for j, (reader, protocol, condition, corruption, severity) in enumerate(
                columns
            ):
                base = unit.correct[
                    reader, protocol, randomization.BASELINE, corruption, severity
                ]
                diff[s, :, j] = (
                    unit.correct[reader, protocol, condition, corruption, severity]
                    - base
                )
                base_acc[s, j] = base.mean()
        boot = numpy.zeros((num_bootstrap, len(columns)))
        for s in range(len(group)):
            counts = _resample_counts(num_images, num_bootstrap, rng)
            boot += counts @ diff[s] / num_images / len(group)
        point = diff.mean(axis=1).mean(axis=0)
        baseline = base_acc.mean(axis=0)

        # Weights of the compared quantities: each cell, each corruption (mean over
        # severities), each severity (mean over corruptions) and the grand mean.
        corruptions = sorted({c[0] for c in cells})
        severities = sorted({c[1] for c in cells})
        levels: list[tuple[str, str, str, str, int, FloatArray]] = []

        def weights(members: list[int], size: int = len(columns)) -> FloatArray:
            w = numpy.zeros(size)
            w[members] = 1.0 / len(members)
            return w

        for combo in combos:
            mine = [j for j, col in enumerate(columns) if col[:3] == combo]
            for j in mine:
                levels.append((*combo, columns[j][3], columns[j][4], weights([j])))
            for corruption in corruptions:
                members = [j for j in mine if columns[j][3] == corruption]
                levels.append((*combo, corruption, AVERAGED, weights(members)))
            for severity in severities:
                members = [j for j in mine if columns[j][4] == severity]
                levels.append((*combo, OVERALL, severity, weights(members)))
            levels.append((*combo, OVERALL, AVERAGED, weights(mine)))
        for reader, protocol, condition, corruption, severity, w in levels:
            draws = boot @ w
            lo90, hi90, lo95, hi95 = (
                float(v) for v in numpy.quantile(draws, [0.05, 0.95, 0.025, 0.975])
            )
            diff_mean = float(point @ w)
            base_mean = float(baseline @ w)
            out.append(
                ShiftComparison(
                    lam=lam,
                    reader=reader,
                    protocol=protocol,
                    condition=condition,
                    corruption=corruption,
                    severity=severity,
                    seeds=len(group),
                    baseline=base_mean,
                    accuracy=base_mean + diff_mean,
                    difference=diff_mean,
                    ci90=(lo90, hi90),
                    ci95=(lo95, hi95),
                    verdict=randomization.verdict((lo90, hi90), (lo95, hi95), margin),
                )
            )
    return out


def _interval(ci: tuple[float, float]) -> str:
    return f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"


def cell_rows(comparisons: list[ShiftComparison], lam: float) -> list[Row]:
    """Per (corruption, severity, reader, condition), both protocols side by side."""
    rows: list[Row] = []
    selected = [
        c
        for c in comparisons
        if c.lam == lam and c.severity != AVERAGED and c.corruption != OVERALL
    ]
    index = {
        (c.corruption, c.severity, c.reader, c.condition, c.protocol): c
        for c in selected
    }
    for corruption, severity, reader, condition, _ in list(index):
        if _ != randomization.FROZEN:
            continue
        frozen = index[corruption, severity, reader, condition, randomization.FROZEN]
        retrain = index[corruption, severity, reader, condition, randomization.RETRAIN]
        rows.append(
            {
                "corruption": corruption,
                "severity": severity,
                "reader": reader,
                "condition": condition,
                "baseline acc": f"{frozen.baseline:.4f}",
                "frozen: change (points)": f"{100 * frozen.difference:+.2f}",
                "frozen: 95% CI": _interval(frozen.ci95),
                "frozen: verdict": frozen.verdict,
                "retrain: change (points)": f"{100 * retrain.difference:+.2f}",
                "retrain: 95% CI": _interval(retrain.ci95),
                "retrain: verdict": retrain.verdict,
            }
        )
    return rows


def corruption_rows(comparisons: list[ShiftComparison], lam: float) -> list[Row]:
    """Per corruption (and overall), averaged over severities, both protocols."""
    rows: list[Row] = []
    for c in comparisons:
        if c.lam != lam or c.severity != AVERAGED:
            continue
        rows.append(
            {
                "corruption": c.corruption,
                "reader": c.reader,
                "protocol": c.protocol,
                "condition": c.condition,
                "baseline acc (corrupted)": f"{c.baseline:.4f}",
                "condition acc": f"{c.accuracy:.4f}",
                "change (points)": f"{100 * c.difference:+.2f}",
                "90% CI": _interval(c.ci90),
                "95% CI": _interval(c.ci95),
                "verdict": c.verdict,
            }
        )
    return rows


def _mean_accuracy(
    group: list[ShiftUnit],
    reader: str,
    cells: list[CellId],
    protocol: str = randomization.FROZEN,
    condition: str = randomization.BASELINE,
) -> float:
    """Accuracy averaged over the seeds of `group` and the given cells."""
    return statistics.fmean(
        float(u.correct[reader, protocol, condition, *c].mean())
        for u in group
        for c in cells
    )


def context_rows(
    units: list[ShiftUnit], comparisons: list[ShiftComparison]
) -> list[Row]:
    """Radius shift, clean-map residual and accuracies per lambda and corruption.

    Severities and seeds are averaged. The radius shift is the paired mean change
    of the center-view log-radius versus the clean twin; the residual is the mean
    of `log r - f(u)` with the clean-fitted map `f` (held out; 0 on average for
    clean images), the quantity E6 reports for the multi-view `rho_bar`. The
    accuracies are those of the frozen readers on the unmodified embeddings, and
    the last columns the frozen `constant` change.
    """
    rows: list[Row] = []
    for lam in sorted({u.lam for u in units}):
        group = [u for u in units if u.lam == lam]
        cells = [c for c in group[0].cells() if c[0] != CLEAN]
        clean = [(CLEAN, 0)]
        resid = statistics.fmean(u.clean_map_residual[CLEAN, 0] for u in group)
        rows.append(
            {
                "lambda": lam,
                "corruption": "clean test (reference)",
                "log-radius shift": "0",
                "clean-map residual": f"{resid:+.4f}",
                **{
                    f"{r} acc": f"{_mean_accuracy(group, r, clean):.4f}"
                    for r in _readers(units)
                },
            }
        )
        for corruption in sorted({c[0] for c in cells}):
            mine = [c for c in cells if c[0] == corruption]
            shift_mean = statistics.fmean(
                u.radius_shift[c] for u in group for c in mine
            )
            resid = statistics.fmean(
                u.clean_map_residual[c] for u in group for c in mine
            )
            row: Row = {
                "lambda": lam,
                "corruption": corruption,
                "log-radius shift": f"{shift_mean:+.4f}",
                "clean-map residual": f"{resid:+.4f}",
            }
            for reader in _readers(units):
                row[f"{reader} acc"] = f"{_mean_accuracy(group, reader, mine):.4f}"
            for reader in _readers(units):
                for condition, label in (
                    (randomization.CONSTANT, "unit norm"),
                    (SCALED_CONSTANT, "bank-median radius"),
                ):
                    const = next(
                        x
                        for x in comparisons
                        if x.lam == lam
                        and x.corruption == corruption
                        and x.severity == AVERAGED
                        and x.reader == reader
                        and x.protocol == randomization.FROZEN
                        and x.condition == condition
                    )
                    row[f"{reader}: frozen, {label} (points)"] = (
                        f"{100 * const.difference:+.2f}"
                    )
            rows.append(row)
    return rows


def shift_correlations(units: list[ShiftUnit]) -> list[Row]:
    """Spearman correlation of the radius shift with the accuracy change.

    Over the 50 cells, seed means, frozen protocol, per lambda, reader and
    condition. Descriptive only: the cells share images and corruption families,
    so there is no p-value.
    """
    rows: list[Row] = []
    for lam in sorted({u.lam for u in units}):
        group = [u for u in units if u.lam == lam]
        cells = [c for c in group[0].cells() if c[0] != CLEAN]
        radius_shift = numpy.array(
            [statistics.fmean(u.radius_shift[c] for u in group) for c in cells]
        )
        for reader in _readers(units):
            base = [_mean_accuracy(group, reader, [c]) for c in cells]
            for condition in TREATMENTS:
                change = numpy.array(
                    [
                        _mean_accuracy(group, reader, [c], condition=condition) - b
                        for c, b in zip(cells, base, strict=True)
                    ]
                )
                rho = metrics.spearman(radius_shift, change).value
                rows.append(
                    {
                        "lambda": lam,
                        "reader": reader,
                        "condition": condition,
                        "Spearman(radius shift, accuracy change)": f"{rho:+.3f}",
                    }
                )
    return rows


def figures(
    comparisons: list[ShiftComparison], lam: float
) -> list[tuple[str, plotting.go.Figure, str]]:
    """Figures for one lambda: `(file stem, figure, alt text)`.

    For each reader: accuracy change vs severity of the frozen `constant`
    condition per corruption (does normalizing a corrupted test embedding recover
    accuracy?), and the mean over corruptions per condition and protocol.
    """
    out: list[tuple[str, plotting.go.Figure, str]] = []
    corruptions = sorted(
        {c.corruption for c in comparisons if c.lam == lam and c.corruption != OVERALL}
    )
    for reader in [r for r in SHIFT_READERS if any(c.reader == r for c in comparisons)]:
        slug = "".join(ch if ch.isalnum() else "_" for ch in reader.split(" (")[0])
        per_corruption: dict[str, plotting.Series] = {}
        for corruption in corruptions:
            rows = sorted(
                (
                    c
                    for c in comparisons
                    if c.lam == lam
                    and c.reader == reader
                    and c.protocol == randomization.FROZEN
                    and c.condition == SCALED_CONSTANT
                    and c.corruption == corruption
                    and c.severity != AVERAGED
                ),
                key=lambda c: c.severity,
            )
            per_corruption[corruption] = plotting.Series(
                x=[c.severity for c in rows],
                y=[100 * c.difference for c in rows],
                low=[100 * c.ci95[0] for c in rows],
                high=[100 * c.ci95[1] for c in rows],
            )
        out.append(
            (
                f"lam{lam:g}_{slug}_constant_by_corruption",
                plotting.line_plot(
                    per_corruption,
                    f"Frozen {reader}, median-radius test (lambda {lam:g})",
                    "severity",
                    "accuracy change vs baseline (points)",
                    reference=0.0,
                ),
                f"constant-radius change per corruption, {reader}, lambda {lam:g}",
            )
        )
        by_condition: dict[str, plotting.Series] = {}
        for protocol in randomization.PROTOCOLS:
            for condition in TREATMENTS:
                rows = sorted(
                    (
                        c
                        for c in comparisons
                        if c.lam == lam
                        and c.reader == reader
                        and c.protocol == protocol
                        and c.condition == condition
                        and c.corruption == OVERALL
                        and c.severity != AVERAGED
                    ),
                    key=lambda c: c.severity,
                )
                by_condition[f"{condition} ({protocol})"] = plotting.Series(
                    x=[c.severity for c in rows],
                    y=[100 * c.difference for c in rows],
                    low=[100 * c.ci95[0] for c in rows],
                    high=[100 * c.ci95[1] for c in rows],
                )
        out.append(
            (
                f"lam{lam:g}_{slug}_all_conditions",
                plotting.line_plot(
                    by_condition,
                    f"{reader}, mean over corruptions (lambda {lam:g})",
                    "severity",
                    "accuracy change vs baseline (points)",
                    reference=0.0,
                ),
                f"accuracy change vs severity, {reader}, lambda {lam:g}",
            )
        )
    return out
