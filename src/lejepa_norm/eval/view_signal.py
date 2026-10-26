"""Is there a usable signal in how the radius moves under corruption (z only)?

Per run (`evaluate_unit`), on the clean STL-10 `test` images and their STL-10-C twins
(cells = corruption x severity), against the clean STL-10 `train` bank. Definitions
and predictions were written before any result: `research_notes/radius_view_signal/
preregistration.md`. Four parts:

* A, geometry: the displacement `D = z_x - z_c` of a corrupted center-view embedding
  from its clean twin splits into `a = <D, u_c>` (radial, along the clean direction)
  and `t = ||D - a u_c||` (tangential). Per image `a^2`, `t^2` and `log(r_x / r_c)`
  are kept; a null displacement from each clean image to a different clean image
  (fixed permutation) gives the radial share of a pure content change.
* B, oracle radius (readers fitted on the clean bank, frozen): `oracle radius`
  (`r_c u_x`), `oracle direction` (`r_x u_c`), `scaled constant` (bank-median radius,
  `u_x`) and the baseline `z_x`.
* C, severity: per corruption the 6 x N rows (clean twins = severity 0, severities
  1..5) carry the radius residual `res = log r - f(u)` (`f` the clean-fitted
  direction -> log-radius kernel ridge) and held-out severity predictions of ridge
  models on the direction, on `[res, |res|, log r]` and on both (5 folds grouped by
  image).
* D, aggregation: an image-level direction from the K augmented views by one of
  `AGGREGATIONS`, scaled to the bank-median radius and read by readers fitted on the
  bank directions at that scale.

Only z is used (LeJEPA's product is z; h is not analyzed). Pure numpy arrays in, a
`ViewSignalUnit` out; `view_signal_stats` aggregates units into comparisons.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import dataclasses
import os
import typing

import numpy
import numpy.typing
import sklearn.linear_model

from lejepa_norm.eval import features
from lejepa_norm.eval import randomization
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards
from lejepa_norm.eval import shift

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
CellId = shift.CellId
CLEAN = shift.CLEAN

READERS = shift.SHIFT_READERS

# Part B conditions.
BASELINE = randomization.BASELINE
ORACLE_RADIUS = "oracle radius (clean twin's radius)"
ORACLE_DIRECTION = "oracle direction (clean twin's direction)"
SCALED = "scaled constant (bank median radius)"
ORACLE_CONDITIONS = (BASELINE, ORACLE_RADIUS, ORACLE_DIRECTION, SCALED)

# Part D schemes.
AGG_MEAN = "mean of directions"
AGG_RADIUS = "radius-weighted"
AGG_INVERSE = "inverse-radius-weighted"
AGG_RESIDUAL = "residual-weighted"
AGG_TOP = "top-radius view"
AGG_BOTTOM = "bottom-radius view"
AGG_CENTER = "center view only (context)"
# Added after the first results (post hoc, not in the pre-registration): the views
# are i.i.d. augmentations, so view 1 is a random view; the reference for "is the
# top / bottom radius view better / worse than a random single view".
AGG_FIRST = "random single view (post hoc context)"
# Also post hoc: single views picked by the radius *residual* (the part of the radius
# that the direction does not predict), to separate "the radius ranks views" from
# "the direction-predictable radius ranks views".
AGG_TOP_RESIDUAL = "highest-residual view (post hoc)"
AGG_BOTTOM_RESIDUAL = "lowest-residual view (post hoc)"
AGGREGATIONS = (
    AGG_MEAN,
    AGG_RADIUS,
    AGG_INVERSE,
    AGG_RESIDUAL,
    AGG_TOP,
    AGG_BOTTOM,
    AGG_CENTER,
    AGG_FIRST,
    AGG_TOP_RESIDUAL,
    AGG_BOTTOM_RESIDUAL,
)

# Part C models.
MODEL_DIRECTION = "direction"
MODEL_RADIUS = "radius [res, |res|, log r]"
MODEL_BOTH = "direction + radius"
SEVERITY_MODELS = (MODEL_DIRECTION, MODEL_RADIUS, MODEL_BOTH)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SEVERITY_FOLDS = 5

# Augmented views of an image: views 1..K of the dumps (0 is the center view).
_FIRST_AUGMENTED = 1

Key = tuple[str, str, str, int]
SEP = "|"


@dataclasses.dataclass
class SeverityRows:
    """Rows of the severity task of one corruption (6 x N, severity-major).

    Attributes:
        severity: Severity 0..5 of every row.
        res: Radius residual `log r - f(u)`.
        rho: Log-radius.
        prediction: Held-out severity prediction per model of `SEVERITY_MODELS`.
    """

    severity: IntArray
    res: FloatArray
    rho: FloatArray
    prediction: dict[str, FloatArray]


@dataclasses.dataclass
class ViewSignalUnit:
    """All per-image outcomes of one run.

    Attributes:
        name: Run name.
        lam: SIGReg weight.
        seed: Training seed.
        image_id: Per cell, the `image_id` of every row (alignment check).
        geometry: Per corrupted cell, float `(5, N)` (`GEOM_*` rows).
        null_geometry: `(5, N)`: the same rows for the clean-to-other-image null.
        correct: Per (reader, condition, corruption, severity), the per-image
            correctness `(N,)` of parts B (`ORACLE_CONDITIONS`) and D
            (`AGGREGATIONS`); the clean test set is the cell `(CLEAN, 0)`.
        severity_rows: Part C rows per corruption.
        effective_views: Per aggregation, the mean effective number of views
            `1 / sum w^2` (weights normalized to sum 1), over images and cells.
        residual_sd: Standard deviation of the bank's held-out residual `s`.
    """

    name: str
    lam: float
    seed: int
    image_id: dict[CellId, IntArray]
    geometry: dict[CellId, FloatArray]
    null_geometry: FloatArray
    correct: dict[Key, FloatArray]
    severity_rows: dict[str, SeverityRows]
    effective_views: dict[str, float]
    residual_sd: float

    def cells(self) -> list[CellId]:
        """Corrupted cells (corruption, severity), sorted."""
        return sorted(c for c in self.image_id if c[0] != CLEAN)

    def corruptions(self) -> list[str]:
        """Corruption names present, sorted."""
        return sorted({c[0] for c in self.cells()})


def save_unit(path: str | os.PathLike[str], unit: ViewSignalUnit) -> None:
    """Writes a unit as an uncompressed `.npz` (atomically)."""
    arrays: dict[str, numpy.typing.NDArray[typing.Any]] = {
        "name": numpy.asarray(unit.name),
        "lam": numpy.asarray(unit.lam),
        "seed": numpy.asarray(unit.seed),
        "null": unit.null_geometry,
        "sd": numpy.asarray(unit.residual_sd),
    }
    for cell, ids in unit.image_id.items():
        arrays[SEP.join(["i", cell[0], str(cell[1])])] = ids
    for cell, g in unit.geometry.items():
        arrays[SEP.join(["g", cell[0], str(cell[1])])] = g.astype(numpy.float32)
    for (reader, condition, corruption, severity), v in unit.correct.items():
        arrays[SEP.join(["c", reader, condition, corruption, str(severity)])] = (
            v.astype(numpy.float32)
        )
    for corruption, rows in unit.severity_rows.items():
        arrays[SEP.join(["vs", corruption])] = rows.severity.astype(numpy.int8)
        arrays[SEP.join(["vr", corruption])] = rows.res.astype(numpy.float32)
        arrays[SEP.join(["vh", corruption])] = rows.rho.astype(numpy.float32)
        for model, pred in rows.prediction.items():
            arrays[SEP.join(["vp", corruption, model])] = pred.astype(numpy.float32)
    for scheme, value in unit.effective_views.items():
        arrays[SEP.join(["e", scheme])] = numpy.asarray(value)
    partial = str(path) + ".partial"
    with open(partial, "wb") as f:
        numpy.savez(f, **arrays)  # pyright: ignore[reportArgumentType]
    os.replace(partial, path)


def load_unit(path: str | os.PathLike[str]) -> ViewSignalUnit:
    """Reads a unit written by `save_unit`."""
    image_id: dict[CellId, IntArray] = {}
    geometry: dict[CellId, FloatArray] = {}
    correct: dict[Key, FloatArray] = {}
    sev: dict[str, IntArray] = {}
    res: dict[str, FloatArray] = {}
    rho: dict[str, FloatArray] = {}
    pred: dict[str, dict[str, FloatArray]] = {}
    effective: dict[str, float] = {}
    with numpy.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            p = name.split(SEP)
            if p[0] == "i":
                image_id[p[1], int(p[2])] = arrays[name]
            elif p[0] == "g":
                geometry[p[1], int(p[2])] = arrays[name].astype(numpy.float64)
            elif p[0] == "c":
                correct[p[1], p[2], p[3], int(p[4])] = arrays[name].astype(
                    numpy.float64
                )
            elif p[0] == "vs":
                sev[p[1]] = arrays[name].astype(numpy.int64)
            elif p[0] == "vr":
                res[p[1]] = arrays[name].astype(numpy.float64)
            elif p[0] == "vh":
                rho[p[1]] = arrays[name].astype(numpy.float64)
            elif p[0] == "vp":
                # The model name may contain the separator.
                pred.setdefault(p[1], {})[SEP.join(p[2:])] = arrays[name].astype(
                    numpy.float64
                )
            elif p[0] == "e":
                effective[p[1]] = float(arrays[name])
        return ViewSignalUnit(
            name=str(arrays["name"]),
            lam=float(arrays["lam"]),
            seed=int(arrays["seed"]),
            image_id=image_id,
            geometry=geometry,
            null_geometry=arrays["null"].astype(numpy.float64),
            correct=correct,
            severity_rows={
                c: SeverityRows(sev[c], res[c], rho[c], pred[c]) for c in sev
            },
            effective_views=effective,
            residual_sd=float(arrays["sd"]),
        )


# Rows of the per-image geometry arrays (`displacement_parts`).
GEOM_RADIAL, GEOM_TANGENTIAL, GEOM_LOG_RATIO, GEOM_RADIUS_CHANGE, GEOM_COSINE = range(5)


def displacement_parts(x: FloatArray, clean: FloatArray) -> FloatArray:
    """Radial and tangential split of `x - clean` along the clean direction.

    Returns:
        `(5, N)` rows (see `GEOM_*`): `a^2`, `t^2`, `log(||x|| / ||clean||)`, and,
        added post hoc, `(||x|| - ||clean||)^2` (the part of `||x - clean||^2` that
        a pure change of radius would explain; at most `a^2 + t^2`) and the cosine
        between the two directions.
    """
    d = x - clean
    u_clean = features.direction(clean)
    a = numpy.einsum("nd,nd->n", d, u_clean)
    total = numpy.einsum("nd,nd->n", d, d)
    return numpy.stack(
        [
            a * a,
            numpy.maximum(total - a * a, 0.0),
            features.log_radius(x) - features.log_radius(clean),
            numpy.square(features.radius(x) - features.radius(clean)),
            numpy.einsum("nd,nd->n", features.direction(x), u_clean),
        ]
    )


def _one_hot(index: IntArray, size: int) -> FloatArray:
    out = numpy.zeros((index.size, size))
    out[numpy.arange(index.size), index] = 1.0
    return out


def aggregation_weights(
    r: FloatArray, res: FloatArray, residual_sd: float
) -> dict[str, FloatArray]:
    """Per-view weights `(N, K)` of the weighted schemes (not normalized)."""
    top = numpy.zeros_like(r)
    top[numpy.arange(r.shape[0]), r.argmax(axis=1)] = 1.0
    bottom = numpy.zeros_like(r)
    bottom[numpy.arange(r.shape[0]), r.argmin(axis=1)] = 1.0
    return {
        AGG_MEAN: numpy.ones_like(r),
        AGG_RADIUS: r,
        AGG_INVERSE: 1.0 / r,
        AGG_RESIDUAL: numpy.exp(-numpy.abs(res) / residual_sd),
        AGG_TOP: top,
        AGG_BOTTOM: bottom,
        AGG_TOP_RESIDUAL: _one_hot(res.argmax(axis=1), r.shape[1]),
        AGG_BOTTOM_RESIDUAL: _one_hot(res.argmin(axis=1), r.shape[1]),
    }


def aggregate(u: FloatArray, weights: FloatArray) -> FloatArray:
    """Unit vector of the weighted sum of the view directions `u` `(N, K, d)`."""
    return features.direction(numpy.einsum("nk,nkd->nd", weights, u))


def effective_number(weights: FloatArray) -> FloatArray:
    """`1 / sum p^2` per image with `p` the weights normalized to sum 1."""
    p = weights / weights.sum(axis=1, keepdims=True)
    return 1.0 / numpy.square(p).sum(axis=1)


def severity_predictions(
    inputs: dict[str, FloatArray],
    severity: FloatArray,
    fold: IntArray,
) -> dict[str, FloatArray]:
    """Held-out ridge predictions of `severity` per model, by `fold` (grouped).

    Features are standardized with the training fold's mean and standard
    deviation; the penalty is chosen by `RidgeCV` (efficient leave-one-out) inside
    each training fold.
    """
    out: dict[str, FloatArray] = {}
    for model, x in inputs.items():
        pred = numpy.empty(severity.shape[0])
        for f in numpy.unique(fold):
            test = fold == f
            mean = x[~test].mean(axis=0)
            sd = numpy.maximum(x[~test].std(axis=0), 1e-12)
            ridge = sklearn.linear_model.RidgeCV(alphas=numpy.asarray(RIDGE_ALPHAS))
            ridge.fit((x[~test] - mean) / sd, severity[~test])
            pred[test] = ridge.predict((x[test] - mean) / sd)
        out[model] = pred
    return out


def _sorted_views(
    shard: shards.ViewShard, rows: numpy.typing.NDArray[numpy.intp]
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Center z, augmented-view z, labels and ids of `rows`, ordered by image id."""
    order = rows[numpy.argsort(shard.image_id[rows], kind="stable")]
    z = shard.z[order]
    return (
        z[:, 0].astype(numpy.float64),
        z[:, _FIRST_AUGMENTED:].astype(numpy.float64),
        shard.label[order],
        shard.image_id[order],
    )


def evaluate_unit(
    bank: FloatArray,
    bank_labels: numpy.typing.ArrayLike,
    clean: shards.ViewShard,
    corrupted: shards.ViewShard,
    k: int,
    cv_folds: int,
    seed: int,
    log: typing.Callable[[str], None] = lambda _msg: None,
) -> tuple[
    dict[CellId, IntArray],
    dict[CellId, FloatArray],
    FloatArray,
    dict[Key, FloatArray],
    dict[str, SeverityRows],
    dict[str, float],
    float,
]:
    """Computes parts A to D of one run; returns the fields of `ViewSignalUnit`.

    Args:
        bank: `(M, d)` clean labeled center-view embeddings (STL-10 `train`).
        bank_labels: `(M,)` labels.
        clean: STL-10 `test` views (the clean twins and the clean reference cell).
        corrupted: STL-10-C views; `corruption`, `severity` and `image_id` (index
            into `clean`) define the cells.
        k: Neighbours of the k-NN reader.
        cv_folds: Outer folds of the bank's out-of-fold radius regression.
        seed: Seed of every random draw.
        log: Progress callback.

    Returns:
        `(image_id, geometry, null_geometry, correct, severity_rows,
        effective_views, residual_sd)`.
    """
    by = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    num_classes = int(max(by.max(), corrupted.label.max(), clean.label.max())) + 1
    rho_bank = features.log_radius(bank)
    u_bank = features.direction(bank)
    rng = numpy.random.default_rng(seed)
    residual_sd = float(
        redundancy.NestedKernelRidge(u_bank, "exp_dot", cv_folds, rng)
        .fit(rho_bank)
        .residuals.std()
    )
    radius_map = redundancy.FittedKernelRidge(u_bank, rho_bank, rng)
    weight_decay = randomization.select_weight_decay(bank, by, num_classes, seed)
    median_radius = float(numpy.median(features.radius(bank)))
    scaled_bank = randomization.constant_radius(bank, median_radius)
    raw_fit: dict[str, randomization.Predictor] = {}
    direction_fit: dict[str, randomization.Predictor] = {}
    for reader in READERS:
        fit = randomization.make_fit(reader, by, k, num_classes, seed, weight_decay)
        raw_fit[reader] = fit(bank)
        direction_fit[reader] = fit(scaled_bank)
    log("readers fitted")

    # Cells: the clean test set first, then every corrupted cell.
    clean_x, clean_v, clean_y, clean_ids = _sorted_views(
        clean, numpy.arange(clean.image_id.size)
    )
    cell_rows = shift.cell_indices(corrupted)
    cells: list[CellId] = [(CLEAN, 0), *cell_rows]
    image_id: dict[CellId, IntArray] = {}
    geometry: dict[CellId, FloatArray] = {}
    correct: dict[Key, FloatArray] = {}
    res_cell: dict[CellId, FloatArray] = {}
    rho_cell: dict[CellId, FloatArray] = {}
    u_cell: dict[CellId, numpy.typing.NDArray[numpy.float32]] = {}
    effective: dict[str, list[float]] = {s: [] for s in AGGREGATIONS}

    perm = numpy.random.default_rng([seed, 7]).permutation(clean_x.shape[0])
    null_geometry = displacement_parts(clean_x[perm], clean_x)

    def store(
        reader: str,
        condition: str,
        cell: CellId,
        pred: IntArray,
        labels: IntArray,
    ) -> None:
        correct[reader, condition, cell[0], cell[1]] = (pred == labels).astype(
            numpy.float64
        )

    for index, cell in enumerate(cells):
        if cell == (CLEAN, 0):
            x, views, labels, ids = clean_x, clean_v, clean_y, clean_ids
        else:
            x, views, labels, ids = _sorted_views(corrupted, cell_rows[cell])
        image_id[cell] = ids
        twin = clean_x[numpy.searchsorted(clean_ids, ids)]
        rho = features.log_radius(x)
        u = features.direction(x)
        res_cell[cell] = rho - radius_map.predict(u)
        rho_cell[cell] = rho
        u_cell[cell] = u.astype(numpy.float32)
        # Parts B (readers on the raw center-view z).
        conditions: dict[str, FloatArray] = {BASELINE: x}
        if cell != (CLEAN, 0):
            geometry[cell] = displacement_parts(x, twin)
            conditions[ORACLE_RADIUS] = randomization.with_radius(
                x, features.radius(twin)
            )
            conditions[ORACLE_DIRECTION] = randomization.with_radius(
                twin, features.radius(x)
            )
            conditions[SCALED] = randomization.constant_radius(x, median_radius)
        for reader in READERS:
            for condition, query in conditions.items():
                store(reader, condition, cell, raw_fit[reader](query), labels)
        # Part D (readers on aggregated directions at the bank-median scale).
        n, num_views, d = views.shape
        r_v = features.radius(views)
        u_v = features.direction(views)
        res_v = features.log_radius(views) - radius_map.predict(
            u_v.reshape(-1, d)
        ).reshape(n, num_views)
        schemes = aggregation_weights(r_v, res_v, residual_sd)
        aggregated = {s: aggregate(u_v, w) for s, w in schemes.items()}
        aggregated[AGG_CENTER] = u
        aggregated[AGG_FIRST] = u_v[:, 0]
        for scheme, direction in aggregated.items():
            query = median_radius * direction
            for reader in READERS:
                store(reader, scheme, cell, direction_fit[reader](query), labels)
            weights = schemes.get(scheme)
            effective[scheme].append(
                float(effective_number(weights).mean()) if weights is not None else 1.0
            )
        log(f"cell {index + 1}/{len(cells)}: {cell[0] or 'clean'} {cell[1]}")

    severity_rows = _severity_rows(
        cells, image_id, u_cell, res_cell, rho_cell, seed, log
    )
    return (
        image_id,
        geometry,
        null_geometry,
        correct,
        severity_rows,
        {s: float(numpy.mean(v)) for s, v in effective.items()},
        residual_sd,
    )


def _severity_rows(
    cells: list[CellId],
    image_id: dict[CellId, IntArray],
    u_cell: dict[CellId, numpy.typing.NDArray[numpy.float32]],
    res_cell: dict[CellId, FloatArray],
    rho_cell: dict[CellId, FloatArray],
    seed: int,
    log: typing.Callable[[str], None],
) -> dict[str, SeverityRows]:
    """Part C: held-out severity predictions per corruption."""
    out: dict[str, SeverityRows] = {}
    n = image_id[CLEAN, 0].size
    fold_of_image = numpy.random.default_rng([seed, 11]).permutation(n) % SEVERITY_FOLDS
    for corruption in sorted({c[0] for c in cells if c[0] != CLEAN}):
        members = [(CLEAN, 0), *sorted(c for c in cells if c[0] == corruption)]
        severity = numpy.concatenate(
            [numpy.full(n, s, dtype=numpy.int64) for _, s in members]
        )
        res = numpy.concatenate([res_cell[c] for c in members])
        rho = numpy.concatenate([rho_cell[c] for c in members])
        u = numpy.concatenate([u_cell[c] for c in members]).astype(numpy.float64)
        radius_features = numpy.stack([res, numpy.abs(res), rho], axis=1)
        inputs = {
            MODEL_DIRECTION: u,
            MODEL_RADIUS: radius_features,
            MODEL_BOTH: numpy.concatenate([u, radius_features], axis=1),
        }
        prediction = severity_predictions(
            inputs,
            severity.astype(numpy.float64),
            numpy.tile(fold_of_image, len(members)),
        )
        out[corruption] = SeverityRows(severity, res, rho, prediction)
        log(f"severity models: {corruption}")
    return out
