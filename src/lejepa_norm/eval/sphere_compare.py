"""E3: a model trained without the radius (`loss.radius_free`) vs the radius model.

The radius-free model is trained on `z~ = sqrt(d) z / ||z||`, so the radius of the
loss-visible embedding is constant and the raw radius of `z` is unconstrained
(meaningless). Every evaluation of a radius-free run therefore uses `z~`. The
comparison partner is the matching radius run (same lambda, the same seed index;
two *different* trained models, so the seed index only pairs the bookkeeping).
Only `z` is analyzed (LeJEPA's product is z).

Per-run units (`run_unit`, computed for the radius-free runs only, the radius runs'
units come from the existing studies):

* `sphere_cells.npz`: per-image correctness of the clean-bank readers (Euclidean
  k-NN, linear probe, MLP; fitted on STL-10 `train`, E1 protocol) on STL-10 `test`
  and every STL-10-C cell, in the `shift.ShiftUnit` layout.
* `sphere_retrieval/<run>.npz`: per-query class and instance retrieval values
  (`retrieval` layout; the baseline and the constant-radius conditions coincide).
* `shift_ood.npz`: the OOD score columns (`ood_radius` layout); only the
  direction-only score is used for a radius-free run (the other columns see the
  meaningless raw radius).

References from the radius runs: `shift_cells.npz` (baseline = the radius model
as trained; `constant` retrained readers = direction only, unit norm),
`<runs>/radius_usage_retrieval/<run>.npz` and `shift_ood.npz`.

Statistics: per-image differences of the two models on the *same* test images,
paired bootstrap over images within each seed pair (`randomization.bootstrap_*`),
averaged over seeds, TOST-style verdict (`randomization.verdict`) with the margins
of the existing studies. Image resampling does not cover the training-seed
variance of two different trained models beyond averaging the seeds. Non-
inferiority (`NON_INFERIOR`): the lower end of the 90% interval exceeds minus the
margin.
"""

import collections.abc
import dataclasses
import math
import pathlib
import statistics

import numpy
import numpy.typing
import torch

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import ood_radius
from lejepa_norm.eval import randomization
from lejepa_norm.eval import retrieval
from lejepa_norm.eval import shards
from lejepa_norm.eval import shift
from lejepa_norm.eval import sweep
from lejepa_norm.eval import usage_shift

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
Row = dict[str, str | int | float]

CELLS_FILE = "sphere_cells.npz"
RETRIEVAL_DIR = "sphere_retrieval"
# Where the radius runs' per-query retrieval shards live (`analyze --retrieval`).
RADIUS_RETRIEVAL_DIR = "radius_usage_retrieval"

READERS = shift.SHIFT_READERS
AS_TRAINED = "radius model as trained (raw z)"
DIRECTION_ONLY = "radius model, direction only (unit norm)"
REFERENCES = (AS_TRAINED, DIRECTION_ONLY)

CLEAN_LEVEL = "clean STL-10 test"
OVERALL_LEVEL = "STL-10-C, all 50 cells"

NON_INFERIOR = "non-inferior"
INFERIOR = "inferior"
UNDECIDED = "undecided"
# SVHN prediction of the pre-registration (see the experiment note).
LOWER = "lower"
DIRECTION_ONLY_LOWER = "direction of effect only"
NOT_LOWER = "not lower"
# "Best radius-using score" of the radius model for the OOD prediction.
DEPLOYABLE_RADIUS_SCORES = (ood_radius.KNN_RAW, ood_radius.COMBINED)
OOD_REFERENCES = (
    ood_radius.KNN_DIRECTION,
    ood_radius.KNN_RAW,
    ood_radius.COMBINED,
    ood_radius.ORACLE,
    ood_radius.POSTHOC_LOW,
)


def to_sphere(x: FloatArray) -> FloatArray:
    """`z~ = sqrt(d) x / ||x||` for every row (the loss-visible embedding)."""
    return math.sqrt(x.shape[-1]) * features.direction(x)


# --------------------------------------------------------------------------- units


def evaluate_clean_bank_cells(
    bank: FloatArray,
    bank_labels: numpy.typing.ArrayLike,
    clean_test: shards.CenterShard,
    corrupted: shards.CenterShard | None,
    k: int,
    seed: int,
    readers: tuple[str, ...] = READERS,
) -> tuple[dict[shift.Key, FloatArray], dict[shift.CellId, IntArray]]:
    """Per-image correctness of the clean-bank readers on every cell.

    Same readers, bank and weight-decay selection as `shift.evaluate_cells`'s
    baseline; the embeddings are used as given (apply `to_sphere` first for a
    radius-free run). The baseline is stored under both protocols (the clean test
    cell has identical frozen and retrained baselines).

    Args:
        bank: `(M, d)` labeled embeddings (STL-10 `train`).
        bank_labels: `(M,)` integer labels.
        clean_test: STL-10 `test` center z (cell `(CLEAN, 0)`).
        corrupted: STL-10-C center z, or None for the clean cell only.
        k: Neighbours of the k-NN reader.
        seed: Seed of the MLP and of its weight-decay selection.
        readers: Subset of `shift.SHIFT_READERS`.

    Returns:
        `(correct, image_id)` as in `shift.ShiftUnit` (cells sorted by `image_id`).
    """
    by = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    num_classes = int(max(by.max(), clean_test.label.max())) + 1
    if corrupted is not None:
        num_classes = max(num_classes, int(corrupted.label.max()) + 1)
    weight_decay = (
        randomization.select_weight_decay(bank, by, num_classes, seed)
        if randomization.MLP in readers
        else 0.0
    )
    order = numpy.argsort(clean_test.image_id, kind="stable")
    cells: dict[shift.CellId, tuple[FloatArray, IntArray, IntArray]] = {
        (shift.CLEAN, 0): (
            clean_test.z[order].astype(numpy.float64),
            clean_test.label[order],
            clean_test.image_id[order],
        )
    }
    if corrupted is not None:
        for cell, rows in shift.cell_indices(corrupted).items():
            cells[cell] = (
                corrupted.z[rows].astype(numpy.float64),
                corrupted.label[rows],
                corrupted.image_id[rows],
            )
    correct: dict[shift.Key, FloatArray] = {}
    for reader in readers:
        predict = randomization.make_fit(
            reader, by, k, num_classes, seed, weight_decay
        )(bank)
        for (corruption, severity), (x, labels, _ids) in cells.items():
            hits = (predict(x) == labels).astype(numpy.float64)
            for protocol in randomization.PROTOCOLS:
                key = (reader, protocol, randomization.BASELINE, corruption, severity)
                correct[key] = hits
    return correct, {cell: value[2] for cell, value in cells.items()}


def retrieval_values(
    train_z: FloatArray,
    train_labels: IntArray,
    test_z: FloatArray,
    test_labels: IntArray,
    device: torch.device | None = None,
    chunk: int = retrieval.DEFAULT_CHUNK,
) -> dict[retrieval.Key, FloatArray]:
    """Per-query class and instance retrieval of embeddings used as given.

    Class-level: queries = test center views, gallery = the other test center views
    or the train center views. Instance-level: the disjoint view pairs of the
    augmented test views, scores averaged per image (as `retrieval.evaluate`).
    The values are stored for the baseline and the constant-radius conditions
    alike (a radius-free embedding has no radius to remove).

    Args:
        train_z: `(M, K + 1, d)` train embeddings (center view first).
        train_labels: `(M,)` labels.
        test_z: `(N, K + 1, d)` test embeddings.
        test_labels: `(N,)` labels.
        device: Torch device.
        chunk: Queries per block.

    Returns:
        `(task, metric, condition)` to per-query values in 0-1 units.
    """
    ytr = numpy.asarray(train_labels, dtype=numpy.int64).ravel()
    yte = numpy.asarray(test_labels, dtype=numpy.int64).ravel()
    c_train = features.select_views(train_z, "center")[:, 0]
    c_test = features.select_views(test_z, "center")[:, 0]
    aug = features.select_views(test_z, "augmented")
    measured: dict[retrieval.Key, FloatArray] = {}
    for task, gallery, gallery_labels, exclude in (
        (retrieval.CLASS_TEST, c_test, yte, True),
        (retrieval.CLASS_TRAIN, c_train, ytr, False),
    ):
        result = retrieval.class_retrieval(
            c_test, yte, gallery, gallery_labels, exclude, chunk, device
        )
        for metric, values in result.items():
            measured[task, metric, retrieval.BASELINE] = values
    hits: dict[str, list[FloatArray]] = {m: [] for m in retrieval.INSTANCE_METRICS}
    for a, b in retrieval.view_pairs(aug.shape[1]):
        result = retrieval.instance_retrieval(aug[:, a], aug[:, b], chunk, device)
        for metric, values in result.items():
            hits[metric].append(values)
    for metric, parts in hits.items():
        measured[retrieval.INSTANCE, metric, retrieval.BASELINE] = numpy.mean(
            parts, axis=0
        )
    return {
        **measured,
        **{(t, m, retrieval.CONSTANT): v for (t, m, _c), v in measured.items()},
    }


def run_unit(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    run_name: str,
    log: collections.abc.Callable[[str], None],
    device: torch.device | None = None,
) -> list[pathlib.Path]:
    """Computes and saves the E3 unit files of one radius-free run.

    Raises:
        ValueError: the run was not trained with `loss.radius_free`.
        FileNotFoundError: a required embedding dump is missing.
    """
    run_dir = locations.runs_dir / run_name
    resolved = config.load_config(run_dir / "config.resolved.yaml")
    if not resolved.loss.radius_free:
        raise ValueError(f"{run_name} was not trained with loss.radius_free")

    def center(name: str) -> shards.CenterShard:
        loaded = dumps.load_center_set(
            dumps.set_dir(locations.embed_dir, run_name, name)
        )
        if loaded is None:
            raise FileNotFoundError(f"{run_name}: no {name} embedding dump")
        return loaded

    bank, test = center(dumps.STL10_TRAIN), center(dumps.STL10_TEST)
    corrupted = center(dumps.STL10_C)
    log(f"{run_name}: {corrupted.z.shape[0]} corrupted center embeddings loaded")

    def on_sphere(shard: shards.CenterShard) -> shards.CenterShard:
        return dataclasses.replace(shard, z=to_sphere(shard.z.astype(numpy.float64)))

    correct, image_id = evaluate_clean_bank_cells(
        to_sphere(bank.z.astype(numpy.float64)),
        bank.label,
        on_sphere(test),
        on_sphere(corrupted),
        cfg.eval.knn_k,
        cfg.seed,
    )
    out_cells = run_dir / CELLS_FILE
    shift.save_unit(
        out_cells,
        shift.ShiftUnit(
            run_name, resolved.loss.lambda_, resolved.seed, correct, image_id, {}, {}
        ),
    )
    log(f"wrote {out_cells}")
    train_set = dumps.load_set(
        dumps.set_dir(locations.embed_dir, run_name, "stl10_train")
    )
    test_set = dumps.load_set(
        dumps.set_dir(locations.embed_dir, run_name, "stl10_test")
    )
    if train_set is None or test_set is None:
        raise FileNotFoundError(f"{run_name}: needs stl10_train and stl10_test")
    values = retrieval_values(
        to_sphere(train_set.z.astype(numpy.float64)),
        train_set.label,
        to_sphere(test_set.z.astype(numpy.float64)),
        test_set.label,
        device,
    )
    out_ret = retrieval.save_result(
        run_dir / RETRIEVAL_DIR,
        retrieval.RunResult(
            run_name, resolved.loss.lambda_, resolved.seed, values, float("nan")
        ),
    )
    log(f"wrote {out_ret}")
    written = [out_cells, out_ret]
    written += usage_shift.run_unit(cfg, locations, run_name, "ood", log)
    return written


# ------------------------------------------------------------------- comparisons


class Resampler:
    """Image resamples per seed pair, shared by every comparison of one call."""

    def __init__(
        self, num_seeds: int, num_bootstrap: int, rng: numpy.random.Generator
    ) -> None:
        """Stores the sampling plan; draws are created lazily per image count."""
        self._num_seeds = num_seeds
        self._num_bootstrap = num_bootstrap
        self._rng = rng
        self._draws: dict[int, list[IntArray]] = {}

    def draws(self, num_images: int) -> list[IntArray]:
        """The `(B, num_images)` resample of every seed (same across calls)."""
        if num_images not in self._draws:
            self._draws[num_images] = randomization.bootstrap_draws(
                num_images, self._num_seeds, self._num_bootstrap, self._rng
            )
        return self._draws[num_images]


def paired_difference(
    diffs: collections.abc.Sequence[FloatArray], resampler: Resampler
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Seed-mean of per-image differences with its paired bootstrap intervals.

    Args:
        diffs: Per seed, the `(N,)` per-image differences (model A minus model B).
        resampler: Source of the image resamples.

    Returns:
        `(difference, ci90, ci95)`.
    """
    draws = resampler.draws(diffs[0].size)
    boot = randomization.bootstrap_seed_mean(
        draws, lambda s, idx: diffs[s][idx].mean(axis=1)
    )
    ci90, ci95 = randomization.intervals(boot)
    return statistics.fmean(float(d.mean()) for d in diffs), ci90, ci95


def non_inferiority(
    ci90: tuple[float, float], ci95: tuple[float, float], margin: float
) -> str:
    """`NON_INFERIOR` if the 90% lower bound exceeds `-margin`, see the docstring."""
    if ci90[0] > -margin:
        return NON_INFERIOR
    if ci95[1] < -margin:
        return INFERIOR
    return UNDECIDED


def pair_by_seed[T: shift.ShiftUnit | retrieval.RunResult | ood_radius.OodUnit](
    sphere: collections.abc.Sequence[T], radius: collections.abc.Sequence[T]
) -> list[tuple[T, T]]:
    """Pairs the units of the two groups by (lambda, seed index).

    Raises:
        ValueError: the two groups do not cover the same (lambda, seed) pairs.
    """
    a = {(u.lam, u.seed): u for u in sphere}
    b = {(u.lam, u.seed): u for u in radius}
    if a.keys() != b.keys() or len(a) != len(sphere) or len(b) != len(radius):
        raise ValueError(f"cannot pair seeds: {sorted(a)} vs {sorted(b)}")
    return [(a[key], b[key]) for key in sorted(a)]


@dataclasses.dataclass(frozen=True)
class AccuracyComparison:
    """Radius-free minus radius model accuracy at one level, reader and reference.

    Attributes:
        lam: SIGReg weight.
        level: `CLEAN_LEVEL`, `OVERALL_LEVEL` or a corruption name.
        reader: Reader name.
        reference: `AS_TRAINED` or `DIRECTION_ONLY`.
        seeds: Seed pairs averaged.
        sphere: Seed-mean accuracy of the radius-free model.
        other: Seed-mean accuracy of the reference.
        difference: `sphere - other` (0-1 units).
        ci90: 90% paired bootstrap interval of the difference.
        ci95: 95% interval.
        verdict: `randomization.verdict` at the margin.
    """

    lam: float
    level: str
    reader: str
    reference: str
    seeds: int
    sphere: float
    other: float
    difference: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    verdict: str


def _reference_key(
    reference: str, reader: str, corruption: str, severity: int
) -> shift.Key:
    if reference == AS_TRAINED:
        return (
            reader,
            randomization.FROZEN,
            randomization.BASELINE,
            corruption,
            severity,
        )
    return (reader, randomization.RETRAIN, randomization.CONSTANT, corruption, severity)


def compare_accuracy(
    pairs: list[tuple[shift.ShiftUnit, shift.ShiftUnit]],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[AccuracyComparison]:
    """Clean, pooled-corrupted and per-corruption accuracy differences, per lambda.

    Args:
        pairs: `(radius-free unit, radius unit)` pairs from `pair_by_seed`.
        margin: Equivalence margin (accuracy units).
        num_bootstrap: Bootstrap replicates.
        rng: Generator of the resamples.

    Returns:
        One comparison per lambda, level, reader and reference.

    Raises:
        ValueError: the two models of a pair were evaluated on different images.
    """
    out: list[AccuracyComparison] = []
    for lam in sorted({a.lam for a, _ in pairs}):
        group = [(a, b) for a, b in pairs if a.lam == lam]
        for a, b in group:
            for cell, ids in a.image_id.items():
                if not numpy.array_equal(ids, b.image_id[cell]):
                    raise ValueError(f"{a.name} and {b.name}: different images")
        resampler = Resampler(len(group), num_bootstrap, rng)
        cells = [c for c in group[0][0].cells() if c[0] != shift.CLEAN]
        levels: dict[str, list[shift.CellId]] = {CLEAN_LEVEL: [(shift.CLEAN, 0)]}
        if cells:
            levels[OVERALL_LEVEL] = cells
            for corruption in sorted({c[0] for c in cells}):
                levels[corruption] = [c for c in cells if c[0] == corruption]
        for level, members in levels.items():
            for reader in READERS:
                for reference in REFERENCES:
                    sphere_acc: list[FloatArray] = []
                    other_acc: list[FloatArray] = []
                    for a, b in group:
                        sphere_acc.append(
                            numpy.mean(
                                [
                                    a.correct[_reference_key(AS_TRAINED, reader, *cell)]
                                    for cell in members
                                ],
                                axis=0,
                            )
                        )
                        other_acc.append(
                            numpy.mean(
                                [
                                    b.correct[_reference_key(reference, reader, *cell)]
                                    for cell in members
                                ],
                                axis=0,
                            )
                        )
                    diff, ci90, ci95 = paired_difference(
                        [s - o for s, o in zip(sphere_acc, other_acc, strict=True)],
                        resampler,
                    )
                    out.append(
                        AccuracyComparison(
                            lam=lam,
                            level=level,
                            reader=reader,
                            reference=reference,
                            seeds=len(group),
                            sphere=statistics.fmean(
                                float(v.mean()) for v in sphere_acc
                            ),
                            other=statistics.fmean(float(v.mean()) for v in other_acc),
                            difference=diff,
                            ci90=ci90,
                            ci95=ci95,
                            verdict=randomization.verdict(ci90, ci95, margin),
                        )
                    )
    return out


def _interval(ci: tuple[float, float]) -> str:
    return f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"


def accuracy_rows(comparisons: list[AccuracyComparison]) -> list[Row]:
    """Accuracy comparisons as table rows (points = accuracy x 100)."""
    return [
        {
            "lambda": c.lam,
            "level": c.level,
            "reader": c.reader,
            "reference": c.reference,
            "radius-free acc": f"{c.sphere:.4f}",
            "reference acc": f"{c.other:.4f}",
            "difference (points)": f"{100 * c.difference:+.2f}",
            "90% CI (points)": _interval(c.ci90),
            "95% CI (points)": _interval(c.ci95),
            "verdict": c.verdict,
        }
        for c in comparisons
    ]


@dataclasses.dataclass(frozen=True)
class RetrievalComparison:
    """Radius-free minus radius model retrieval metric, one task and reference.

    Attributes:
        lam: SIGReg weight.
        task: Retrieval task (`retrieval.TASKS`).
        metric: Metric name.
        reference: `AS_TRAINED` or `DIRECTION_ONLY`.
        seeds: Seed pairs averaged.
        sphere: Seed-mean metric of the radius-free model.
        other: Seed-mean metric of the reference.
        difference: `sphere - other` (0-1 units).
        ci90: 90% paired bootstrap interval over queries.
        ci95: 95% interval.
        verdict: `randomization.verdict` at the margin.
        non_inferior: `non_inferiority` at the margin.
    """

    lam: float
    task: str
    metric: str
    reference: str
    seeds: int
    sphere: float
    other: float
    difference: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    verdict: str
    non_inferior: str


def compare_retrieval(
    pairs: list[tuple[retrieval.RunResult, retrieval.RunResult]],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[RetrievalComparison]:
    """Per-query retrieval differences per lambda, task, metric and reference."""
    out: list[RetrievalComparison] = []
    conditions = {AS_TRAINED: retrieval.BASELINE, DIRECTION_ONLY: retrieval.CONSTANT}
    for lam in sorted({a.lam for a, _ in pairs}):
        group = [(a, b) for a, b in pairs if a.lam == lam]
        resampler = Resampler(len(group), num_bootstrap, rng)
        for task in retrieval.TASKS:
            for metric in retrieval.task_metrics(task):
                for reference, condition in conditions.items():
                    own = [a.values[task, metric, retrieval.BASELINE] for a, _ in group]
                    other = [b.values[task, metric, condition] for _, b in group]
                    diff, ci90, ci95 = paired_difference(
                        [s - o for s, o in zip(own, other, strict=True)], resampler
                    )
                    out.append(
                        RetrievalComparison(
                            lam=lam,
                            task=task,
                            metric=metric,
                            reference=reference,
                            seeds=len(group),
                            sphere=statistics.fmean(float(v.mean()) for v in own),
                            other=statistics.fmean(float(v.mean()) for v in other),
                            difference=diff,
                            ci90=ci90,
                            ci95=ci95,
                            verdict=randomization.verdict(ci90, ci95, margin),
                            non_inferior=non_inferiority(ci90, ci95, margin),
                        )
                    )
    return out


def retrieval_rows(comparisons: list[RetrievalComparison]) -> list[Row]:
    """Retrieval comparisons as table rows (points = metric x 100)."""
    return [
        {
            "lambda": c.lam,
            "task": retrieval.TASK_TITLES[c.task],
            "metric": c.metric,
            "reference": c.reference,
            "radius-free (%)": f"{100 * c.sphere:.2f}",
            "reference (%)": f"{100 * c.other:.2f}",
            "difference (points)": f"{100 * c.difference:+.2f}",
            "90% CI (points)": _interval(c.ci90),
            "95% CI (points)": _interval(c.ci95),
            "verdict": c.verdict,
            "non-inferior": c.non_inferior,
        }
        for c in comparisons
    ]


def svhn_prediction(difference: FloatArray) -> str:
    """Pre-registered reading of (direction-only minus best radius-using) AUROC.

    `LOWER` iff the 95% interval lies entirely below 0; `DIRECTION_ONLY_LOWER` if
    only the point difference is negative; otherwise `NOT_LOWER`.
    """
    lo, hi = numpy.quantile(difference[1:], [0.025, 0.975])
    del lo
    if float(hi) < 0:
        return LOWER
    return DIRECTION_ONLY_LOWER if float(difference[0]) < 0 else NOT_LOWER


@dataclasses.dataclass(frozen=True)
class OodComparison:
    """OOD AUROC of the radius-free direction score vs the radius model's scores.

    Attributes:
        lam: SIGReg weight.
        ood_set: OOD set name.
        sphere: The radius-free `KNN_DIRECTION` result arrays `(1 + B,)`.
        radius: The radius model's AUROC arrays per score in `OOD_REFERENCES`.
        sphere_fpr: Radius-free FPR@95TPR arrays.
        radius_fpr: The radius model's FPR@95TPR arrays.
        best_radius_score: The larger seed-mean deployable radius score.
    """

    lam: float
    ood_set: str
    sphere: FloatArray
    radius: dict[str, FloatArray]
    sphere_fpr: FloatArray
    radius_fpr: dict[str, FloatArray]
    best_radius_score: str


def compare_ood(
    pairs: list[tuple[ood_radius.OodUnit, ood_radius.OodUnit]],
    num_bootstrap: int,
    seed: int,
) -> list[OodComparison]:
    """OOD AUROC / FPR of both groups on identical bootstrap resamples.

    `ood_radius.evaluate` is run on each group with a generator seeded the same
    way; the resamples depend only on the image counts, the number of units and
    the replicate count, so the two groups are resampled identically (checked: the
    image counts must match) and the replicate-wise differences are paired.

    Args:
        pairs: `(radius-free unit, radius unit)` pairs from `pair_by_seed`.
        num_bootstrap: Bootstrap replicates.
        seed: Seed of the (shared) resampling generator.

    Raises:
        ValueError: the paired units have different numbers of ID images.
    """
    for a, b in pairs:
        if a.num_id != b.num_id:
            raise ValueError(f"{a.name} and {b.name}: different ID images")
    sphere_results = ood_radius.evaluate(
        [a for a, _ in pairs], num_bootstrap, numpy.random.default_rng(seed)
    )
    radius_results = ood_radius.evaluate(
        [b for _, b in pairs], num_bootstrap, numpy.random.default_rng(seed)
    )
    out: list[OodComparison] = []
    for s, r in zip(sphere_results, radius_results, strict=True):
        if (s.lam, s.ood_set, s.num_ood) != (r.lam, r.ood_set, r.num_ood):
            raise ValueError("the groups differ in lambda or OOD set sizes")
        best = max(DEPLOYABLE_RADIUS_SCORES, key=lambda name: float(r.auroc[name][0]))
        out.append(
            OodComparison(
                lam=s.lam,
                ood_set=s.ood_set,
                sphere=s.auroc[ood_radius.KNN_DIRECTION],
                radius={name: r.auroc[name] for name in OOD_REFERENCES},
                sphere_fpr=s.fpr[ood_radius.KNN_DIRECTION],
                radius_fpr={name: r.fpr[name] for name in OOD_REFERENCES},
                best_radius_score=best,
            )
        )
    return out


def _fmt_ci(values: FloatArray, digits: int = 3) -> str:
    lo, hi = numpy.quantile(values[1:], [0.025, 0.975])
    return f"{values[0]:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def ood_score_rows(comparisons: list[OodComparison]) -> list[Row]:
    """AUROC and FPR@95TPR of the radius-free direction score and the references."""
    rows: list[Row] = []
    for c in comparisons:
        rows.append(
            {
                "lambda": c.lam,
                "OOD set": c.ood_set,
                "model": "radius-free",
                "score": ood_radius.KNN_DIRECTION,
                "AUROC [95% CI]": _fmt_ci(c.sphere),
                "FPR@95TPR [95% CI]": _fmt_ci(c.sphere_fpr),
            }
        )
        for name in OOD_REFERENCES:
            rows.append(
                {
                    "lambda": c.lam,
                    "OOD set": c.ood_set,
                    "model": "radius",
                    "score": name,
                    "AUROC [95% CI]": _fmt_ci(c.radius[name]),
                    "FPR@95TPR [95% CI]": _fmt_ci(c.radius_fpr[name]),
                }
            )
    return rows


def ood_difference_rows(comparisons: list[OodComparison]) -> list[Row]:
    """Radius-free direction score minus each radius-model score (paired)."""
    rows: list[Row] = []
    for c in comparisons:
        for name in OOD_REFERENCES:
            diff = c.sphere - c.radius[name]
            lo90, hi90 = numpy.quantile(diff[1:], [0.05, 0.95])
            rows.append(
                {
                    "lambda": c.lam,
                    "OOD set": c.ood_set,
                    "radius-model score": name
                    + (" (best deployable)" if name == c.best_radius_score else ""),
                    "difference [95% CI]": _fmt_ci(diff),
                    "90% CI": f"[{lo90:+.3f}, {hi90:+.3f}]",
                    "verdict (margin 0.01)": ood_radius.addition_verdict(diff),
                    "prediction 3 reading": (
                        svhn_prediction(diff) if name == c.best_radius_score else ""
                    ),
                }
            )
    return rows


# ----------------------------------------------------------------- training health

HEALTH_KEYS = (
    ("loss_inv_epoch_mean", "invariance loss"),
    ("loss_dist_epoch_mean", "SIGReg loss"),
    ("knn_acc_h", "k-NN acc h (monitor)"),
    ("knn_acc_z", "k-NN acc z (monitor, cosine)"),
    ("kappa_z_epoch_mean", "kappa_z of raw z"),
    ("z_norm_mean_epoch_mean", "mean ||z|| of raw z"),
    ("z_norm_std_epoch_mean", "sd ||z|| of raw z"),
)
HEALTH_EPOCHS = (10, 50, 100, 200)
# Descriptive health bar (documented in the note): final monitor accuracy within
# this many accuracy units of the radius runs'.
HEALTH_KNN_TOLERANCE = 0.02


def _final_knn(runs: list[sweep.Run]) -> float:
    """Mean over runs of the last logged `knn_acc_h` (NaN if none was logged)."""
    last: list[float] = []
    for run in runs:
        logged = [e for e, values in run.epochs.items() if "knn_acc_h" in values]
        if logged:
            last.append(run.epochs[max(logged)]["knn_acc_h"])
    return statistics.fmean(last) if last else float("nan")


def health_rows(
    groups: dict[str, list[sweep.Run]], epochs: tuple[int, ...] = HEALTH_EPOCHS
) -> list[Row]:
    """Mean +- sd over seeds of the logged curves at selected epochs, per group."""
    rows: list[Row] = []
    for name, runs in groups.items():
        for key, title in HEALTH_KEYS:
            row: Row = {"group": name, "quantity": title}
            for epoch in epochs:
                values = [
                    r.epochs[epoch][key] for r in runs if key in r.epochs.get(epoch, {})
                ]
                if values and len(values) == len(runs):
                    sd = statistics.stdev(values) if len(values) > 1 else 0.0
                    row[f"epoch {epoch}"] = f"{statistics.fmean(values):.4g} ± {sd:.2g}"
                else:
                    row[f"epoch {epoch}"] = ""
            rows.append(row)
    return rows


def health_summary(groups: dict[str, list[sweep.Run]]) -> list[Row]:
    """Completed runs, finite metrics and final monitor accuracy, per group."""
    return [
        {
            "group": name,
            "runs": len(runs),
            "all complete": all(r.complete for r in runs),
            "all finite": all(
                math.isfinite(v)
                for r in runs
                for values in r.epochs.values()
                for v in values.values()
            ),
            "final knn_acc_h (mean)": _final_knn(runs),
        }
        for name, runs in groups.items()
    ]


def is_healthy(radius_free: list[sweep.Run], radius: list[sweep.Run]) -> bool:
    """The descriptive health bar of the note (see `HEALTH_KNN_TOLERANCE`)."""
    gap = abs(_final_knn(radius_free) - _final_knn(radius))
    return (
        bool(radius_free)
        and all(r.complete for r in radius_free)
        and all(
            math.isfinite(v)
            for r in radius_free
            for values in r.epochs.values()
            for v in values.values()
        )
        and gap <= HEALTH_KNN_TOLERANCE
    )


# ------------------------------------------------------------------------- note


def _curve_figure(
    groups: dict[str, list[sweep.Run]], key: str, title: str
) -> plotting.go.Figure:
    series: dict[str, plotting.Series] = {}
    for name, runs in groups.items():
        c = sweep.curve(runs, key)
        if c.epochs:
            series[name] = plotting.Series(c.epochs, c.mean, c.low, c.high)
    return plotting.line_plot(series, title, "epoch", title)


def figures(
    groups: dict[str, list[sweep.Run]],
) -> list[tuple[str, plotting.go.Figure, str]]:
    """`(file stem, figure, alt text)` of the training-curve comparisons."""
    return [
        (
            f"curve_{key}",
            _curve_figure(groups, key, title),
            f"{title}: radius-free vs radius runs (mean, min-max over seeds)",
        )
        for key, title in HEALTH_KEYS
    ]


def readings(
    accuracy: list[AccuracyComparison],
    retrievals: list[RetrievalComparison],
    ood: list[OodComparison],
    healthy: bool,
) -> str:
    """Generated one-line readings of the three pre-registered predictions."""
    lines: list[str] = []
    for lam in sorted({c.lam for c in accuracy}):
        clean = [
            c
            for c in accuracy
            if c.lam == lam and c.level == CLEAN_LEVEL and c.reference == AS_TRAINED
        ]
        held = all(c.verdict == randomization.EQUIVALENT for c in clean)
        detail = "; ".join(
            f"{c.reader}: {100 * c.difference:+.2f} pts, {c.verdict}" for c in clean
        )
        lines.append(
            f"- lambda {lam:g}, prediction 1 (accuracy equivalent): "
            f"**{'held' if held else 'did not hold'}** ({detail})."
        )
        inst = [
            c
            for c in retrievals
            if c.lam == lam
            and c.task == retrieval.INSTANCE
            and c.reference == DIRECTION_ONLY
        ]
        held = all(c.non_inferior == NON_INFERIOR for c in inst)
        detail = "; ".join(
            f"{c.metric}: {100 * c.difference:+.2f} pts, 90% CI {_interval(c.ci90)}, "
            f"{c.non_inferior}"
            for c in inst
        )
        lines.append(
            f"- lambda {lam:g}, prediction 2 (instance retrieval recovers through "
            f"direction): **{'held' if held else 'did not hold'}** ({detail})."
        )
        for c in ood:
            if c.lam != lam or c.ood_set != dumps.SVHN_TEST:
                continue
            diff = c.sphere - c.radius[c.best_radius_score]
            lines.append(
                f"- lambda {lam:g}, prediction 3 (SVHN direction-only AUROC lower "
                f"than the radius model's best radius-using score, here "
                f"`{c.best_radius_score}`): **{svhn_prediction(diff)}** "
                f"(difference {_fmt_ci(diff)})."
            )
    lines.append(
        f"- Training healthy (descriptive bar): **{'yes' if healthy else 'no'}**."
    )
    return "\n".join(lines)


def write_note(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    prefix: str,
    baseline_prefix: str,
    note: pathlib.Path,
    make_figures: bool,
) -> pathlib.Path:
    """Aggregates the units of runs `prefix_*` vs `baseline_prefix_*` into `note`.

    Raises:
        ValueError: the margin differs from the pre-registered one.
        FileNotFoundError: a unit file of a run is missing.
    """
    if cfg.eval.equivalence_margin != shift.EQUIVALENCE_MARGIN:
        raise ValueError("eval.equivalence_margin differs from the pre-registered one")
    runs = sweep.load_runs(locations.runs_dir, prefix)
    lams = {r.lam for r in runs}
    base_runs = [
        r for r in sweep.load_runs(locations.runs_dir, baseline_prefix) if r.lam in lams
    ]

    def own(run: sweep.Run) -> pathlib.Path:
        return locations.runs_dir / run.name

    def retrieval_of(run: sweep.Run, file: pathlib.Path) -> retrieval.RunResult:
        return dataclasses.replace(
            retrieval.load_result(file), lam=run.lam, seed=run.seed
        )

    sphere_cells = [shift.load_unit(own(r) / CELLS_FILE) for r in runs]
    radius_cells = [shift.load_unit(own(r) / usage_shift.CELLS_FILE) for r in base_runs]
    sphere_ret = [
        retrieval_of(r, own(r) / RETRIEVAL_DIR / f"{r.name}.npz") for r in runs
    ]
    radius_ret = [
        retrieval_of(r, locations.runs_dir / RADIUS_RETRIEVAL_DIR / f"{r.name}.npz")
        for r in base_runs
    ]
    sphere_ood = [ood_radius.load_unit(own(r) / usage_shift.OOD_FILE) for r in runs]
    radius_ood = [
        ood_radius.load_unit(own(r) / usage_shift.OOD_FILE) for r in base_runs
    ]

    rng = numpy.random.default_rng(cfg.seed)
    boots = cfg.eval.num_bootstrap
    margin = cfg.eval.equivalence_margin
    accuracy = compare_accuracy(
        pair_by_seed(sphere_cells, radius_cells), margin, boots, rng
    )
    retrievals = compare_retrieval(
        pair_by_seed(sphere_ret, radius_ret), cfg.eval.retrieval_margin, boots, rng
    )
    ood = compare_ood(pair_by_seed(sphere_ood, radius_ood), boots, cfg.seed)
    groups = {"radius-free (sigreg-sphere)": runs, "radius (baseline)": base_runs}

    path = locations.notes_dir / note
    reporting.update_section(
        path,
        "analyze.sphere_inputs",
        "### Inputs\n\nRadius-free runs: "
        + ", ".join(f"`{r.name}`" for r in runs)
        + ". Radius runs: "
        + ", ".join(f"`{r.name}`" for r in base_runs)
        + ".",
    )
    reporting.update_section(
        path,
        "analyze.sphere_readings",
        "### Readings of the pre-registered predictions (generated)\n\n"
        + readings(accuracy, retrievals, ood, is_healthy(runs, base_runs)),
    )
    reporting.update_section(
        path,
        "analyze.sphere_accuracy",
        "### Accuracy: radius-free minus radius model\n\n"
        "Readers fitted on STL-10 train (clean) and applied to STL-10 test and "
        "STL-10-C; differences in accuracy points of the radius-free model (z~) "
        "minus the reference on the same test images, averaged over the seed "
        f"pairs; paired bootstrap over images ({boots} replicates). Verdict with "
        f"margin ±{100 * margin:g} points: `equivalent` if the 90% interval lies "
        "inside the margin, `different` if not and the 95% interval excludes 0, "
        "else `inconclusive`. The intervals do not cover training-seed variance "
        "of the two different trained models beyond averaging the seeds.\n\n"
        + reporting.markdown_table(accuracy_rows(accuracy)),
    )
    reporting.update_section(
        path,
        "analyze.sphere_retrieval",
        "### Retrieval: radius-free minus radius model\n\n"
        "Euclidean retrieval in z (radius-free: z~); per-query paired differences "
        f"in points, margin ±{100 * cfg.eval.retrieval_margin:g}. `non-inferior`: "
        "lower end of the 90% interval above minus the margin.\n\n"
        + reporting.markdown_table(retrieval_rows(retrievals)),
    )
    reporting.update_section(
        path,
        "analyze.sphere_ood",
        "### OOD detection (ID: STL-10 test, bank: STL-10 train)\n\n"
        "AUROC / FPR@95TPR of the radius-free model's direction-only score "
        "against the radius model's scores (definitions: "
        "`lejepa_norm.eval.ood_radius`); paired bootstrap, ID and OOD images "
        "resampled separately, the same resamples for both models.\n\n"
        + reporting.markdown_table(ood_score_rows(ood))
        + "\n\n#### Differences (radius-free direction-only minus radius model)\n\n"
        + reporting.markdown_table(ood_difference_rows(ood)),
    )
    reporting.update_section(
        path,
        "analyze.sphere_health",
        "### Training health\n\n"
        + reporting.markdown_table(health_summary(groups))
        + "\n\nMean ± sd over seeds of the epoch means (loss, raw-z statistics) "
        "and of the k-NN monitor at selected epochs:\n\n"
        + reporting.markdown_table(health_rows(groups)),
    )
    if make_figures:
        links = [
            reporting.image_link(
                plotting.export_png(
                    fig, path.parent / f"sphere_{stem}.png", locations.kaleido_chrome
                ),
                path,
                alt,
            )
            for stem, fig, alt in figures(groups)
        ]
        reporting.update_section(
            path, "analyze.sphere_figures", "### Figures\n\n" + "\n\n".join(links)
        )
    return path
