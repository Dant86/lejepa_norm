"""Is the radius used when labels are scarce? E1 with a small labeled bank.

If the radius carried a confidence-like signal it could matter most when little
labeled data is available. This repeats the E1 readers (`randomization`) on class-
balanced random subsets of the labeled STL-10 `train` split (500 images per class)
with `PER_CLASS_BUDGETS` labels per class (5, 10, 50, 500 = 50, 100, 500, 5000
labels in total, i.e. 1%, 2%, 10% and 100% of the split). Only `z` is analyzed
(LeJEPA's product is z), center views, test set = STL-10 `test`.

Design, fixed before any condition effect was looked at:

* Readers: Euclidean k-NN, linear probe, one-hidden-layer MLP (cosine k-NN is
  radius-invariant by construction and is dropped). Conditions and protocols are
  those of `randomization` (the same code runs them: `randomization.condition_cells`).
* Subsets: `NUM_SUBSETS` (10) independent class-balanced draws per budget, the same
  draws for every run (the subset rng depends only on the budget and the index, so
  a subset is the same set of images in all nine runs, and the same for all
  conditions: comparisons are paired). The budget that takes the whole split has a
  single subset (all draws would be identical).
* Hyperparameters, chosen per subset on the *clean* labeled subset by stratified
  cross-validation inside the subset (`min(MAX_CV_FOLDS, per-class count)` folds) and
  then reused for every condition and protocol: the linear probe's inverse
  regularization `C` from `LINEAR_CS`, the MLP's weight decay from
  `randomization.MLP_WEIGHT_DECAYS`. The number of neighbours is
  `min(knn_k, per-class count)`: with 5 labeled images per class a neighbourhood of
  20 would be wider than the class. The grids and rules are constants of this module,
  not tuned to the results.
* Radius model of the residual shuffle: the direction -> log-radius regression uses
  no labels, so it is fitted once per run on the *whole* unlabeled `train` split
  (`randomization.fit_radius_model`, out-of-fold for the bank) and the predictions of
  the subset's images are taken from it. Fitting it on 50 images would make the
  "predictable part" meaningless and confound the residual condition with the
  labeled budget.
* Statistics: per-image correctness is averaged over subsets and shuffle repeats
  and compared with the baseline on the same test images by
  `randomization.compare` (paired bootstrap over test images within each seed,
  averaged over seeds; TOST margin `eval.equivalence_margin`, 0.5 points). That CI
  conditions on the particular subsets, so the subset-to-subset spread is reported
  next to it: the standard error over subsets of the paired difference
  (`subset_se`; the per-subset difference is measured on all test images, so its
  spread also contains test-image noise and over-states the subset noise slightly)
  and a second verdict from a normal approximation that adds it in quadrature to
  the bootstrap standard error (`verdict_with_subsets`). Both verdicts are given;
  the second is the stricter one.
"""

import collections.abc
import dataclasses
import json
import math
import os
import pathlib
import statistics
import typing

import numpy
import numpy.typing

from lejepa_norm.eval import downstream
from lejepa_norm.eval import randomization

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
Row = dict[str, str | int | float]

PER_CLASS_BUDGETS = (5, 10, 50, 500)
NUM_SUBSETS = 10
SUBSET_SEED = 20261024
MAX_CV_FOLDS = 5
LINEAR_CS = (0.01, 0.1, 1.0)
READERS = (
    randomization.KNN_EUCLIDEAN,
    randomization.LINEAR,
    randomization.MLP,
)
# Normal quantiles of the 90% (TOST) and 95% intervals.
Z90, Z95 = 1.6449, 1.9600

Key = randomization.Key


def class_balanced_subset(
    labels: numpy.typing.ArrayLike, per_class: int, rng: numpy.random.Generator
) -> IntArray:
    """Sorted indices of `per_class` random images of every class.

    Args:
        labels: `(M,)` integer labels.
        per_class: Images per class (capped at the class size).
        rng: Generator for the draw.

    Returns:
        Sorted indices into `labels`.
    """
    y = numpy.asarray(labels, dtype=numpy.int64).ravel()
    chosen = [
        rng.permutation(numpy.flatnonzero(y == c))[:per_class] for c in numpy.unique(y)
    ]
    return numpy.sort(numpy.concatenate(chosen)).astype(numpy.int64)


def subset_indices(
    labels: numpy.typing.ArrayLike, per_class: int, num_subsets: int = NUM_SUBSETS
) -> list[IntArray]:
    """The subsets of one budget (deterministic; one if it takes the whole split)."""
    y = numpy.asarray(labels, dtype=numpy.int64).ravel()
    if per_class >= int(numpy.bincount(y).min()):
        return [numpy.arange(y.size, dtype=numpy.int64)]
    return [
        class_balanced_subset(
            y, per_class, numpy.random.default_rng([SUBSET_SEED, per_class, index])
        )
        for index in range(num_subsets)
    ]


def stratified_folds(
    labels: IntArray, folds: int, rng: numpy.random.Generator
) -> IntArray:
    """Fold id of every row such that each class is spread evenly over folds."""
    out = numpy.zeros(labels.size, dtype=numpy.int64)
    for c in numpy.unique(labels):
        rows = rng.permutation(numpy.flatnonzero(labels == c))
        out[rows] = numpy.arange(rows.size) % folds
    return out


def select_hyperparameter(
    x: FloatArray,
    y: IntArray,
    candidates: tuple[float, ...],
    predict: collections.abc.Callable[
        [float, FloatArray, IntArray, FloatArray], IntArray
    ],
    folds: int,
    seed: int,
) -> float:
    """The candidate with the best stratified cross-validated accuracy (ties: first).

    Args:
        x: `(M, d)` features of the clean labeled subset.
        y: `(M,)` labels.
        candidates: Values to try.
        predict: `(candidate, x_fit, y_fit, x_val) -> predictions`.
        folds: Number of folds (at least 2).
        seed: Seed of the fold assignment.

    Returns:
        The selected candidate.
    """
    fold = stratified_folds(y, folds, numpy.random.default_rng(seed))
    scores: list[float] = []
    for value in candidates:
        hits = 0
        for f in range(folds):
            val, fit = fold == f, fold != f
            hits += int((predict(value, x[fit], y[fit], x[val]) == y[val]).sum())
        scores.append(hits / y.size)
    return candidates[int(numpy.argmax(scores))]


def _linear_predict(
    c: float, x_fit: FloatArray, y_fit: IntArray, x_val: FloatArray, num_classes: int
) -> IntArray:
    return downstream.LinearProbe(x_fit, y_fit, num_classes, c=c).predict(x_val)


def _mlp_predict(
    wd: float,
    x_fit: FloatArray,
    y_fit: IntArray,
    x_val: FloatArray,
    num_classes: int,
    seed: int,
) -> IntArray:
    return downstream.MlpProbe(x_fit, y_fit, num_classes, wd, seed=seed).predict(x_val)


def _make_fit(
    reader: str,
    x: FloatArray,
    y: IntArray,
    k: int,
    num_classes: int,
    folds: int,
    seed: int,
) -> tuple[randomization.Fit, float]:
    """The reader's `Fit` on labels `y` and its cross-validated hyperparameter.

    The hyperparameter (`C`, weight decay; 0 for k-NN) is selected on the clean
    features `x` only.
    """
    if reader == randomization.KNN_EUCLIDEAN:

        def knn(bank: FloatArray) -> randomization.Predictor:
            return lambda q: randomization.euclidean_knn_votes(
                bank, y, q, k, num_classes
            ).argmax(axis=1)

        return knn, 0.0
    if reader == randomization.LINEAR:
        c = select_hyperparameter(
            x,
            y,
            LINEAR_CS,
            lambda v, a, b, q: _linear_predict(v, a, b, q, num_classes),
            folds,
            seed,
        )

        def linear(bank: FloatArray) -> randomization.Predictor:
            return downstream.LinearProbe(bank, y, num_classes, c=c).predict

        return linear, c
    wd = select_hyperparameter(
        x,
        y,
        randomization.MLP_WEIGHT_DECAYS,
        lambda v, a, b, q: _mlp_predict(v, a, b, q, num_classes, seed),
        folds,
        seed,
    )

    def mlp(bank: FloatArray) -> randomization.Predictor:
        return downstream.MlpProbe(bank, y, num_classes, wd, seed=seed).predict

    return mlp, wd


@dataclasses.dataclass(frozen=True)
class BudgetResult:
    """All readers, protocols and conditions at one labeled-data budget.

    Attributes:
        per_class: Labeled images per class.
        num_labels: Labeled images in total.
        num_subsets: Random subsets averaged.
        cells: Per (reader, protocol, condition): `correct` is the per-test-image
            correctness averaged over subsets (and shuffle repeats), and
            `repeat_accuracy` holds the test accuracy of each *subset* (averaged
            over its shuffle repeats).
        hyper: One row per (subset, reader) with the selected hyperparameter.
    """

    per_class: int
    num_labels: int
    num_subsets: int
    cells: dict[Key, randomization.Cell]
    hyper: list[Row]


def evaluate_budget(
    bank: FloatArray,
    bank_labels: IntArray,
    query: FloatArray,
    query_labels: IntArray,
    radius_model: randomization.RadiusModel,
    per_class: int,
    k: int,
    num_repeats: int,
    seed: int,
    num_subsets: int = NUM_SUBSETS,
    readers: tuple[str, ...] = READERS,
) -> BudgetResult:
    """Runs the E1 conditions for every subset of one budget and averages them.

    Args:
        bank: `(M, d)` labeled-split embeddings.
        bank_labels: `(M,)` labels.
        query: `(N, d)` test embeddings.
        query_labels: `(N,)` labels.
        radius_model: Held-out direction -> log-radius predictions for the whole
            bank and the test set (`randomization.fit_radius_model`).
        per_class: Labeled images per class.
        k: Maximum number of neighbours (capped at the per-class count).
        num_repeats: Shuffles per randomized condition.
        seed: Base seed of shuffles, folds and the MLP.
        num_subsets: Subsets per budget.
        readers: Subset of `READERS`.

    Returns:
        The `BudgetResult`.
    """
    num_classes = int(max(bank_labels.max(), query_labels.max())) + 1
    subsets = subset_indices(bank_labels, per_class, num_subsets)
    correct: dict[Key, FloatArray] = {}
    accuracy: dict[Key, list[float]] = {}
    hyper: list[Row] = []
    for index, rows in enumerate(subsets):
        x, y = bank[rows], bank_labels[rows]
        counts = int(numpy.bincount(y).min())
        sub_seed = (seed * 1009 + per_class) * 1009 + index
        model = randomization.RadiusModel(
            bank=radius_model.bank[rows],
            query=radius_model.query,
            r2_bank=float("nan"),
            r2_query=float("nan"),
        )
        for reader in readers:
            fit, value = _make_fit(
                reader,
                x,
                y,
                min(k, counts),
                num_classes,
                max(2, min(MAX_CV_FOLDS, counts)),
                sub_seed,
            )
            hyper.append(
                {
                    "labels per class": per_class,
                    "subset": index,
                    "reader": reader,
                    "hyperparameter": value,
                }
            )
            cells = randomization.condition_cells(
                fit, x, query, query_labels, model, num_repeats, sub_seed
            )
            for (protocol, condition), cell in cells.items():
                key = (reader, protocol, condition)
                correct[key] = correct.get(key, 0.0) + cell.correct / len(subsets)
                accuracy.setdefault(key, []).append(
                    statistics.fmean(cell.repeat_accuracy)
                )
    return BudgetResult(
        per_class=per_class,
        num_labels=per_class * num_classes if len(subsets) > 1 else bank.shape[0],
        num_subsets=len(subsets),
        cells={
            key: randomization.Cell(
                correct=correct[key], repeat_accuracy=tuple(accuracy[key])
            )
            for key in correct
        },
        hyper=hyper,
    )


def evaluate_run(
    bank: FloatArray,
    bank_labels: numpy.typing.ArrayLike,
    query: FloatArray,
    query_labels: numpy.typing.ArrayLike,
    k: int,
    num_repeats: int,
    cv_folds: int,
    seed: int,
    per_class_budgets: tuple[int, ...] = PER_CLASS_BUDGETS,
    num_subsets: int = NUM_SUBSETS,
    readers: tuple[str, ...] = READERS,
    log: collections.abc.Callable[[str], None] = lambda _message: None,
) -> list[BudgetResult]:
    """`evaluate_budget` for every budget of one run (center-view z embeddings)."""
    by = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    qy = numpy.asarray(query_labels, dtype=numpy.int64).ravel()
    radius_model = randomization.fit_radius_model(
        bank, query, cv_folds, numpy.random.default_rng(seed)
    )
    out: list[BudgetResult] = []
    for per_class in per_class_budgets:
        out.append(
            evaluate_budget(
                bank,
                by,
                query,
                qy,
                radius_model,
                per_class,
                k,
                num_repeats,
                seed,
                num_subsets,
                readers,
            )
        )
        log(f"budget {per_class} labels per class done")
    return out


# --- Shards (one file per run, written by the per-run jobs) ---------------------


def save_run(path: str | os.PathLike[str], results: list[BudgetResult]) -> None:
    """Writes the budgets of one run atomically to a `.npz` file."""
    arrays: dict[str, typing.Any] = {}
    meta: list[dict[str, typing.Any]] = []
    for result in results:
        meta.append(
            {
                "per_class": result.per_class,
                "num_labels": result.num_labels,
                "num_subsets": result.num_subsets,
                "hyper": result.hyper,
                "keys": [list(key) for key in result.cells],
            }
        )
        for index, (_key, cell) in enumerate(result.cells.items()):
            arrays[f"{result.per_class}/{index}/correct"] = cell.correct
            arrays[f"{result.per_class}/{index}/accuracy"] = numpy.asarray(
                cell.repeat_accuracy
            )
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".tmp.npz")
    numpy.savez_compressed(tmp, meta=numpy.asarray(json.dumps(meta)), **arrays)
    os.replace(tmp, target)


def load_run(path: str | os.PathLike[str]) -> list[BudgetResult]:
    """Reads the file written by `save_run`."""
    with numpy.load(path, allow_pickle=False) as data:
        meta = typing.cast(list[dict[str, typing.Any]], json.loads(str(data["meta"])))
        out: list[BudgetResult] = []
        for entry in meta:
            cells: dict[Key, randomization.Cell] = {}
            for index, key in enumerate(entry["keys"]):
                prefix = f"{entry['per_class']}/{index}"
                cells[key[0], key[1], key[2]] = randomization.Cell(
                    correct=typing.cast(FloatArray, data[f"{prefix}/correct"]),
                    repeat_accuracy=tuple(
                        float(a) for a in numpy.asarray(data[f"{prefix}/accuracy"])
                    ),
                )
            out.append(
                BudgetResult(
                    per_class=int(entry["per_class"]),
                    num_labels=int(entry["num_labels"]),
                    num_subsets=int(entry["num_subsets"]),
                    cells=cells,
                    hyper=entry["hyper"],
                )
            )
    return out


# --- Aggregation over runs ------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunBudgets:
    """The budgets of one run.

    Attributes:
        name: Run name.
        lam: SIGReg weight.
        seed: Training seed.
        budgets: Results per budget (same budgets for all runs).
    """

    name: str
    lam: float
    seed: int
    budgets: list[BudgetResult]


@dataclasses.dataclass(frozen=True)
class BudgetComparison:
    """E1's `Comparison` at one budget plus the subset-to-subset spread.

    Attributes:
        per_class: Labeled images per class.
        num_labels: Labeled images in total.
        num_subsets: Subsets averaged.
        comparison: Paired bootstrap comparison over test images.
        subset_se: Standard error (accuracy units) of the mean paired difference
            over subsets, combined over seeds.
        subset_sd_baseline: Standard deviation over subsets of the baseline
            accuracy, mean over seeds.
        ci90_with_subsets: 90% interval with the subset SE added in quadrature.
        verdict_with_subsets: Verdict with the subset standard error added to the
            bootstrap one (normal approximation).
    """

    per_class: int
    num_labels: int
    num_subsets: int
    comparison: randomization.Comparison
    subset_se: float
    subset_sd_baseline: float
    verdict_with_subsets: str
    ci90_with_subsets: tuple[float, float]


def _sd(values: collections.abc.Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def compare(
    runs: list[RunBudgets],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
    readers: tuple[str, ...] = READERS,
) -> list[BudgetComparison]:
    """E1's paired comparison per lambda and budget, with the subset spread."""
    out: list[BudgetComparison] = []
    for per_class in sorted({b.per_class for r in runs for b in r.budgets}):
        by_run = {
            r.name: next(b for b in r.budgets if b.per_class == per_class) for r in runs
        }
        results = [
            randomization.RunResult(
                r.name,
                r.lam,
                r.seed,
                randomization.Evaluation(
                    cells=by_run[r.name].cells, model=[], weight_decay=float("nan")
                ),
            )
            for r in runs
        ]
        for c in randomization.compare(results, margin, num_bootstrap, rng, readers):
            group = [by_run[r.name] for r in runs if r.lam == c.lam]
            diffs = [
                numpy.asarray(
                    g.cells[c.reader, c.protocol, c.condition].repeat_accuracy
                )
                - numpy.asarray(
                    g.cells[
                        c.reader, c.protocol, randomization.BASELINE
                    ].repeat_accuracy
                )
                for g in group
            ]
            num_subsets = group[0].num_subsets
            subset_se = (
                math.sqrt(sum(float(d.var(ddof=1)) / d.size for d in diffs))
                / len(diffs)
                if num_subsets > 1
                else 0.0
            )
            baseline_sd = statistics.fmean(
                _sd(
                    g.cells[
                        c.reader, c.protocol, randomization.BASELINE
                    ].repeat_accuracy
                )
                for g in group
            )
            se_boot = (c.ci90[1] - c.ci90[0]) / (2 * Z90)
            se_total = math.hypot(se_boot, subset_se)
            ci90 = (c.difference - Z90 * se_total, c.difference + Z90 * se_total)
            ci95 = (c.difference - Z95 * se_total, c.difference + Z95 * se_total)
            out.append(
                BudgetComparison(
                    per_class=per_class,
                    num_labels=group[0].num_labels,
                    num_subsets=num_subsets,
                    comparison=c,
                    subset_se=subset_se,
                    subset_sd_baseline=baseline_sd,
                    verdict_with_subsets=randomization.verdict(ci90, ci95, margin),
                    ci90_with_subsets=ci90,
                )
            )
    return out


def comparison_rows(comparisons: list[BudgetComparison]) -> list[Row]:
    """Comparisons as table rows, differences in accuracy points (x100)."""

    def interval(ci: tuple[float, float]) -> str:
        return f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"

    return [
        {
            "lambda": b.comparison.lam,
            "labels": b.num_labels,
            "subsets": b.num_subsets,
            "reader": b.comparison.reader,
            "protocol": b.comparison.protocol,
            "condition": b.comparison.condition,
            "baseline acc": f"{b.comparison.baseline:.4f}",
            "condition acc": f"{b.comparison.accuracy:.4f}",
            "difference (points)": f"{100 * b.comparison.difference:+.2f}",
            "90% CI, test images (points)": interval(b.comparison.ci90),
            "subset SE (points)": f"{100 * b.subset_se:.2f}",
            "verdict (test images)": b.comparison.verdict,
            "verdict (with subsets)": b.verdict_with_subsets,
        }
        for b in comparisons
    ]


def accuracy_rows(runs: list[RunBudgets]) -> list[Row]:
    """Accuracy per lambda, budget, reader, protocol and condition.

    `mean ± sd over seeds`: over the training seeds, each seed's value being its
    mean over subsets. `sd over subsets`: within-run standard deviation over
    subsets, averaged over seeds.
    """
    rows: list[Row] = []
    for lam in sorted({r.lam for r in runs}):
        group = [r for r in runs if r.lam == lam]
        for index, first in enumerate(group[0].budgets):
            budgets = [r.budgets[index] for r in group]
            for reader in READERS:
                for protocol in randomization.PROTOCOLS:
                    for condition in randomization.CONDITIONS:
                        accs = [
                            b.cells[reader, protocol, condition].repeat_accuracy
                            for b in budgets
                        ]
                        means = [statistics.fmean(a) for a in accs]
                        rows.append(
                            {
                                "lambda": lam,
                                "labels": first.num_labels,
                                "reader": reader,
                                "protocol": protocol,
                                "condition": condition,
                                "accuracy, mean ± sd over seeds": (
                                    f"{statistics.fmean(means):.4f} ± {_sd(means):.4f}"
                                ),
                                "sd over subsets (mean over seeds)": (
                                    f"{statistics.fmean(_sd(a) for a in accs):.4f}"
                                ),
                            }
                        )
    return rows


def hyper_rows(runs: list[RunBudgets]) -> list[Row]:
    """How often each hyperparameter value was selected, per lambda/budget/reader."""
    rows: list[Row] = []
    for lam in sorted({r.lam for r in runs}):
        group = [r for r in runs if r.lam == lam]
        for index, first in enumerate(group[0].budgets):
            for reader in READERS[1:]:
                values = sorted(
                    float(h["hyperparameter"])
                    for r in group
                    for h in r.budgets[index].hyper
                    if h["reader"] == reader
                )
                counts = {v: values.count(v) for v in sorted(set(values))}
                rows.append(
                    {
                        "lambda": lam,
                        "labels": first.num_labels,
                        "reader": reader,
                        "selected (value: count over seeds x subsets)": ", ".join(
                            f"{v:g}: {n}" for v, n in counts.items()
                        ),
                    }
                )
    return rows


def accuracy_curve(
    runs: list[RunBudgets], lam: float, reader: str, protocol: str, condition: str
) -> tuple[list[int], list[float], list[float]]:
    """Labels, accuracy (mean over seeds and subsets) and sd over subsets of a curve."""
    group = [r for r in runs if r.lam == lam]
    labels: list[int] = []
    mean: list[float] = []
    sd: list[float] = []
    for index, first in enumerate(group[0].budgets):
        accs = [
            r.budgets[index].cells[reader, protocol, condition].repeat_accuracy
            for r in group
        ]
        labels.append(first.num_labels)
        mean.append(statistics.fmean(statistics.fmean(a) for a in accs))
        sd.append(statistics.fmean(_sd(a) for a in accs))
    return labels, mean, sd


_LETTER = {
    randomization.EQUIVALENT: "E",
    randomization.DIFFERENT: "D",
    randomization.INCONCLUSIVE: "I",
}


def summary_rows(comparisons: list[BudgetComparison]) -> list[Row]:
    """Compact table, one row per reader/protocol/condition.

    One column per lambda and budget, each cell `difference [90% CI with subsets]
    test-image verdict/with-subset verdict` (E equivalent, D different, I
    inconclusive; points).

    The frozen constant-radius condition (a distribution-shift artifact: chance
    level) is left out.
    """
    columns = sorted({(b.comparison.lam, b.num_labels) for b in comparisons})
    rows: dict[tuple[str, str, str], Row] = {}
    for b in comparisons:
        c = b.comparison
        if c.protocol == randomization.FROZEN and c.condition == randomization.CONSTANT:
            continue
        row = rows.setdefault(
            (c.protocol, c.reader, c.condition),
            {"protocol": c.protocol, "reader": c.reader, "condition": c.condition},
        )
        lo, hi = b.ci90_with_subsets
        row[f"lambda {c.lam:g}, {b.num_labels} labels"] = (
            f"{100 * c.difference:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] "
            f"{_LETTER[c.verdict]}/{_LETTER[b.verdict_with_subsets]}"
        )
    order = {name: i for i, name in enumerate(randomization.CONDITIONS)}
    ordered = sorted(
        rows.values(),
        key=lambda r: (
            randomization.PROTOCOLS.index(str(r["protocol"])),
            READERS.index(str(r["reader"])),
            order[str(r["condition"])],
        ),
    )
    names = [f"lambda {lam:g}, {n} labels" for lam, n in columns]
    return [{**r, **{n: r.get(n, "") for n in names}} for r in ordered]
