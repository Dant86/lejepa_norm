"""Why does a frozen probe depend (slightly) on the radius? A mechanistic dissection.

E1 (`randomization`) found that a linear probe fitted on the clean STL-10 train
embeddings and *frozen* loses a small, lambda-dependent amount of accuracy when the
radius of the test embeddings is permuted across images, while probes *retrained*
on permuted features do not. This module asks whether that is an artifact of how a
linear classifier treats a scale (not evidence that the radius carries label
information) or something more. Only `z` is analyzed (LeJEPA's product is z), on
the center views of STL-10 `train` (bank) and `test`.

**Exact decomposition.** A logistic probe on standardized features has the logits
`s(x) = x W^T + b` in the raw features (the standardization folded into `W`, `b`;
`downstream.LinearProbe.affine`). With `x = r u` (radius `r`, direction `u`)
`s = r (u W^T) + b`: the radius multiplies the direction logits and the *effective
bias* `b` (intercept minus `W mu / sigma`, `mu` = mean of the training features)
does not. A change of radius therefore changes the *relative weight of the bias*
and nothing else; the predicted class changes only when the margin between the top
two classes is smaller than what the rescaling moves. For the pair (top class `c1`,
runner-up `c2`) the margin is `r da + db` (`da`, `db` = differences of the direction
logits and of the biases) and it changes sign at the crossing radius
`t* = -db / da` (if positive). The effective bias `b` and the logit offset `r W
u_bar` of the mean direction `u_bar` of the training set are class-level constants
(bias and radius-scaled bias); only the third part `r W (u - u_bar)` is image
specific. Their spreads over classes are reported.

**Probes** (all fitted on the clean bank unless noted; evaluated frozen on test
embeddings whose radius is modified with the conditions of `randomization`):

* `RAW`: the E1 probe (standardized z, intercept).
* `NOINT`: standardized (centered) z, no intercept: the effective bias is only
  `-W mu / sigma`.
* `HOMOG`: scale-only standardization, no intercept: logits are homogeneous in `z`,
  hence *exactly* radius invariant (a sanity check of the machinery).
* `NOBIAS`: the `RAW` weights with the effective bias set to 0 (not refitted).
* `UNIT`: trained and tested on the direction `u` alone (radius-free).
* `AUG`: trained on `AUG_COPIES` copies of the bank in which every embedding takes
  the radius of a random bank image (radius independent of direction and label),
  tested on the real z.
* `MLP_RAW`, `MLP_UNIT`, `MLP_AUG`: the same for the one-hidden-layer MLP of E1
  (weight decay chosen once on the clean bank).

**Per-image diagnostics** (probes `RAW` and `MLP_RAW`, `permute` condition): flips,
the baseline margin, the log radius ratio `log(r' / r)` between permuted and original
radius, the *mismatch* `log r' - pred` (the permuted radius against the
direction-predicted log-radius, held out as in E1) and the original residual
`log r - pred`; Spearman correlations with the per-image flip rate and accuracy
loss, a margin x |log ratio| grid, and the accuracy when every image gets the same
radius `t` (a grid of multiples of the median radius).

Statistics are those of E1: paired bootstrap over test images within each seed,
averaged over seeds, 90% CI for the equivalence verdict (margin
`eval.equivalence_margin`, 0.5 points, fixed before the results were looked at) and
95% CI otherwise. The margin is only meaningful for accuracy differences;
correlations, flip rates and the other quantities are reported with 95% CIs and no
verdict. The shuffle repeats only average out shuffle noise.
"""

# scipy has incomplete type information; its use is confined to `_rank`.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import collections.abc
import dataclasses
import os
import pathlib
import statistics
import typing

import numpy
import numpy.typing
import scipy.stats

from lejepa_norm import config
from lejepa_norm.eval import downstream
from lejepa_norm.eval import features
from lejepa_norm.eval import randomization
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
Row = dict[str, str | int | float]
Scorer = collections.abc.Callable[[FloatArray], FloatArray]

RAW = "linear: raw z (E1 probe)"
NOINT = "linear: raw z, centered, no intercept"
HOMOG = "linear: raw z, uncentered, no intercept"
NOBIAS = "linear: raw-z weights, effective bias removed"
UNIT = "linear: direction u only"
AUG = "linear: raw z, radius-permuted training"
MLP_RAW = "MLP: raw z (E1 probe)"
MLP_UNIT = "MLP: direction u only"
MLP_AUG = "MLP: raw z, radius-permuted training"
PROBES = (RAW, NOINT, HOMOG, NOBIAS, UNIT, AUG, MLP_RAW, MLP_UNIT, MLP_AUG)
# Probe each probe's baseline is compared with in the cost table.
FAMILY_RAW = {
    RAW: RAW,
    NOINT: RAW,
    HOMOG: RAW,
    NOBIAS: RAW,
    UNIT: RAW,
    AUG: RAW,
    MLP_RAW: MLP_RAW,
    MLP_UNIT: MLP_RAW,
    MLP_AUG: MLP_RAW,
}
LINEAR_READER = "linear"
MLP_READER = "MLP"
READERS = {LINEAR_READER: RAW, MLP_READER: MLP_RAW}

# Copies of the bank (each with its own radius permutation) of the AUG probes.
AUG_COPIES = 5
# Constant-radius grid: multiples (powers of 2) of the median test radius.
CURVE_EXPONENTS = tuple(numpy.arange(-3.0, 3.01, 0.5))
# Distance (log radius) assigned to images whose top-two margin never crosses 0.
NO_CROSSING = 10.0
_LOG_FLOOR = 1e-300
_BIN_MARGIN = 5
_BIN_RATIO = 3
_CHUNK = 250


def _log(p: FloatArray) -> FloatArray:
    return numpy.log(numpy.maximum(p, _LOG_FLOOR))


def _linear_scorer(
    probe: downstream.LinearProbe, direction_only: bool, bias: bool = True
) -> Scorer:
    """Logits of a linear probe as a function of the raw (or direction) features."""
    weights, intercept = probe.affine()

    def scores(x: FloatArray) -> FloatArray:
        out = (features.direction(x) if direction_only else x) @ weights.T
        return out + intercept if bias else out

    return scores


def _mlp_scorer(probe: downstream.MlpProbe, direction_only: bool) -> Scorer:
    def scores(x: FloatArray) -> FloatArray:
        return _log(probe.predict_proba(features.direction(x) if direction_only else x))

    return scores


def augmented_bank(
    bank: FloatArray, labels: IntArray, seed: int, copies: int = AUG_COPIES
) -> tuple[FloatArray, IntArray]:
    """`copies` copies of the bank, each with its radii permuted across images."""
    parts = [
        randomization.permute_radius(bank, numpy.random.default_rng([seed, 100 + c, 0]))
        for c in range(copies)
    ]
    return numpy.concatenate(parts), numpy.tile(labels, copies)


def fit_probes(
    bank: FloatArray,
    labels: IntArray,
    num_classes: int,
    seed: int,
    weight_decay: float,
) -> tuple[dict[str, Scorer], downstream.LinearProbe]:
    """Fits every probe on the clean bank; returns scorers and the `RAW` probe."""
    raw = downstream.LinearProbe(bank, labels, num_classes)
    noint = downstream.LinearProbe(bank, labels, num_classes, fit_intercept=False)
    homog = downstream.LinearProbe(
        bank, labels, num_classes, center=False, fit_intercept=False
    )
    unit = downstream.LinearProbe(features.direction(bank), labels, num_classes)
    aug_x, aug_y = augmented_bank(bank, labels, seed)
    aug = downstream.LinearProbe(aug_x, aug_y, num_classes)

    def mlp(x: FloatArray, y: IntArray) -> downstream.MlpProbe:
        return downstream.MlpProbe(x, y, num_classes, weight_decay, seed=seed)

    scorers: dict[str, Scorer] = {
        RAW: _linear_scorer(raw, False),
        NOINT: _linear_scorer(noint, False),
        HOMOG: _linear_scorer(homog, False),
        NOBIAS: _linear_scorer(raw, False, bias=False),
        UNIT: _linear_scorer(unit, True),
        AUG: _linear_scorer(aug, False),
        MLP_RAW: _mlp_scorer(mlp(bank, labels), False),
        MLP_UNIT: _mlp_scorer(mlp(features.direction(bank), labels), True),
        MLP_AUG: _mlp_scorer(mlp(aug_x, aug_y), False),
    }
    return scorers, raw


def top_two(scores: FloatArray) -> tuple[IntArray, IntArray, FloatArray]:
    """Top class, runner-up class and their score margin per row."""
    order = numpy.argsort(-scores, axis=1)[:, :2].astype(numpy.int64)
    rows = numpy.arange(scores.shape[0])
    margin = scores[rows, order[:, 0]] - scores[rows, order[:, 1]]
    return order[:, 0], order[:, 1], margin


@dataclasses.dataclass(frozen=True)
class RunAnalysis:
    """Everything the aggregation needs from one run (see `analyze_run`).

    Attributes:
        name: Run name.
        lam: SIGReg weight of the run.
        seed: Training seed of the run.
        arrays: Named float arrays: per-image correctness per (probe, condition)
            (`correct|probe|condition`), per-image diagnostics and curves.
        scalars: Named scalar diagnostics.
    """

    name: str
    lam: float
    seed: int
    arrays: dict[str, FloatArray]
    scalars: dict[str, float]


def _cell(probe: str, condition: str) -> str:
    return f"correct|{probe}|{condition}"


def analyze_run(
    name: str,
    lam: float,
    train_seed: int,
    bank: FloatArray,
    bank_labels: numpy.typing.ArrayLike,
    query: FloatArray,
    query_labels: numpy.typing.ArrayLike,
    num_repeats: int,
    cv_folds: int,
    seed: int,
) -> RunAnalysis:
    """Fits all probes on the bank and dissects their radius dependence on the test set.

    Args:
        name: Run name.
        lam: SIGReg weight of the run.
        train_seed: Training seed of the run (identification only).
        bank: `(M, d)` train center-view embeddings.
        bank_labels: `(M,)` integer labels.
        query: `(N, d)` test center-view embeddings.
        query_labels: `(N,)` integer labels.
        num_repeats: Shuffles per randomized condition.
        cv_folds: Outer folds of the direction -> radius regression.
        seed: Seed of every random draw (shuffles as in E1, regression folds, MLP).

    Returns:
        The per-image outcomes and diagnostics of the run.
    """
    by = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    qy = numpy.asarray(query_labels, dtype=numpy.int64).ravel()
    num_classes = int(max(by.max(), qy.max())) + 1
    model = randomization.fit_radius_model(
        bank, query, cv_folds, numpy.random.default_rng(seed)
    )
    weight_decay = randomization.select_weight_decay(bank, by, num_classes, seed)
    scorers, raw = fit_probes(bank, by, num_classes, seed, weight_decay)

    # Conditions of E1 with the same random streams (test-set stream of each repeat).
    permuted: list[FloatArray] = []
    residual: list[FloatArray] = []
    for repeat in range(num_repeats):
        permuted.append(
            randomization.permute_radius(
                query, numpy.random.default_rng([seed, repeat, 1])
            )
        )
        residual.append(
            randomization.residual_shuffle(
                query, model.query, numpy.random.default_rng([seed, repeat, 1])
            )
        )
    conditions: dict[str, list[FloatArray]] = {
        randomization.BASELINE: [query],
        randomization.PERMUTE: permuted,
        randomization.RESIDUAL: residual,
        randomization.CONSTANT: [randomization.constant_radius(query)],
    }

    arrays: dict[str, FloatArray] = {"label": qy.astype(numpy.float64)}
    predictions: dict[str, dict[str, list[IntArray]]] = {}
    for probe, scorer in scorers.items():
        predictions[probe] = {}
        for condition, queries in conditions.items():
            preds = [scorer(q).argmax(axis=1).astype(numpy.int64) for q in queries]
            predictions[probe][condition] = preds
            hits = numpy.stack([p == qy for p in preds]).astype(numpy.float64)
            arrays[_cell(probe, condition)] = hits.mean(axis=0)

    radius = features.radius(query)
    log_r = features.log_radius(query)
    log_ratio = numpy.stack([features.log_radius(q) - log_r for q in permuted])
    mismatch = numpy.stack([features.log_radius(q) - model.query for q in permuted])
    arrays["radius"] = radius
    arrays["log_ratio"] = log_ratio
    arrays["mismatch"] = mismatch
    arrays["resid"] = log_r - model.query

    for reader, probe in READERS.items():
        base_scores = scorers[probe](query)
        _, runner, margin = top_two(base_scores)
        base_pred = predictions[probe][randomization.BASELINE][0]
        flips, to_runner, d_correct = [], [], []
        for pred in predictions[probe][randomization.PERMUTE]:
            changed = pred != base_pred
            flips.append(changed)
            to_runner.append(changed & (pred == runner))
            d_correct.append((pred == qy).astype(float) - (base_pred == qy))
        arrays[f"flip|{reader}"] = numpy.stack(flips).astype(numpy.float64)
        arrays[f"to_runner_up|{reader}"] = numpy.stack(to_runner).astype(numpy.float64)
        arrays[f"d_correct|{reader}"] = numpy.stack(d_correct).astype(numpy.float64)
        arrays[f"margin|{reader}"] = margin

    # Constant-radius curves: accuracy when every image has radius 2^e * median.
    median_radius = float(numpy.median(radius))
    unit_query = features.direction(query)
    arrays["curve_exponent"] = numpy.asarray(CURVE_EXPONENTS, dtype=numpy.float64)
    for probe, scorer in scorers.items():
        arrays[f"curve|{probe}"] = numpy.asarray(
            [
                float(
                    (
                        scorer(unit_query * (2.0**e) * median_radius).argmax(axis=1)
                        == qy
                    ).mean()
                )
                for e in CURVE_EXPONENTS
            ]
        )

    scalars = _decomposition(raw, bank, query, arrays)
    scalars["weight_decay"] = weight_decay
    scalars["median_radius"] = median_radius
    scalars["radius_r2_test"] = model.r2_query
    return RunAnalysis(name, lam, train_seed, arrays, scalars)


def _centered_spread(x: FloatArray) -> FloatArray:
    """Standard deviation over the last axis (classes) after removing the mean."""
    return (x - x.mean(axis=-1, keepdims=True)).std(axis=-1)


def _decomposition(
    probe: downstream.LinearProbe,
    bank: FloatArray,
    query: FloatArray,
    arrays: dict[str, FloatArray],
) -> dict[str, float]:
    """Scalars of the exact logit decomposition `r (u W^T) + b` of the RAW probe.

    Also stores the per-image distance (log radius) to the top-two crossing in
    `arrays["cross_distance"]`.
    """
    weights, bias = probe.affine()
    u_bank_mean = features.direction(bank).mean(axis=0)
    u = features.direction(query)
    radius = features.radius(query)
    direction_logits = u @ weights.T
    scores = radius[:, None] * direction_logits + bias
    top, runner, margin = top_two(scores)
    rows = numpy.arange(query.shape[0])
    d_a = direction_logits[rows, top] - direction_logits[rows, runner]
    d_b = bias[top] - bias[runner]
    with numpy.errstate(divide="ignore", invalid="ignore"):
        crossing = numpy.where(d_a != 0, -d_b / d_a, numpy.nan)
        distance = numpy.abs(numpy.log(crossing / radius))
    # No crossing at a positive radius (or none that is finite): never flips.
    arrays["cross_distance"] = numpy.where(
        numpy.isfinite(distance) & (crossing > 0), distance, NO_CROSSING
    )
    median_radius = float(numpy.median(radius))
    mean_part = median_radius * (weights @ u_bank_mean)
    image_part = radius[:, None] * ((u - u_bank_mean) @ weights.T)
    return {
        "bias spread (sd over classes)": float(_centered_spread(bias)),
        "mean-direction offset spread (r_med W u_bar)": float(
            _centered_spread(mean_part)
        ),
        "image-specific logit spread (median over images)": float(
            numpy.median(_centered_spread(image_part))
        ),
        "mean resultant length of the bank directions": float(
            numpy.linalg.norm(u_bank_mean)
        ),
        "median top-two margin": float(numpy.median(margin)),
        "median |d margin / d log r| = r |da|": float(numpy.median(radius * abs(d_a))),
        "median |db| (bias gap, top two)": float(numpy.median(abs(d_b))),
        "share of images with a crossing": float(
            numpy.mean(arrays["cross_distance"] < NO_CROSSING)
        ),
    }


# --- Persistence (one shard per run, so that SLURM jobs can run in parallel) ------


def save_run(path: str | os.PathLike[str], result: RunAnalysis) -> None:
    """Writes a run analysis atomically as an `.npz` file."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    payload: dict[str, typing.Any] = dict(result.arrays)
    payload.update({f"scalar|{k}": numpy.asarray(v) for k, v in result.scalars.items()})
    payload["meta|name"] = numpy.asarray(result.name)
    payload["meta|lam"] = numpy.asarray(result.lam)
    payload["meta|seed"] = numpy.asarray(result.seed)
    with partial.open("wb") as handle:
        numpy.savez_compressed(handle, **payload)
    partial.replace(target)


def load_run(path: str | os.PathLike[str]) -> RunAnalysis:
    """Reads a file written by `save_run`."""
    with numpy.load(path, allow_pickle=False) as data:
        arrays = {
            k: numpy.asarray(data[k], dtype=numpy.float64)
            for k in data.files
            if not k.startswith(("scalar|", "meta|"))
        }
        scalars = {
            k.removeprefix("scalar|"): float(data[k])
            for k in data.files
            if k.startswith("scalar|")
        }
        return RunAnalysis(
            str(data["meta|name"]),
            float(data["meta|lam"]),
            int(data["meta|seed"]),
            arrays,
            scalars,
        )


def analyze_shards(
    name: str,
    lam: float,
    seed: int,
    bank: shards.EmbeddingShard,
    query: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    draw_seed: int,
) -> RunAnalysis:
    """`analyze_run` on the z center views of the `stl10_train` / `stl10_test` dumps."""
    return analyze_run(
        name,
        lam,
        seed,
        randomization.center_embeddings(bank.z),
        bank.label,
        randomization.center_embeddings(query.z),
        query.label,
        num_repeats=cfg.num_randomizations,
        cv_folds=cfg.cv_folds,
        seed=draw_seed,
    )


# --- Aggregation over seeds, per lambda ----------------------------------------


def _groups(results: list[RunAnalysis]) -> dict[float, list[RunAnalysis]]:
    return {
        lam: sorted((r for r in results if r.lam == lam), key=lambda r: r.seed)
        for lam in sorted({r.lam for r in results})
    }


def _num_images(group: list[RunAnalysis]) -> int:
    return group[0].arrays["label"].size


def _points(value: float) -> str:
    return f"{100 * value:+.2f}"


def _interval(ci: tuple[float, float]) -> str:
    return f"[{_points(ci[0])}, {_points(ci[1])}]"


def _mean_accuracy(group: list[RunAnalysis], key: str) -> float:
    return statistics.fmean(float(r.arrays[key].mean()) for r in group)


def _difference_row(
    lam: float,
    group: list[RunAnalysis],
    draws: list[IntArray],
    first: str,
    second: str,
    margin: float,
) -> Row | None:
    """Mean paired accuracy difference `first - second` (cell keys) with verdict."""
    if any(first not in r.arrays or second not in r.arrays for r in group):
        return None
    diffs = [r.arrays[first] - r.arrays[second] for r in group]
    ci90, ci95 = randomization.intervals(
        randomization.bootstrap_seed_mean(
            draws, lambda s, idx: diffs[s][idx].mean(axis=1)
        )
    )
    row: Row = {
        "lambda": lam,
        "reference acc": f"{_mean_accuracy(group, second):.4f}",
        "acc": f"{_mean_accuracy(group, first):.4f}",
        "difference (points)": _points(
            statistics.fmean(float(d.mean()) for d in diffs)
        ),
        "90% CI (points)": _interval(ci90),
        "95% CI (points)": _interval(ci95),
        "verdict": randomization.verdict(ci90, ci95, margin),
    }
    return row


def change_rows(
    results: list[RunAnalysis],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Row]:
    """Frozen accuracy change of every probe under each radius condition."""
    rows: list[Row] = []
    for lam, group in _groups(results).items():
        draws = randomization.bootstrap_draws(
            _num_images(group), len(group), num_bootstrap, rng
        )
        for probe in PROBES:
            for condition in randomization.CONDITIONS[1:]:
                out = _difference_row(
                    lam,
                    group,
                    draws,
                    _cell(probe, condition),
                    _cell(probe, randomization.BASELINE),
                    margin,
                )
                if out is not None:
                    rows.append({"probe": probe, "condition": condition, **out})
    return rows


def cost_rows(
    results: list[RunAnalysis],
    margin: float,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[Row]:
    """Clean accuracy of each probe against the E1 probe of its family."""
    rows: list[Row] = []
    for lam, group in _groups(results).items():
        draws = randomization.bootstrap_draws(
            _num_images(group), len(group), num_bootstrap, rng
        )
        for probe in PROBES:
            if FAMILY_RAW[probe] == probe:
                continue
            out = _difference_row(
                lam,
                group,
                draws,
                _cell(probe, randomization.BASELINE),
                _cell(FAMILY_RAW[probe], randomization.BASELINE),
                margin,
            )
            if out is not None:
                rows.append({"probe": probe, "versus": FAMILY_RAW[probe], **out})
    return rows


def _rank(x: FloatArray) -> FloatArray:
    """Average ranks along the last axis."""
    return numpy.asarray(scipy.stats.rankdata(x, axis=-1), dtype=numpy.float64)


def spearman(x: FloatArray, y: FloatArray, idx: IntArray) -> FloatArray:
    """Spearman correlation of `x` and `y` on every resample of `idx` (B, N)."""
    out: list[FloatArray] = []
    for start in range(0, idx.shape[0], _CHUNK):
        block = idx[start : start + _CHUNK]
        rx, ry = _rank(x[block]), _rank(y[block])
        rx = rx - rx.mean(axis=1, keepdims=True)
        ry = ry - ry.mean(axis=1, keepdims=True)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            out.append(
                (rx * ry).sum(axis=1)
                / numpy.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
            )
    return numpy.concatenate(out)


Statistic = collections.abc.Callable[[int, IntArray], FloatArray]


def _estimate(
    group: list[RunAnalysis], draws: list[IntArray], statistic: Statistic
) -> tuple[float, tuple[float, float]]:
    """Seed-averaged statistic on the full data and its 95% bootstrap interval."""
    everything = numpy.arange(_num_images(group), dtype=numpy.int64)[None, :]
    value = float(numpy.mean([statistic(s, everything)[0] for s in range(len(group))]))
    _, ci95 = randomization.intervals(
        randomization.bootstrap_seed_mean(draws, statistic)
    )
    return value, ci95


def _with_ci(value: float, ci: tuple[float, float], digits: int = 3) -> str:
    return f"{value:+.{digits}f} [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def _mean_of(values: list[FloatArray]) -> Statistic:
    """Per seed, the mean of per-image `values` over a resample."""
    return lambda s, idx: values[s][idx].mean(axis=1)


def _ratio_of(numerator: list[FloatArray], denominator: list[FloatArray]) -> Statistic:
    """Per seed, ratio of the resample means of two per-image quantities."""
    return lambda s, idx: (
        numerator[s][idx].mean(axis=1) / denominator[s][idx].mean(axis=1)
    )


def _spearman_of(x: list[FloatArray], y: list[FloatArray]) -> Statistic:
    """Per seed, the Spearman correlation of per-image `x` and `y` on a resample."""
    return lambda s, idx: spearman(x[s], y[s], idx)


def flip_rows(
    results: list[RunAnalysis], num_bootstrap: int, rng: numpy.random.Generator
) -> list[Row]:
    """Flip statistics of the frozen RAW probes under the `permute` condition."""
    rows: list[Row] = []
    for lam, group in _groups(results).items():
        draws = randomization.bootstrap_draws(
            _num_images(group), len(group), num_bootstrap, rng
        )
        for reader in READERS:
            flip = [numpy.atleast_2d(r.arrays[f"flip|{reader}"]) for r in group]
            runner = [r.arrays[f"to_runner_up|{reader}"].mean(axis=0) for r in group]
            flipped = [f.mean(axis=0) for f in flip]
            d_corr = [r.arrays[f"d_correct|{reader}"] for r in group]
            lose = [(d == -1).mean(axis=0) for d in d_corr]
            gain = [(d == 1).mean(axis=0) for d in d_corr]
            net = [d.mean(axis=0) for d in d_corr]

            row: Row = {"lambda": lam, "reader": reader}
            for label, stat in (
                ("flip rate (points)", _mean_of(flipped)),
                ("correct -> wrong (points)", _mean_of(lose)),
                ("wrong -> correct (points)", _mean_of(gain)),
                ("net accuracy change (points)", _mean_of(net)),
            ):
                value, ci = _estimate(group, draws, stat)
                row[label] = _with_ci(100 * value, (100 * ci[0], 100 * ci[1]), 2)
            value, ci = _estimate(group, draws, _ratio_of(runner, flipped))
            row["share of flips that go to the baseline runner-up"] = _with_ci(
                value, ci, 2
            )
            rows.append(row)
    return rows


def _predictors(reader: str, r: RunAnalysis) -> dict[str, FloatArray]:
    """Per-image candidate predictors of flips (larger = more flip-prone)."""
    out = {
        "|log radius ratio| (permuted / original)": numpy.abs(
            r.arrays["log_ratio"]
        ).mean(axis=0),
        "|mismatch| (permuted log r - predicted)": numpy.abs(r.arrays["mismatch"]).mean(
            axis=0
        ),
        "|residual| (original log r - predicted)": numpy.abs(r.arrays["resid"]),
        "- baseline margin": -r.arrays[f"margin|{reader}"],
    }
    if reader == LINEAR_READER:
        out["- log distance to the crossing radius"] = -r.arrays["cross_distance"]
    return out


def correlation_rows(
    results: list[RunAnalysis], num_bootstrap: int, rng: numpy.random.Generator
) -> list[Row]:
    """Spearman correlations of per-image predictors with flips and accuracy loss."""
    rows: list[Row] = []
    for lam, group in _groups(results).items():
        draws = randomization.bootstrap_draws(
            _num_images(group), len(group), num_bootstrap, rng
        )
        for reader in READERS:
            targets = {
                "flip rate": [
                    numpy.atleast_2d(r.arrays[f"flip|{reader}"]).mean(axis=0)
                    for r in group
                ],
                "accuracy loss": [
                    -r.arrays[f"d_correct|{reader}"].mean(axis=0) for r in group
                ],
            }
            predictors = [_predictors(reader, r) for r in group]
            for name in predictors[0]:
                row: Row = {"lambda": lam, "reader": reader, "predictor": name}
                for tname, target in targets.items():
                    value, ci = _estimate(
                        group,
                        draws,
                        _spearman_of([p[name] for p in predictors], target),
                    )
                    row[f"Spearman with {tname}"] = _with_ci(value, ci)
                rows.append(row)
    return rows


def _quantile_bins(x: FloatArray, bins: int) -> IntArray:
    edges = numpy.quantile(x, numpy.linspace(0, 1, bins + 1)[1:-1])
    return numpy.searchsorted(edges, x, side="right").astype(numpy.int64)


def grid_rows(results: list[RunAnalysis]) -> list[Row]:
    """Flip rate (points) on a baseline-margin quintile x |log ratio| tercile grid.

    Bins are quantiles within each seed (margin over images, ratio over all
    image/repeat pairs); the cell is the mean over seeds. Descriptive: no CI.
    """
    rows: list[Row] = []
    for lam, group in _groups(results).items():
        for reader in READERS:
            cells = numpy.zeros((len(group), _BIN_MARGIN, _BIN_RATIO))
            for s, r in enumerate(group):
                flip = numpy.atleast_2d(r.arrays[f"flip|{reader}"])
                ratio = numpy.abs(r.arrays["log_ratio"])
                m_bin = numpy.broadcast_to(
                    _quantile_bins(r.arrays[f"margin|{reader}"], _BIN_MARGIN),
                    flip.shape,
                )
                r_bin = _quantile_bins(ratio.ravel(), _BIN_RATIO).reshape(ratio.shape)
                for i in range(_BIN_MARGIN):
                    for j in range(_BIN_RATIO):
                        mask = (m_bin == i) & (r_bin == j)
                        cells[s, i, j] = flip[mask].mean() if mask.any() else numpy.nan
            mean = numpy.nanmean(cells, axis=0)
            for i in range(_BIN_MARGIN):
                row: Row = {
                    "lambda": lam,
                    "reader": reader,
                    "baseline-margin quintile": f"Q{i + 1}"
                    + (" (smallest)" if i == 0 else ""),
                }
                for j in range(_BIN_RATIO):
                    row[f"|log ratio| tercile {j + 1}"] = f"{100 * mean[i, j]:.2f}"
                rows.append(row)
    return rows


def decomposition_rows(results: list[RunAnalysis]) -> list[Row]:
    """Mean ± sd over seeds of the scalars of the exact decomposition, per lambda."""
    rows: list[Row] = []
    for lam, group in _groups(results).items():
        for key in group[0].scalars:
            values = [r.scalars[key] for r in group]
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            rows.append(
                {
                    "lambda": lam,
                    "quantity": key,
                    "mean ± sd over seeds": (
                        f"{statistics.fmean(values):.4g} ± {sd:.2g}"
                    ),
                }
            )
    return rows


@dataclasses.dataclass(frozen=True)
class CurvePoint:
    """Accuracy of a probe when every test image has the same radius.

    Attributes:
        lam: SIGReg weight.
        probe: Probe name.
        exponent: The radius is `2**exponent` times the median test radius.
        accuracy: Mean accuracy over seeds.
    """

    lam: float
    probe: str
    exponent: float
    accuracy: float


def curve_points(results: list[RunAnalysis]) -> list[CurvePoint]:
    """Constant-radius accuracy curves averaged over seeds."""
    out: list[CurvePoint] = []
    for lam, group in _groups(results).items():
        for probe in PROBES:
            curve = numpy.mean([r.arrays[f"curve|{probe}"] for r in group], axis=0)
            out += [
                CurvePoint(lam, probe, float(e), float(a))
                for e, a in zip(group[0].arrays["curve_exponent"], curve, strict=True)
            ]
    return out


def curve_rows(points: list[CurvePoint], probes: tuple[str, ...]) -> list[Row]:
    """Wide table of the constant-radius accuracies of `probes` (mean over seeds)."""
    rows: list[Row] = []
    keys = sorted({(p.lam, p.exponent) for p in points})
    for lam, exponent in keys:
        row: Row = {"lambda": lam, "log2(radius / median radius)": exponent}
        for p in points:
            if p.lam == lam and p.exponent == exponent and p.probe in probes:
                row[p.probe] = f"{p.accuracy:.4f}"
        rows.append(row)
    return rows
