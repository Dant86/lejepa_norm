"""Is the radius signal on SVHN explained by low-level image statistics?

Follow-up to the shift study (`ood_radius`, `usage_shift`): STL-10 `test` (ID) versus
DTD / SVHN `test`, z of the center views only (LeJEPA's product is z; h is not
analyzed). The shift note found that OOD images have the *smaller* radius and that
adding the radius deviation to the direction score helps on SVHN. The caveat: this may
reflect low-level image statistics (`image_stats`) rather than semantic novelty. No
training and no new embeddings: the statistics come from the raw prepared images, the
embeddings and the shift scores (`shift_ood.npz`) from the shift study. Image
statistics are standardized (after the log transform of `image_stats.LOG_NAMES`) with
the bank (STL-10 `train`) alone; every model of statistics is fitted on the bank (ID
only), never on OOD data.

**Pre-registration** (fixed before any result was looked at; the margins are in the
constants below). Intervals are 95% percentile intervals of `B` image bootstrap
replicates (the three seeds of a lambda are averaged; ID and OOD images resampled
separately), the verdict rule for a difference `A - B` is `difference_verdict`:
`A higher` iff the lower 95% bound exceeds the margin, `equivalent` iff the 90%
interval lies inside (-margin, +margin), `B higher` iff the upper 95% bound is below
-margin, otherwise `inconclusive`.

(a) *How much of the in-distribution log-radius do the statistics explain?* Models are
fitted on the bank and evaluated on STL-10 `test` (held-out `R^2`; hyperparameters by
cross-validation inside the bank): ridge on the 15 statistics, gradient boosting
(`HistGradientBoostingRegressor`, early stopping) on the statistics, the direction ->
log-radius kernel ridge of E1/E6 (`randomization.fit_radius_model`), and the sum of the
direction prediction and a statistics model fitted on the bank's out-of-fold direction
residual. Margins: the statistics *explain log-radius materially* iff the lower 95%
bound of their `R^2` exceeds `R2_MARGIN` = 0.05; the comparison with the direction and
the increment of statistics beyond the direction use `R2_COMPARE_MARGIN` = 0.01.

(b) *OOD AUROC of the statistics alone, unsupervised.* Scores (higher = more OOD):
Mahalanobis distance of the statistics to the bank (Ledoit-Wolf covariance); the
statistics-predicted log-radius (gradient boosting fitted on the bank), low = OOD;
per statistic, the two-sided deviation `|z|` and the one-sided score oriented by the
sign of the statistic's bank correlation with log-radius (low-radius direction; fixed
by ID alone; single-statistic rows are descriptive). Compared with the radius alone
(raw `||z||`, the orientation-free deviation `|log r - mu| / sd`, and `-log r`, whose
orientation was read off the shift note's radius AUROC below 0.5) and with the
direction-only k-NN, at margin `ood_radius.OOD_AUROC_MARGIN` = 0.01.

(c) *Does the radius still add to the direction score after removing the statistics?*
The log-radius is regressed on the statistics (ID only: fitted on the bank; primary
model gradient boosting, secondary ridge; bank residuals out of fold) and the score
component is the residual deviation `|res - mu| / sd`, standardized with the bank and
added with equal weight to the standardized direction score, exactly as the
pre-registered combination of the shift study. Primary criterion: the lower 95% bound
of `AUROC(direction + residual deviation) - AUROC(direction)` exceeds 0.01 (the radius
still adds). Exploratory (as in the shift note): the signed residual with weight -1.

(d) *Does direction + statistics (no radius) match direction + radius?* Equal-weight
sum of the standardized direction score and the standardized statistics Mahalanobis
distance; primary comparison `AUROC(direction + radius deviation) - AUROC(direction +
statistics)`; the radius is *not* needed beyond the statistics if the verdict is
`equivalent` or `B higher`; `A higher` means the radius carries information that the
statistics do not. Additional: the radius deviation added on top of direction +
statistics.

*Matched control.* Standardized (bank) mean luminance, contrast (luminance std) and
log Laplacian variance (blur) define the matching space. Every SVHN image is matched
to its nearest STL-10 `test` image in that space (with replacement), giving a
statistics-matched ID reference; `closed` = `1 - mean(rho_svhn - rho_matched) /
(mean(rho_svhn) - mean(rho_id))` is the fraction of the SVHN-versus-STL-10 log-radius
gap that the matching removes (`rho` = log-radius; bootstrap over SVHN images with the
ID images treated as a fixed reference). Also: the AUROC of `-log r` for SVHN against
(i) all STL-10 `test`, (ii) the matched reference with the match multiplicities as
weights, (iii) the `SUBSET_FRACTION` = 10% of STL-10 `test` images closest to SVHN in
the matching space. Criteria: the radius is *still informative on matched images* iff
the 95% interval of AUROC (ii) lies outside [0.45, 0.55]
(`ood_radius.RADIUS_NULL_BAND`), *not informative* iff entirely inside.

*Overall reading for SVHN, per lambda (fixed in advance).* `explained by low-level
statistics`: the residual combination of (c) does not add (verdict not `A higher`),
(d) is `equivalent` or `B higher` and the matched AUROC (ii) is not informative.
`independent of low-level statistics`: (c) adds, (d) is `A higher` and (ii) is
informative. Everything else: `partly explained`. The fraction of the AUROC gain of
the shift combination that remains after removing the statistics is reported
alongside.

*Places365 study* (`PLACES`; `research_notes/radius_usage/places_control.md`). The
same analysis with Places365 validation (only downsampled to 96x96) as the decisive
OOD set, because the upsampled SVHN saturated the statistics. Differences: matching
is capped at `PLACES_MAX_MATCHES` per STL-10 test image (`capped_nearest`), and the
overall reading is `INVALID` unless the validity checks hold: the statistics
Mahalanobis AUROC is at most `VALID_MAX_MAHAL_AUROC`, the standardized mean
difference after matching is below `VALID_MAX_ABS_SMD` on each matching statistic,
and the cap holds.
"""

# scikit-learn ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import collections.abc
import dataclasses
import os
import pathlib
import typing

import numpy
import numpy.typing
import sklearn.covariance
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm import reporting
from lejepa_norm.data import layout
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import image_stats
from lejepa_norm.eval import ood_radius
from lejepa_norm.eval import randomization
from lejepa_norm.eval import sweep

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.intp]
Row = dict[str, str | int | float]

R2_MARGIN = 0.05
R2_COMPARE_MARGIN = 0.01
AUROC_MARGIN = ood_radius.OOD_AUROC_MARGIN
SUBSET_FRACTION = 0.1
MATCH_NAMES = ("lum_mean", "lum_std", "laplacian_var")
RIDGE_ALPHAS = tuple(float(a) for a in numpy.logspace(-3, 3, 13))
GBM_ITERATIONS = 300
STATS_FILE = "image_stats.npz"
_SPLITS = {
    dumps.STL10_TRAIN: ("stl10", "train"),
    dumps.STL10_TEST: ("stl10", "test"),
    dumps.DTD_TEST: ("dtd", "test"),
    dumps.SVHN_TEST: ("svhn", "test"),
    dumps.PLACES_TEST: ("places365", "val"),
}
# Validity of the Places365 control (pre-registered in `places_control.md`).
VALID_MAX_MAHAL_AUROC = 0.90
VALID_MAX_ABS_SMD = 0.25
PLACES_MAX_MATCHES = 5


@dataclasses.dataclass(frozen=True)
class Study:
    """One control study: which OOD set is decisive and how it is matched.

    Attributes:
        name: Short name; prefix of unit/statistics files, note sections and figures.
        label: Human-readable name of the decisive OOD set.
        target: The decisive OOD set (`dumps` name).
        ood_sets: OOD sets analyzed (DTD is descriptive).
        max_matches: Cap on the matches per STL-10 test image (None: unlimited).
        validity: Whether the pre-registered validity checks V1 to V3 apply.
    """

    name: str
    label: str
    target: str
    ood_sets: tuple[str, ...]
    max_matches: int | None
    validity: bool

    @property
    def stat_sets(self) -> tuple[str, ...]:
        """Sets whose image statistics are needed."""
        return (dumps.STL10_TRAIN, dumps.STL10_TEST, *self.ood_sets)

    @property
    def unit_file(self) -> str:
        """Name of the per-run unit file."""
        return f"{self.name}_control.npz"

    @property
    def stats_dir(self) -> str:
        """Directory (under the runs directory) of the statistics file."""
        return f"_{self.name}_control"


SVHN = Study("svhn", "SVHN", dumps.SVHN_TEST, dumps.OOD_SETS, None, False)
PLACES = Study(
    "places",
    "Places365",
    dumps.PLACES_TEST,
    (dumps.DTD_TEST, dumps.PLACES_TEST),
    PLACES_MAX_MATCHES,
    True,
)
STUDIES = {study.name: study for study in (SVHN, PLACES)}

# (a) predictors of the ID log-radius.
R2_STATS_RIDGE = "statistics, ridge"
R2_STATS_GBM = "statistics, gradient boosting"
R2_DIRECTION = "direction (kernel ridge, E1/E6 map)"
R2_JOINT_RIDGE = "direction + statistics (ridge on direction residual)"
R2_JOINT_GBM = "direction + statistics (gradient boosting on direction residual)"
R2_NAMES = (R2_STATS_RIDGE, R2_STATS_GBM, R2_DIRECTION, R2_JOINT_RIDGE, R2_JOINT_GBM)

# (b)-(d) OOD score columns (higher = more OOD). The first five are the columns of
# the shift study (`ood_radius`), reused as they are.
RADIUS_HIGH = ood_radius.RADIUS
DEVIATION = ood_radius.DEVIATION
KNN_DIRECTION = ood_radius.KNN_DIRECTION
COMBINED = ood_radius.COMBINED
POSTHOC_LOW = f"{ood_radius.ORACLE} | w=-1"
RADIUS_LOW = "radius, low = OOD (-log r; orientation read from the shift note)"
STATS_MAHAL = "statistics: Mahalanobis distance to the bank"
STATS_PRED = "statistics: predicted log-radius, low = OOD (gradient boosting)"
RES_DEV = "residual radius deviation given statistics (gradient boosting)"
RES_DEV_RIDGE = "residual radius deviation given statistics (ridge)"
RES_LOW = "residual log-radius given statistics, low = OOD (gradient boosting)"
DIR_RES_DEV = "direction + residual radius deviation (gradient boosting)"
DIR_RES_DEV_RIDGE = "direction + residual radius deviation (ridge)"
DIR_RES_LOW = "post hoc: direction + low residual radius (w = -1, gradient boosting)"
DIR_STATS = "direction + statistics Mahalanobis (no radius)"
DIR_STATS_PRED = "direction + statistics-predicted low radius (no radius)"
DIR_STATS_RADIUS = "direction + statistics Mahalanobis + radius deviation"
BOOT_COLUMNS = (
    RADIUS_HIGH,
    DEVIATION,
    RADIUS_LOW,
    KNN_DIRECTION,
    COMBINED,
    POSTHOC_LOW,
    STATS_MAHAL,
    STATS_PRED,
    RES_DEV,
    RES_DEV_RIDGE,
    RES_LOW,
    DIR_RES_DEV,
    DIR_RES_DEV_RIDGE,
    DIR_RES_LOW,
    DIR_STATS,
    DIR_STATS_PRED,
    DIR_STATS_RADIUS,
)
STAT_DEV = "stat_dev::"
STAT_LOW = "stat_low::"

# (name, A, B): pre-registered AUROC differences with a verdict (margin 0.01).
COMPARISONS = (
    ("(b) statistics Mahalanobis - radius deviation", STATS_MAHAL, DEVIATION),
    ("(b) statistics-predicted low radius - low radius", STATS_PRED, RADIUS_LOW),
    ("(b) statistics Mahalanobis - direction only", STATS_MAHAL, KNN_DIRECTION),
    ("(b) low radius - direction only", RADIUS_LOW, KNN_DIRECTION),
    (
        "(c) reference: direction + radius deviation - direction",
        COMBINED,
        KNN_DIRECTION,
    ),
    (
        "(c) PRIMARY: direction + residual deviation (GB) - direction",
        DIR_RES_DEV,
        KNN_DIRECTION,
    ),
    (
        "(c) direction + residual deviation (ridge) - direction",
        DIR_RES_DEV_RIDGE,
        KNN_DIRECTION,
    ),
    (
        "(c) direction + residual deviation (GB) - direction + radius deviation",
        DIR_RES_DEV,
        COMBINED,
    ),
    (
        "(c) post hoc: direction + low residual - direction",
        DIR_RES_LOW,
        KNN_DIRECTION,
    ),
    ("(d) direction + statistics - direction", DIR_STATS, KNN_DIRECTION),
    (
        "(d) PRIMARY: direction + radius deviation - direction + statistics",
        COMBINED,
        DIR_STATS,
    ),
    (
        "(d) post hoc: direction + low radius - direction + statistics-predicted low "
        "radius",
        POSTHOC_LOW,
        DIR_STATS_PRED,
    ),
    (
        "(d) radius deviation on top of direction + statistics",
        DIR_STATS_RADIUS,
        DIR_STATS,
    ),
)
A_HIGHER = "A higher"
EQUIVALENT = "equivalent"
B_HIGHER = "B higher"
INCONCLUSIVE = ood_radius.INCONCLUSIVE
INVALID = "invalid control (not interpreted)"
EXPLAINED = "explained by low-level statistics"
INDEPENDENT = "independent of low-level statistics"
PARTLY = "partly explained"


def difference_verdict(difference: FloatArray, margin: float) -> str:
    """The pre-registered reading of a difference `A - B` (see module docstring)."""
    lo95, hi95 = (float(v) for v in numpy.quantile(difference[1:], [0.025, 0.975]))
    lo90, hi90 = (float(v) for v in numpy.quantile(difference[1:], [0.05, 0.95]))
    if lo95 > margin:
        return A_HIGHER
    if lo90 > -margin and hi90 < margin:
        return EQUIVALENT
    if hi95 < -margin:
        return B_HIGHER
    return INCONCLUSIVE


# --------------------------------------------------------------------------- stats


def _stats_path(runs_dir: str | os.PathLike[str], study: Study) -> pathlib.Path:
    return pathlib.Path(runs_dir) / study.stats_dir / STATS_FILE


def compute_image_stats(
    locations: paths.Paths,
    log: collections.abc.Callable[[str], None],
    study: Study = SVHN,
) -> pathlib.Path:
    """Computes `image_stats` of the study's sets once; returns the written file."""
    arrays: dict[str, FloatArray] = {}
    for name in study.stat_sets:
        dataset, split = _SPLITS[name]
        images = numpy.load(layout.images_path(locations.data_dir, dataset, split))
        arrays[name] = image_stats.compute(images)
        log(f"{name}: statistics of {images.shape[0]} images")
    out = _stats_path(locations.runs_dir, study)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with open(partial, "wb") as f:
        numpy.savez(f, **arrays)  # pyright: ignore[reportArgumentType]
    os.replace(partial, out)
    return out


def load_image_stats(
    runs_dir: str | os.PathLike[str], study: Study = SVHN
) -> dict[str, FloatArray]:
    """Raw (untransformed) statistics per set written by `compute_image_stats`."""
    with numpy.load(_stats_path(runs_dir, study)) as arrays:
        return {name: arrays[name] for name in study.stat_sets}


# ------------------------------------------------------------------ model building


@dataclasses.dataclass
class StatsModel:
    """A regressor of log-radius on statistics, fitted on the bank only.

    Attributes:
        out_of_fold: Bank out-of-fold predictions (for standardization/residuals).
        predict: Predictor of the full bank fit.
    """

    out_of_fold: FloatArray
    predict: collections.abc.Callable[[FloatArray], FloatArray]


def _make(kind: str, seed: int) -> typing.Any:
    if kind == "ridge":
        return sklearn.linear_model.RidgeCV(alphas=RIDGE_ALPHAS)
    return sklearn.ensemble.HistGradientBoostingRegressor(
        max_iter=GBM_ITERATIONS,
        learning_rate=0.05,
        early_stopping=True,  # pyright: ignore[reportArgumentType]
        validation_fraction=0.15,
        random_state=seed,
    )


def fit_stats_model(
    kind: str, x: FloatArray, y: FloatArray, folds: int, seed: int
) -> StatsModel:
    """Fits `kind` (`"ridge"` or `"gbm"`) of `y` on `x` with out-of-fold predictions."""
    oof = numpy.empty(y.shape[0])
    splitter = sklearn.model_selection.KFold(folds, shuffle=True, random_state=seed)
    for fit, held in splitter.split(x):
        oof[held] = _make(kind, seed).fit(x[fit], y[fit]).predict(x[held])
    model = _make(kind, seed).fit(x, y)
    return StatsModel(
        oof, lambda q: numpy.asarray(model.predict(q), dtype=numpy.float64)
    )


@dataclasses.dataclass(frozen=True)
class Standardizer:
    """Bank mean / sd of the transformed statistics."""

    mean: FloatArray
    sd: FloatArray

    def __call__(self, raw: FloatArray) -> FloatArray:
        """Log-transforms and standardizes raw statistics."""
        return (image_stats.transform(raw) - self.mean) / self.sd


def standardizer(bank_raw: FloatArray) -> Standardizer:
    """The bank standardizer of raw statistics."""
    t = image_stats.transform(bank_raw)
    return Standardizer(t.mean(axis=0), numpy.maximum(t.std(axis=0), 1e-12))


def _z(values: FloatArray, mean: float, sd: float) -> FloatArray:
    return (values - mean) / sd


def _bank_z(bank_values: FloatArray) -> tuple[float, float]:
    return float(bank_values.mean()), float(bank_values.std())


def ood_columns(
    shift_scores: dict[str, FloatArray],
    bank_state: ood_radius.BankStats,
    rho_query: FloatArray,
    x_query: FloatArray,
    x_bank: FloatArray,
    rho_bank: FloatArray,
    models: dict[str, StatsModel],
) -> dict[str, FloatArray]:
    """All score columns of one OOD set (pooled query `[id_test, ood]`).

    Args:
        shift_scores: The shift study's columns of this OOD set and run.
        bank_state: Bank constants of `ood_radius.bank_stats`.
        rho_query: Log-radius of the pooled query.
        x_query: Standardized statistics of the pooled query.
        x_bank: Standardized bank statistics.
        rho_bank: Bank log-radius.
        models: Statistics -> log-radius models by kind (`"ridge"`, `"gbm"`).

    Returns:
        Column name to `(N,)` scores, higher = more OOD (`BOOT_COLUMNS` and the
        per-statistic `STAT_DEV` / `STAT_LOW` columns).
    """
    direction = _z(
        shift_scores[KNN_DIRECTION], bank_state.direction_mean, bank_state.direction_sd
    )
    deviation = _z(
        shift_scores[DEVIATION], bank_state.deviation_mean, bank_state.deviation_sd
    )
    cov = sklearn.covariance.LedoitWolf().fit(x_bank)

    def mahalanobis(x: FloatArray) -> FloatArray:
        return numpy.sqrt(numpy.maximum(cov.mahalanobis(x), 0.0))

    m_mean, m_sd = _bank_z(mahalanobis(x_bank))
    mahal = _z(mahalanobis(x_query), m_mean, m_sd)

    columns: dict[str, FloatArray] = {
        RADIUS_HIGH: shift_scores[RADIUS_HIGH],
        DEVIATION: shift_scores[DEVIATION],
        RADIUS_LOW: -rho_query,
        KNN_DIRECTION: shift_scores[KNN_DIRECTION],
        COMBINED: shift_scores[COMBINED],
        POSTHOC_LOW: shift_scores[POSTHOC_LOW],
        STATS_MAHAL: mahal,
    }
    for kind, dev_name, dir_name in (
        ("gbm", RES_DEV, DIR_RES_DEV),
        ("ridge", RES_DEV_RIDGE, DIR_RES_DEV_RIDGE),
    ):
        model = models[kind]
        res_bank = rho_bank - model.out_of_fold
        r_mean, r_sd = _bank_z(res_bank)
        res = _z(rho_query - model.predict(x_query), r_mean, r_sd)
        d_mean, d_sd = _bank_z(numpy.abs((res_bank - r_mean) / r_sd))
        dev_std = _z(numpy.abs(res), d_mean, d_sd)
        columns[dev_name] = numpy.abs(res)
        columns[dir_name] = direction + dev_std
        if kind == "gbm":
            columns[RES_LOW] = -res
            columns[DIR_RES_LOW] = direction - res
            p_mean, p_sd = _bank_z(model.out_of_fold)
            pred = _z(model.predict(x_query), p_mean, p_sd)
            columns[STATS_PRED] = -pred
            columns[DIR_STATS_PRED] = direction - pred
    columns[DIR_STATS] = direction + mahal
    columns[DIR_STATS_RADIUS] = direction + mahal + deviation
    for i, name in enumerate(image_stats.NAMES):
        sign = float(numpy.sign(numpy.corrcoef(x_bank[:, i], rho_bank)[0, 1]))
        columns[STAT_DEV + name] = numpy.abs(x_query[:, i])
        columns[STAT_LOW + name] = -sign * x_query[:, i]
    return columns


# ------------------------------------------------------------------------- the unit


@dataclasses.dataclass
class Unit:
    """Everything one run contributes (arrays are over images).

    Attributes:
        name: Run name.
        lam: SIGReg weight.
        seed: Training seed.
        num_id: ID images (first rows of every OOD column).
        r2: `R2_NAMES` to held-out predictions of the STL-10 test log-radius, plus
            the key `"target"` with the true log-radius.
        rho: Set name to center-view log-radius of the set.
        scores: OOD set to column to `(N_id + N_ood,)` scores.
    """

    name: str
    lam: float
    seed: int
    num_id: int
    r2: dict[str, FloatArray]
    rho: dict[str, FloatArray]
    scores: dict[str, dict[str, FloatArray]]


def save_unit(path: str | os.PathLike[str], unit: Unit) -> None:
    """Writes a unit as an uncompressed `.npz` (atomically)."""
    arrays: dict[str, numpy.typing.NDArray[typing.Any]] = {
        "name": numpy.asarray(unit.name),
        "lam": numpy.asarray(unit.lam),
        "seed": numpy.asarray(unit.seed),
        "num_id": numpy.asarray(unit.num_id),
    }
    for key, values in unit.r2.items():
        arrays[f"r2::{key}"] = values
    for key, values in unit.rho.items():
        arrays[f"rho::{key}"] = values
    for ood_set, columns in unit.scores.items():
        for key, values in columns.items():
            arrays[f"x::{ood_set}::{key}"] = values
    partial = str(path) + ".partial"
    with open(partial, "wb") as f:
        numpy.savez(f, **arrays)  # pyright: ignore[reportArgumentType]
    os.replace(partial, path)


def load_unit(path: str | os.PathLike[str]) -> Unit:
    """Reads a unit written by `save_unit`."""
    r2: dict[str, FloatArray] = {}
    rho: dict[str, FloatArray] = {}
    scores: dict[str, dict[str, FloatArray]] = {}
    with numpy.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            if key.startswith("r2::"):
                r2[key[4:]] = arrays[key]
            elif key.startswith("rho::"):
                rho[key[5:]] = arrays[key]
            elif key.startswith("x::"):
                _, ood_set, name = key.split("::", 2)
                scores.setdefault(ood_set, {})[name] = arrays[key]
        return Unit(
            str(arrays["name"]),
            float(arrays["lam"]),
            int(arrays["seed"]),
            int(arrays["num_id"]),
            r2,
            rho,
            scores,
        )


def compute_unit(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    run_name: str,
    log: collections.abc.Callable[[str], None],
    study: Study = SVHN,
) -> Unit:
    """Computes the unit of one run from its center-view dumps and the statistics.

    The score columns of an OOD set the shift study did not cover (Places365) are
    computed here with `ood_radius.compute_scores`, exactly as the shift study does.

    Raises:
        FileNotFoundError: a dump, `shift_ood.npz` or the statistics file is missing.
    """
    run_dir = locations.runs_dir / run_name
    resolved = config.load_config(run_dir / "config.resolved.yaml")
    shift_unit = ood_radius.load_unit(run_dir / "shift_ood.npz")
    raw = load_image_stats(locations.runs_dir, study)
    z: dict[str, FloatArray] = {}
    for name in study.stat_sets:
        loaded = dumps.load_center_set(
            dumps.set_dir(locations.embed_dir, run_name, name)
        )
        if loaded is None:
            raise FileNotFoundError(f"{run_name}: no {name} embedding dump")
        if loaded.z.shape[0] != raw[name].shape[0] or not numpy.array_equal(
            loaded.image_id, numpy.arange(loaded.z.shape[0])
        ):
            raise ValueError(f"{name}: dump rows do not match the image arrays")
        z[name] = loaded.z.astype(numpy.float64)
    rho = {name: features.log_radius(z[name]) for name in study.stat_sets}
    std = standardizer(raw[dumps.STL10_TRAIN])
    x = {name: std(raw[name]) for name in study.stat_sets}
    seed, folds = cfg.seed, cfg.eval.cv_folds
    x_bank, rho_bank = x[dumps.STL10_TRAIN], rho[dumps.STL10_TRAIN]
    x_test, rho_test = x[dumps.STL10_TEST], rho[dumps.STL10_TEST]

    # (a) held-out R^2 on STL-10 test of models fitted on the bank.
    direction = randomization.fit_radius_model(
        z[dumps.STL10_TRAIN],
        z[dumps.STL10_TEST],
        folds,
        numpy.random.default_rng([seed, 1]),
    )
    models = {
        kind: fit_stats_model(kind, x_bank, rho_bank, folds, seed)
        for kind in ("ridge", "gbm")
    }
    r2: dict[str, FloatArray] = {
        "target": rho_test,
        R2_STATS_RIDGE: models["ridge"].predict(x_test),
        R2_STATS_GBM: models["gbm"].predict(x_test),
        R2_DIRECTION: direction.query,
    }
    resid = rho_bank - direction.bank
    for name, kind in ((R2_JOINT_RIDGE, "ridge"), (R2_JOINT_GBM, "gbm")):
        on_residual = fit_stats_model(kind, x_bank, resid, folds, seed)
        r2[name] = direction.query + on_residual.predict(x_test)
    log(f"{run_name}: R^2 models fitted")

    # (b)-(d) OOD columns.
    state = ood_radius.bank_stats(z[dumps.STL10_TRAIN], cfg.eval.knn_k)
    scores: dict[str, dict[str, FloatArray]] = {}
    for ood in study.ood_sets:
        shift_scores = shift_unit.scores.get(ood)
        if shift_scores is None:
            shift_scores = ood_radius.compute_scores(
                z[dumps.STL10_TRAIN],
                z[dumps.STL10_TEST],
                z[ood],
                cfg.eval.knn_k,
                cfg.seed,
            )
        scores[ood] = ood_columns(
            shift_scores,
            state,
            numpy.concatenate([rho_test, rho[ood]]),
            numpy.concatenate([x_test, x[ood]]),
            x_bank,
            rho_bank,
            models,
        )
        log(f"{run_name}: {ood} columns done")
    return Unit(
        run_name,
        resolved.loss.lambda_,
        resolved.seed,
        shift_unit.num_id,
        r2,
        {k: rho[k] for k in (dumps.STL10_TEST, *study.ood_sets)},
        scores,
    )


def run_unit(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    run_name: str,
    log: collections.abc.Callable[[str], None],
    study: Study = SVHN,
) -> pathlib.Path:
    """Computes and saves the unit file of one run; returns its path."""
    out = locations.runs_dir / run_name / study.unit_file
    save_unit(out, compute_unit(cfg, locations, run_name, log, study))
    return out


# ------------------------------------------------------------------------ aggregation


def r2_replicates(
    units: list[Unit], draws: list[IntArray], predictor: str
) -> FloatArray:
    """Seed-mean held-out `R^2` of `predictor`: element 0 full sample, then B."""

    def r2(y: FloatArray, pred: FloatArray) -> FloatArray:
        ss_res = numpy.square(y - pred).sum(axis=-1)
        ss_tot = numpy.square(y - y.mean(axis=-1, keepdims=True)).sum(axis=-1)
        return 1.0 - ss_res / ss_tot

    def one(seed_index: int, idx: IntArray) -> FloatArray:
        y, pred = units[seed_index].r2["target"], units[seed_index].r2[predictor]
        full = r2(y, pred)
        out = numpy.empty(idx.shape[0] + 1)
        out[0] = full
        for start in range(0, idx.shape[0], 250):
            block = idx[start : start + 250]
            out[1 + start : 1 + start + block.shape[0]] = r2(y[block], pred[block])
        return out

    return randomization.bootstrap_seed_mean(draws, one)


def _match_features(
    stats: dict[str, FloatArray], std: Standardizer
) -> dict[str, FloatArray]:
    columns = [image_stats.NAMES.index(n) for n in MATCH_NAMES]
    return {name: std(stats[name])[:, columns] for name in stats}


def nearest(
    query: FloatArray, reference: FloatArray, chunk: int = 1024
) -> tuple[IntArray, FloatArray]:
    """Index of and Euclidean distance to the nearest `reference` row, per query."""
    ref_sq = numpy.square(reference).sum(axis=1)
    index = numpy.empty(query.shape[0], dtype=numpy.intp)
    dist = numpy.empty(query.shape[0])
    for start in range(0, query.shape[0], chunk):
        q = query[start : start + chunk]
        d2 = numpy.square(q).sum(axis=1)[:, None] - 2.0 * q @ reference.T + ref_sq
        arg = d2.argmin(axis=1)
        index[start : start + chunk] = arg
        dist[start : start + chunk] = numpy.sqrt(
            numpy.maximum(d2[numpy.arange(q.shape[0]), arg], 0.0)
        )
    return index, dist


def capped_nearest(
    query: FloatArray,
    reference: FloatArray,
    cap: int,
    num_candidates: int = 64,
    chunk: int = 1024,
) -> IntArray:
    """Nearest `reference` row per query with at most `cap` queries per row.

    Greedy: queries are served in order of increasing distance to their nearest
    reference row; each takes its nearest row that still has capacity (among its
    `num_candidates` nearest, otherwise the nearest row with capacity overall).

    Raises:
        ValueError: `cap * len(reference) < len(query)`.
    """
    if cap * reference.shape[0] < query.shape[0]:
        raise ValueError("matching cap too small for the number of queries")
    ref_sq = numpy.square(reference).sum(axis=1)
    k = min(num_candidates, reference.shape[0])
    cand = numpy.empty((query.shape[0], k), dtype=numpy.intp)
    dist = numpy.empty((query.shape[0], k))
    for start in range(0, query.shape[0], chunk):
        q = query[start : start + chunk]
        d2 = numpy.square(q).sum(axis=1)[:, None] - 2.0 * q @ reference.T + ref_sq
        part = numpy.argpartition(d2, k - 1, axis=1)[:, :k]
        d_part = numpy.take_along_axis(d2, part, axis=1)
        order = numpy.argsort(d_part, axis=1, kind="stable")
        cand[start : start + chunk] = numpy.take_along_axis(part, order, axis=1)
        dist[start : start + chunk] = numpy.take_along_axis(d_part, order, axis=1)
    used = numpy.zeros(reference.shape[0], dtype=numpy.intp)
    out = numpy.empty(query.shape[0], dtype=numpy.intp)
    for i in numpy.argsort(dist[:, 0], kind="stable"):
        free = cand[i][used[cand[i]] < cap]
        if free.size:
            j = int(free[0])
        else:
            open_rows = numpy.flatnonzero(used < cap)
            d2 = numpy.square(reference[open_rows] - query[i]).sum(axis=1)
            j = int(open_rows[d2.argmin()])
        out[i] = j
        used[j] += 1
    return out


@dataclasses.dataclass(frozen=True)
class Matching:
    """Statistics-matched ID reference of one OOD set.

    Attributes:
        nearest_id: For each OOD image, the index of its nearest STL-10 test image.
        subset: Indices of the `SUBSET_FRACTION` STL-10 test images closest to the
            OOD set (smallest distance to their nearest OOD image).
        balance: Rows describing the balance of the matching statistics.
        max_multiplicity: Largest number of OOD images matched to one STL-10 image.
        num_unique: Number of distinct STL-10 test images used as matches.
    """

    nearest_id: IntArray
    subset: IntArray
    balance: list[Row]
    max_multiplicity: int
    num_unique: int


def build_matching(
    features_id: FloatArray,
    features_ood: FloatArray,
    lam: float,
    ood_set: str,
    max_matches: int | None = None,
) -> Matching:
    """Matches `features_ood` rows to STL-10 test rows (see module docstring).

    With `max_matches`, no STL-10 test image is matched more than that often
    (`capped_nearest`); otherwise matching is with unlimited replacement.
    """
    if max_matches is None:
        nearest_id, _ = nearest(features_ood, features_id)
    else:
        nearest_id = capped_nearest(features_ood, features_id, max_matches)
    _, to_ood = nearest(features_id, features_ood)
    size = max(1, int(round(SUBSET_FRACTION * features_id.shape[0])))
    subset = numpy.argsort(to_ood, kind="stable")[:size]
    multiplicity = numpy.bincount(nearest_id, minlength=features_id.shape[0])
    weights = multiplicity / multiplicity.sum()
    rows: list[Row] = []
    for j, name in enumerate(MATCH_NAMES):
        ood_mean = float(features_ood[:, j].mean())
        rows.append(
            {
                "lambda": lam,
                "OOD set": ood_set,
                "statistic (bank-standardized, log where applicable)": name,
                "STL-10 test": float(features_id[:, j].mean()),
                "closest 10% subset": float(features_id[subset, j].mean()),
                "matched reference (weighted)": float(weights @ features_id[:, j]),
                "OOD set mean": ood_mean,
                "std. mean difference, matched - OOD": (
                    float(weights @ features_id[:, j]) - ood_mean
                )
                / float(features_ood[:, j].std()),
                "std. mean difference, all STL-10 - OOD": (
                    float(features_id[:, j].mean()) - ood_mean
                )
                / float(features_ood[:, j].std()),
            }
        )
    rows.append(
        {
            "lambda": lam,
            "OOD set": ood_set,
            "statistic (bank-standardized, log where applicable)": "match counts",
            "STL-10 test": f"unique matches {int((multiplicity > 0).sum())}",
            "closest 10% subset": f"subset size {size}",
            "matched reference (weighted)": (
                f"max multiplicity {int(multiplicity.max())}"
            ),
        }
    )
    return Matching(
        nearest_id,
        subset,
        rows,
        int(multiplicity.max()),
        int((multiplicity > 0).sum()),
    )


@dataclasses.dataclass(frozen=True)
class MatchedResult:
    """Seed-mean matched-control statistics (arrays: element 0 = full sample).

    Attributes:
        gap: `mean(rho_ood) - mean(rho_id)`.
        matched_gap: `mean(rho_ood - rho_matched)`.
        closed: Fraction of the gap closed by the matching.
        auroc_all: AUROC of `-log r` against all STL-10 test images.
        auroc_matched: Same against the weighted matched reference.
        auroc_subset: Same against the closest-10% subset.
        balance: Balance rows.
        max_multiplicity: Largest number of OOD images matched to one STL-10 image.
        num_unique: Number of distinct STL-10 test images used as matches.
    """

    gap: FloatArray
    matched_gap: FloatArray
    closed: FloatArray
    auroc_all: FloatArray
    auroc_matched: FloatArray
    auroc_subset: FloatArray
    balance: list[Row]
    max_multiplicity: int
    num_unique: int


def _auroc_low(
    rho_ref: FloatArray, rho_ood: FloatArray, counts: FloatArray
) -> FloatArray:
    scores = -numpy.concatenate([rho_ref, rho_ood])[:, None]
    is_ood = numpy.arange(scores.shape[0]) >= rho_ref.shape[0]
    return ood_radius.roc_statistics(scores, is_ood, counts)[0][:, 0]


def _flat_counts(index: IntArray, size: int) -> FloatArray:
    """Row-wise `bincount` of `(B, n)` indices into `size` bins, `(B, size)`."""
    offset = index + numpy.arange(index.shape[0])[:, None] * size
    return (
        numpy.bincount(offset.ravel(), minlength=index.shape[0] * size)
        .reshape(index.shape[0], size)
        .astype(numpy.float64)
    )


def matched_replicates(
    units: list[Unit],
    ood_set: str,
    matching: Matching,
    num_bootstrap: int,
    rng: numpy.random.Generator,
    chunk: int = 125,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Seed-mean matched-control replicates (see `MatchedResult` for the order)."""
    n_id = units[0].rho[dumps.STL10_TEST].shape[0]
    n_ood = units[0].rho[ood_set].shape[0]
    n_sub = matching.subset.shape[0]
    parts: list[list[FloatArray]] = [[] for _ in range(5)]
    done = -1  # the first replicate is the full sample
    while done < num_bootstrap:
        size = min(chunk, num_bootstrap - max(done, 0))
        idx = rng.integers(0, n_ood, size=(size, n_ood))
        counts_all = ood_radius.stratified_counts(n_id, n_ood, size, rng)
        counts_sub = ood_radius.stratified_counts(n_sub, n_ood, size, rng)
        if done < 0:
            idx = numpy.concatenate([numpy.arange(n_ood)[None], idx])
            ones_all = numpy.ones((1, n_id + n_ood))
            ones_sub = numpy.ones((1, n_sub + n_ood))
            counts_all = numpy.concatenate([ones_all, counts_all])
            counts_sub = numpy.concatenate([ones_sub, counts_sub])
        counts_matched = numpy.concatenate(
            [
                _flat_counts(matching.nearest_id[idx], n_id),
                _flat_counts(idx, n_ood),
            ],
            axis=1,
        )
        per_seed: list[list[FloatArray]] = [[] for _ in range(5)]
        for unit in units:
            rho_id, rho_ood = unit.rho[dumps.STL10_TEST], unit.rho[ood_set]
            matched = rho_ood[idx] - rho_id[matching.nearest_id[idx]]
            for slot, value in enumerate(
                (
                    rho_ood[idx].mean(axis=1) - rho_id.mean(),
                    matched.mean(axis=1),
                    _auroc_low(rho_id, rho_ood, counts_all),
                    _auroc_low(rho_id, rho_ood, counts_matched),
                    _auroc_low(rho_id[matching.subset], rho_ood, counts_sub),
                )
            ):
                per_seed[slot].append(value)
        for slot in range(5):
            parts[slot].append(numpy.mean(per_seed[slot], axis=0))
        done = max(done, 0) + size
    gap, matched_gap, a_all, a_matched, a_subset = (numpy.concatenate(p) for p in parts)
    closed = 1.0 - matched_gap / gap
    return gap, matched_gap, closed, a_all, a_matched, a_subset


@dataclasses.dataclass(frozen=True)
class OodStats:
    """Seed-mean AUROC of every column (bootstrap columns and point estimates)."""

    num_id: int
    num_ood: int
    auroc: dict[str, FloatArray]
    point: dict[str, float]


def ood_replicates(
    units: list[Unit],
    ood_set: str,
    num_bootstrap: int,
    rng: numpy.random.Generator,
    chunk: int = 125,
) -> OodStats:
    """Seed-mean AUROC with paired image bootstrap (`ood_radius` machinery)."""
    sums: dict[str, FloatArray] = {}
    points: dict[str, float] = {}
    n_id = n_ood = 0
    for unit in units:
        columns = unit.scores[ood_set]
        n_id = unit.num_id
        total = columns[KNN_DIRECTION].shape[0]
        n_ood = total - n_id
        is_ood = numpy.arange(total) >= n_id
        boot = numpy.stack([columns[c] for c in BOOT_COLUMNS], axis=1)
        parts = [ood_radius.roc_statistics(boot, is_ood, numpy.ones((1, total)))[0]]
        done = 0
        while done < num_bootstrap:
            size = min(chunk, num_bootstrap - done)
            counts = ood_radius.stratified_counts(n_id, n_ood, size, rng)
            parts.append(ood_radius.roc_statistics(boot, is_ood, counts)[0])
            done += size
        auroc = numpy.concatenate(parts)
        for i, name in enumerate(BOOT_COLUMNS):
            sums[name] = sums.get(name, 0.0) + auroc[:, i] / len(units)
        stat_names = [c for c in columns if c.startswith((STAT_DEV, STAT_LOW))]
        stat_matrix = numpy.stack([columns[c] for c in stat_names], axis=1)
        stat_auroc = ood_radius.roc_statistics(
            stat_matrix, is_ood, numpy.ones((1, total))
        )[0][0]
        for i, name in enumerate(stat_names):
            points[name] = points.get(name, 0.0) + float(stat_auroc[i]) / len(units)
    return OodStats(n_id, n_ood, sums, points)


@dataclasses.dataclass(frozen=True)
class LambdaResult:
    """Everything aggregated for one lambda."""

    lam: float
    seeds: int
    r2: dict[str, FloatArray]
    ood: dict[str, OodStats]
    matched: dict[str, MatchedResult]


def evaluate(
    units: list[Unit],
    raw_stats: dict[str, FloatArray],
    num_bootstrap: int,
    rng: numpy.random.Generator,
    study: Study = SVHN,
) -> list[LambdaResult]:
    """Aggregates the units per lambda (seed means, paired image bootstrap)."""
    std = standardizer(raw_stats[dumps.STL10_TRAIN])
    match_features = _match_features(raw_stats, std)
    results: list[LambdaResult] = []
    for lam in sorted({u.lam for u in units}):
        group = sorted((u for u in units if u.lam == lam), key=lambda u: u.seed)
        n_test = group[0].r2["target"].shape[0]
        draws = randomization.bootstrap_draws(n_test, len(group), num_bootstrap, rng)
        r2 = {name: r2_replicates(group, draws, name) for name in R2_NAMES}
        ood: dict[str, OodStats] = {}
        matched: dict[str, MatchedResult] = {}
        for ood_set in study.ood_sets:
            ood[ood_set] = ood_replicates(group, ood_set, num_bootstrap, rng)
            matching = build_matching(
                match_features[dumps.STL10_TEST],
                match_features[ood_set],
                lam,
                ood_set,
                study.max_matches,
            )
            gap, mgap, closed, a_all, a_matched, a_subset = matched_replicates(
                group, ood_set, matching, num_bootstrap, rng
            )
            matched[ood_set] = MatchedResult(
                gap,
                mgap,
                closed,
                a_all,
                a_matched,
                a_subset,
                matching.balance,
                matching.max_multiplicity,
                matching.num_unique,
            )
        results.append(LambdaResult(lam, len(group), r2, ood, matched))
    return results


# ---------------------------------------------------------------------------- reports


def _ci(values: FloatArray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    lo, hi = numpy.quantile(values[1:], [tail, 1.0 - tail])
    return float(lo), float(hi)


def _fmt(values: FloatArray, digits: int = 3) -> str:
    lo, hi = _ci(values)
    return f"{values[0]:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def r2_rows(result: LambdaResult) -> list[Row]:
    """Held-out `R^2` per predictor and the comparisons with verdicts."""
    rows: list[Row] = []
    for name in R2_NAMES:
        boot = result.r2[name]
        lo, _ = _ci(boot)
        row: Row = {"lambda": result.lam, "predictor": name, "R^2 [95% CI]": _fmt(boot)}
        if name in (R2_STATS_RIDGE, R2_STATS_GBM):
            row["explains log-radius materially (lower bound > 0.05)"] = (
                "yes" if lo > R2_MARGIN else "no"
            )
        rows.append(row)
    return rows


R2_DIFFERENCES = (
    ("statistics (gradient boosting) - direction", R2_STATS_GBM, R2_DIRECTION),
    ("statistics (ridge) - direction", R2_STATS_RIDGE, R2_DIRECTION),
    (
        "direction + statistics (GB) - direction  (increment of statistics)",
        R2_JOINT_GBM,
        R2_DIRECTION,
    ),
    (
        "direction + statistics (ridge) - direction  (increment of statistics)",
        R2_JOINT_RIDGE,
        R2_DIRECTION,
    ),
)


def r2_difference_rows(result: LambdaResult) -> list[Row]:
    """`R^2` differences with the verdict at `R2_COMPARE_MARGIN`."""
    rows: list[Row] = []
    for label, a, b in R2_DIFFERENCES:
        diff = result.r2[a] - result.r2[b]
        rows.append(
            {
                "lambda": result.lam,
                "comparison": label,
                "R^2 difference [95% CI]": _fmt(diff),
                "verdict (margin 0.01)": difference_verdict(diff, R2_COMPARE_MARGIN),
            }
        )
    ratio = result.r2[R2_STATS_GBM] / result.r2[R2_DIRECTION]
    rows.append(
        {
            "lambda": result.lam,
            "comparison": "R^2(statistics, GB) / R^2(direction)",
            "R^2 difference [95% CI]": _fmt(ratio),
            "verdict (margin 0.01)": "descriptive",
        }
    )
    return rows


def score_rows(result: LambdaResult) -> list[Row]:
    """AUROC of every bootstrap column per OOD set, with the radius-alone reading."""
    rows: list[Row] = []
    for ood_set, stats in result.ood.items():
        for name in BOOT_COLUMNS:
            row: Row = {
                "lambda": result.lam,
                "OOD set": ood_set,
                "score": name,
                "AUROC [95% CI]": _fmt(stats.auroc[name]),
            }
            if name in (RADIUS_HIGH, DEVIATION, STATS_MAHAL, RES_DEV, RES_LOW):
                row["alone: reading"] = ood_radius.radius_verdict(stats.auroc[name])
            rows.append(row)
    return rows


def comparison_rows(result: LambdaResult) -> list[Row]:
    """The pre-registered AUROC differences with verdicts (margin 0.01)."""
    rows: list[Row] = []
    for ood_set, stats in result.ood.items():
        for label, a, b in COMPARISONS:
            diff = stats.auroc[a] - stats.auroc[b]
            rows.append(
                {
                    "lambda": result.lam,
                    "OOD set": ood_set,
                    "comparison (A - B)": label,
                    "AUROC difference [95% CI]": _fmt(diff),
                    "90% CI": "[{:+.3f}, {:+.3f}]".format(*_ci(diff, 0.90)),
                    "verdict": difference_verdict(diff, AUROC_MARGIN),
                }
            )
    return rows


def per_statistic_rows(result: LambdaResult) -> list[Row]:
    """Descriptive single-statistic AUROCs (point estimates, seed means)."""
    rows: list[Row] = []
    for ood_set, stats in result.ood.items():
        for name in image_stats.NAMES:
            rows.append(
                {
                    "lambda": result.lam,
                    "OOD set": ood_set,
                    "statistic": name,
                    "AUROC two-sided |z|": stats.point[STAT_DEV + name],
                    "AUROC oriented by ID (low-radius direction)": stats.point[
                        STAT_LOW + name
                    ],
                }
            )
    return rows


def matched_rows(result: LambdaResult) -> list[Row]:
    """Matched-control table per OOD set."""
    rows: list[Row] = []
    for ood_set, m in result.matched.items():
        rows.append(
            {
                "lambda": result.lam,
                "OOD set": ood_set,
                "log-radius gap OOD - STL-10 [95% CI]": _fmt(m.gap),
                "gap after matching [95% CI]": _fmt(m.matched_gap),
                "fraction of gap closed [95% CI]": _fmt(m.closed),
                "AUROC low radius vs all STL-10": _fmt(m.auroc_all),
                "vs matched reference": _fmt(m.auroc_matched),
                "matched reading": ood_radius.radius_verdict(m.auroc_matched),
                "vs closest-10% subset": _fmt(m.auroc_subset),
            }
        )
    return rows


def balance_rows(result: LambdaResult) -> list[Row]:
    """Balance of the matching statistics."""
    return [row for m in result.matched.values() for row in m.balance]


SMD_KEY = "std. mean difference, matched - OOD"


def validity_checks(
    result: LambdaResult, ood_set: str, max_matches: int | None
) -> bool:
    """Whether V1 to V3 of the Places365 pre-registration hold for `ood_set`."""
    mahal = float(result.ood[ood_set].auroc[STATS_MAHAL][0])
    matched = result.matched[ood_set]
    smd = [
        abs(float(row[SMD_KEY]))
        for row in matched.balance
        if isinstance(row.get(SMD_KEY), float)
    ]
    capped = max_matches is None or matched.max_multiplicity <= max_matches
    return (
        mahal <= VALID_MAX_MAHAL_AUROC
        and len(smd) == len(MATCH_NAMES)
        and max(smd) < VALID_MAX_ABS_SMD
        and capped
    )


def validity_rows(result: LambdaResult, study: Study) -> list[Row]:
    """The pre-registered validity checks V1 to V3 per OOD set."""
    rows: list[Row] = []
    for ood_set, stats in result.ood.items():
        matched = result.matched[ood_set]
        smd = {
            str(row["statistic (bank-standardized, log where applicable)"]): float(
                row[SMD_KEY]
            )
            for row in matched.balance
            if isinstance(row.get(SMD_KEY), float)
        }
        mahal = stats.auroc[STATS_MAHAL]
        rows.append(
            {
                "lambda": result.lam,
                "OOD set": ood_set,
                "V1 statistics Mahalanobis AUROC [95% CI] (needs <= 0.90)": _fmt(mahal),
                "V2 max |std. mean diff| after matching (needs < 0.25)": max(
                    abs(v) for v in smd.values()
                ),
                "V3 max matches per STL-10 image": matched.max_multiplicity,
                "V3 unique matches": matched.num_unique,
                "valid": "yes"
                if validity_checks(result, ood_set, study.max_matches)
                else "no",
            }
        )
    return rows


def overall_rows(result: LambdaResult, study: Study = SVHN) -> list[Row]:
    """The pre-registered overall reading per OOD set (decisive: `study.target`)."""
    rows: list[Row] = []
    for ood_set, stats in result.ood.items():
        a = stats.auroc
        shift_gain = a[COMBINED] - a[KNN_DIRECTION]
        resid_gain = a[DIR_RES_DEV] - a[KNN_DIRECTION]
        residual = difference_verdict(resid_gain, AUROC_MARGIN)
        radius_vs_stats = difference_verdict(a[COMBINED] - a[DIR_STATS], AUROC_MARGIN)
        matched = ood_radius.radius_verdict(result.matched[ood_set].auroc_matched)
        if (
            residual != A_HIGHER
            and radius_vs_stats in (EQUIVALENT, B_HIGHER)
            and matched == ood_radius.NOT_INFORMATIVE
        ):
            verdict = EXPLAINED
        elif (
            residual == A_HIGHER
            and radius_vs_stats == A_HIGHER
            and matched == ood_radius.INFORMATIVE
        ):
            verdict = INDEPENDENT
        else:
            verdict = PARTLY
        if (
            study.validity
            and ood_set == study.target
            and not validity_checks(result, ood_set, study.max_matches)
        ):
            verdict = INVALID
        rows.append(
            {
                "lambda": result.lam,
                "OOD set": ood_set,
                "(c) residual combination vs direction": residual,
                "(d) radius combination vs statistics combination": radius_vs_stats,
                "matched AUROC reading": matched,
                "AUROC gain of shift combination": _fmt(shift_gain),
                "AUROC gain after removing statistics": _fmt(resid_gain),
                "fraction of gain retained": float(resid_gain[0] / shift_gain[0])
                if shift_gain[0] != 0
                else float("nan"),
                "overall (pre-registered rule)": verdict,
            }
        )
    return rows


def figures(
    result: LambdaResult, study: Study = SVHN
) -> list[tuple[str, typing.Any, str]]:
    """`(file stem, figure, alt text)` of the three summary figures."""
    r2_values = {name: (float(v[0]), *_ci(v)) for name, v in result.r2.items()}
    target = result.ood[study.target].auroc
    auroc_values = {
        name: (float(target[name][0]), *_ci(target[name]))
        for name in (
            RADIUS_LOW,
            DEVIATION,
            STATS_MAHAL,
            STATS_PRED,
            KNN_DIRECTION,
            COMBINED,
            DIR_RES_DEV,
            DIR_STATS,
        )
    }
    matched = result.matched[study.target]
    matched_values = {
        "all STL-10 test": (float(matched.auroc_all[0]), *_ci(matched.auroc_all)),
        "matched reference": (
            float(matched.auroc_matched[0]),
            *_ci(matched.auroc_matched),
        ),
        "closest-10% subset": (
            float(matched.auroc_subset[0]),
            *_ci(matched.auroc_subset),
        ),
    }
    return [
        (
            "r2",
            plotting.bar_plot(
                r2_values,
                f"Held-out R^2 of log-radius on STL-10 test (lambda {result.lam:g})",
                "R^2",
            ),
            "R^2 of log-radius",
        ),
        (
            f"{study.name}_auroc",
            plotting.bar_plot(
                auroc_values,
                f"{study.label} OOD AUROC (lambda {result.lam:g})",
                "AUROC",
                reference=0.5,
            ),
            f"{study.label} AUROC by score",
        ),
        (
            f"{study.name}_matched",
            plotting.bar_plot(
                matched_values,
                f"AUROC of low radius, {study.label} vs reference "
                f"(lambda {result.lam:g})",
                "AUROC",
                reference=0.5,
            ),
            "matched control AUROC",
        ),
    ]


def sections(
    result: LambdaResult, cfg: config.ExperimentConfig, study: Study = SVHN
) -> dict[str, str]:
    """Generated note sections of one lambda."""
    prefix = f"analyze.{study.name}_control"
    cap = (
        "with unlimited replacement"
        if study.max_matches is None
        else f"with at most {study.max_matches} matches per STL-10 test image"
    )
    boot = (
        f"Seed means with 95% percentile intervals ({cfg.eval.num_bootstrap} image "
        "bootstrap replicates, ID and OOD images resampled separately, paired across "
        "scores). Rules and margins: docstring of `lejepa_norm.eval.svhn_control`."
    )
    out = {
        f"{prefix}_r2": (
            "### (a) Held-out R^2 of the ID log-radius (fit on STL-10 train, "
            "evaluated on STL-10 test)\n\n"
            + boot
            + "\n\n"
            + reporting.markdown_table(r2_rows(result))
            + "\n\n"
            + reporting.markdown_table(r2_difference_rows(result))
        ),
        f"{prefix}_auroc": (
            "### (b) AUROC of each score (higher = more OOD)\n\n"
            + boot
            + "\n\n"
            + reporting.markdown_table(score_rows(result))
        ),
        f"{prefix}_compare": (
            "### (b)-(d) Pre-registered AUROC differences\n\n"
            + boot
            + f" Margin {AUROC_MARGIN:g} AUROC: `{A_HIGHER}` iff the lower 95% bound "
            "exceeds the margin, `equivalent` iff the 90% interval lies inside "
            "(-0.01, +0.01), `B higher` iff the upper 95% bound is below -0.01.\n\n"
            + reporting.markdown_table(comparison_rows(result))
        ),
        f"{prefix}_matched": (
            "### Matched control\n\n"
            "Each OOD image is matched to its nearest STL-10 test image "
            + cap
            + " in the (mean luminance, contrast, log Laplacian variance) space.\n\n"
            + reporting.markdown_table(matched_rows(result))
            + "\n\nBalance of the matching statistics:\n\n"
            + reporting.markdown_table(balance_rows(result))
        ),
        f"{prefix}_overall": (
            "### Overall reading (pre-registered rule)\n\n"
            + reporting.markdown_table(overall_rows(result, study))
        ),
        f"{prefix}_single": (
            "### Single statistics (descriptive, point estimates)\n\n"
            + reporting.markdown_table(per_statistic_rows(result))
        ),
    }
    if study.validity:
        out[f"{prefix}_validity"] = (
            "### Validity of the control (pre-registered V1 to V3)\n\n"
            "The reading of the radius result is only valid where `valid` is `yes`: "
            f"V1 statistics Mahalanobis AUROC at most {VALID_MAX_MAHAL_AUROC:g}, "
            f"V2 absolute standardized mean difference below {VALID_MAX_ABS_SMD:g} on "
            "each matching statistic, V3 the matching cap holds. Properties of the "
            "data, identical for every lambda.\n\n"
            + reporting.markdown_table(validity_rows(result, study))
        )
    return out


def write_notes(
    cfg: config.ExperimentConfig,
    locations: paths.Paths,
    prefix: str,
    note: pathlib.Path,
    figures_on: bool,
    study: Study = SVHN,
) -> list[pathlib.Path]:
    """Aggregates the unit files of runs `prefix_*` into one note per lambda.

    The notes are `<note stem>_lam<lambda>.md` next to `note` (relative to the notes
    directory); returns their paths.
    """
    units: list[Unit] = []
    missing: list[Row] = []
    for run in sweep.load_runs(locations.runs_dir, prefix):
        unit_file = locations.runs_dir / run.name / study.unit_file
        if unit_file.is_file():
            units.append(load_unit(unit_file))
        else:
            missing.append(
                {"run": run.name, "lambda": run.lam, "missing": study.unit_file}
            )
    rng = numpy.random.default_rng(cfg.seed)
    results = evaluate(
        units,
        load_image_stats(locations.runs_dir, study),
        cfg.eval.num_bootstrap,
        rng,
        study,
    )
    base = locations.notes_dir / note
    written: list[pathlib.Path] = []
    for result in results:
        path = base.with_name(f"{base.stem}_lam{result.lam:g}{base.suffix}")
        mine = [r for r in missing if r["lambda"] == result.lam]
        reporting.update_section(
            path,
            f"analyze.{study.name}_control_inputs",
            "### Inputs\n\nRuns aggregated: "
            + ", ".join(f"`{u.name}`" for u in units if u.lam == result.lam)
            + ".\n\nUnit files missing for this lambda:\n\n"
            + (reporting.markdown_table(mine) if mine else "None."),
        )
        for name, content in sections(result, cfg, study).items():
            reporting.update_section(path, name, content)
        if figures_on:
            links: list[str] = []
            for stem, fig, alt in figures(result, study):
                out = plotting.export_png(
                    fig,
                    path.parent / f"{study.name}_control_lam{result.lam:g}_{stem}.png",
                    locations.kaleido_chrome,
                )
                links.append(reporting.image_link(out, path, alt))
            reporting.update_section(
                path,
                f"analyze.{study.name}_control_figures",
                "### Figures\n\n" + "\n\n".join(links),
            )
        written.append(path)
    return written
