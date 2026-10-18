"""All analyses of the Technical Design, computed from loaded embedding dumps.

`run_analyses` takes the evaluation sets of one run (see `dumps`) and returns
tables (lists of rows, ready for `reporting`) plus the arrays needed for
figures. It computes, for both `z` and `h`:

1. Marginal: kappa and radius moments (center view and augmented views).
2. ICC(1) of the log-radius over the K augmented views, with bootstrap CI.
3. Redundancy: held-out R^2 of `rho_bar` on `u_hat` (kernel ridge) vs. linear
   ridge and class one-hot baselines.
4. Difficulty: Spearman correlations (bootstrap CI) of each difficulty score
   with raw `rho_bar`, with the redundancy residual, and partial on class; mean
   paired radius shift vs. severity per corruption type.
5. Downstream: linear probe and k-NN accuracy on the embedding, `u_hat` and
   `[u_hat, rho_bar]`; selective-classification AURC with the radius as
   confidence (baseline: probe max-softmax); OOD AUROC with the radius as score;
   DTD retrieval mAP.

Image-level quantities (`rho_bar`, `u_hat`) for analyses 2-4 use the augmented
views, as defined in the design. Downstream results are reported for both the
center view and the augmented-view summaries. Analyses whose inputs are missing
(e.g. no `stl10c` set) are skipped. Kernel-based analyses run on a random subset
of at most `EvalConfig.max_kernel_images` images.
"""

import dataclasses
import statistics

import numpy
import numpy.typing

from lejepa_norm import config
from lejepa_norm.eval import difficulty
from lejepa_norm.eval import downstream
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import metrics
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
Row = dict[str, str | int | float]

# Table names, in the order they appear in the experiment note.
MARGINAL = "marginal"
ICC = "icc"
REDUNDANCY = "redundancy"
DIFFICULTY = "difficulty"
RADIUS_SHIFT = "radius_shift"
DOWNSTREAM = "downstream"
SELECTIVE = "selective"
OOD = "ood"
RETRIEVAL = "retrieval"
TABLES = (
    MARGINAL,
    ICC,
    REDUNDANCY,
    DIFFICULTY,
    RADIUS_SHIFT,
    DOWNSTREAM,
    SELECTIVE,
    OOD,
    RETRIEVAL,
)


@dataclasses.dataclass
class SpaceFigureData:
    """Arrays behind the figures of one space.

    Attributes:
        radius: Radii of all augmented views of the clean test images.
        dim: Embedding dimension.
        shift: Paired radius shift per (corruption, severity), if available.
        knn_margin: k-NN margin of the difficulty subset, if available.
        residual: Redundancy residual of the difficulty subset, if available.
    """

    radius: FloatArray
    dim: int
    shift: list[difficulty.ShiftRow] = dataclasses.field(
        default_factory=list[difficulty.ShiftRow]
    )
    knn_margin: FloatArray | None = None
    residual: FloatArray | None = None


@dataclasses.dataclass
class AnalysisResult:
    """Tables (by name, see `TABLES`) and figure data (by space)."""

    tables: dict[str, list[Row]]
    figures: dict[features.Space, SpaceFigureData]


def format_estimate(est: metrics.Estimate) -> str:
    """Formats `value [low, high]` with 3 decimals (no CI if not bootstrapped)."""
    if numpy.isnan(est.low):
        return f"{est.value:.3f}"
    return f"{est.value:.3f} [{est.low:.3f}, {est.high:.3f}]"


def _subsample(
    n: int, limit: int, rng: numpy.random.Generator
) -> numpy.typing.NDArray[numpy.int64]:
    if n <= limit:
        return numpy.arange(n)
    return numpy.sort(rng.choice(n, size=limit, replace=False))


def _difficulty_pool(
    test: shards.EmbeddingShard,
    corrupt: shards.EmbeddingShard | None,
    limit: int,
    rng: numpy.random.Generator,
) -> shards.EmbeddingShard:
    """Random subset of the clean test images plus their corrupted versions.

    Rows are drawn uniformly from the union before any array is copied, so the
    (large) STL-10-C set is never concatenated in full.
    """
    n_test = test.label.size
    n_total = n_test + (0 if corrupt is None else corrupt.label.size)
    pidx = _subsample(n_total, limit, rng)
    parts = [dumps.take(test, pidx[pidx < n_test])]
    if corrupt is not None:
        parts.append(dumps.take(corrupt, pidx[pidx >= n_test] - n_test))
    return dumps.concatenate(parts)


def _num_classes(*label_arrays: numpy.typing.NDArray[numpy.int64]) -> int:
    return int(max(int(a.max()) for a in label_arrays if a.size)) + 1


def _params_summary(params: tuple[dict[str, float], ...]) -> str:
    if not params:
        return ""
    keys = sorted(params[0])
    return ", ".join(f"{k}={statistics.median(p[k] for p in params):.3g}" for k in keys)


def _marginal(test: shards.EmbeddingShard, space: features.Space) -> list[Row]:
    x = features.space_array(test, space)
    rows: list[Row] = []
    for views in features.VIEW_SELECTIONS:
        sel = features.select_views(x, views)
        r = features.radius(sel)
        rows.append(
            {
                "space": space,
                "views": views,
                "d": sel.shape[-1],
                "kappa": metrics.kappa(sel),
                "mean_r": float(r.mean()),
                "std_r": float(r.std(ddof=1)),
                "mean_r_sq_over_d": float(numpy.square(r).mean() / sel.shape[-1]),
            }
        )
    return rows


def run_analyses(
    sets: dict[str, shards.EmbeddingShard],
    cfg: config.EvalConfig,
    seed: int = 0,
) -> AnalysisResult:
    """Runs every analysis whose inputs are present in `sets`.

    Args:
        sets: Evaluation sets keyed by `dumps` set name; `stl10_test` is
            required, the others enable further analyses.
        cfg: Analysis settings (k for k-NN, CV folds, bootstrap resamples,
            subset size for kernel methods).
        seed: Seed for subsampling, fold splits and bootstraps.

    Returns:
        The tables and figure data.

    Raises:
        KeyError: `stl10_test` is missing.
    """
    if dumps.STL10_TEST not in sets:
        raise KeyError(f"the {dumps.STL10_TEST!r} set is required")
    rng = numpy.random.default_rng(seed)
    nb = cfg.num_bootstrap
    test = sets[dumps.STL10_TEST]
    train = sets.get(dumps.STL10_TRAIN)
    corrupt = sets.get(dumps.STL10_C)
    tables: dict[str, list[Row]] = {name: [] for name in TABLES}
    figures: dict[features.Space, SpaceFigureData] = {}
    # Subsets are drawn once so that z and h are analyzed on the same images.
    idx = _subsample(test.label.size, cfg.max_kernel_images, rng)
    pool = _difficulty_pool(test, corrupt, cfg.max_kernel_images, rng)

    for space in features.SPACES:
        x_test = features.space_array(test, space)
        aug = features.summarize(x_test, "augmented")
        figures[space] = SpaceFigureData(radius=numpy.exp(aug.rho).ravel(), dim=aug.dim)

        # 1. Marginal.
        tables[MARGINAL].extend(_marginal(test, space))

        # 2. ICC over augmented views.
        icc = metrics.icc1(aug.rho, nb, rng)
        tables[ICC].append(
            {
                "space": space,
                "icc": format_estimate(icc.icc),
                "msb": icc.msb,
                "msw": icc.msw,
                "K": icc.num_views,
                "N": icc.icc.n,
            }
        )

        # 3. Redundancy of rho_bar with u_hat on clean test images.
        fits = {
            "kernel ridge (u_hat)": redundancy.kernel_ridge_cv(
                aug.u_hat[idx], aug.rho_bar[idx], "exp_dot", cfg.cv_folds, rng
            ),
            "linear ridge (u_hat)": redundancy.kernel_ridge_cv(
                aug.u_hat[idx], aug.rho_bar[idx], "linear", cfg.cv_folds, rng
            ),
            "class one-hot": redundancy.group_mean_cv(
                test.label[idx], aug.rho_bar[idx], cfg.cv_folds, rng
            ),
        }
        for model, fit in fits.items():
            tables[REDUNDANCY].append(
                {
                    "space": space,
                    "model": model,
                    "r2": fit.r2,
                    "fold_r2_std": float(numpy.std(fit.fold_r2)),
                    "N": int(idx.size),
                    "params (median over folds)": _params_summary(fit.params),
                }
            )

        # 4b. Paired radius shift vs. severity.
        if corrupt is not None:
            delta = difficulty.paired_radius_shift(
                aug.rho_bar,
                test.image_id,
                features.mean_log_radius(features.space_array(corrupt, space)),
                corrupt.image_id,
            )
            shift = difficulty.shift_by_severity(
                delta, corrupt.corruption, corrupt.severity, nb, rng
            )
            figures[space].shift = shift
            tables[RADIUS_SHIFT].extend(
                {
                    "space": space,
                    "corruption": row.corruption,
                    "severity": row.severity,
                    "mean delta_rho": format_estimate(row.mean),
                    "N": row.mean.n,
                }
                for row in shift
            )

        if train is None:
            continue
        x_train = features.space_array(train, space)
        num_classes = _num_classes(train.label, test.label)

        # 4a. Difficulty correlations on clean (+ corrupted) test images.
        x_pool = features.space_array(pool, space)
        query = features.summarize(x_pool, "augmented")
        bank = features.summarize(x_train, "augmented")
        scores = difficulty.scores(
            bank,
            train.label,
            query,
            pool.label,
            x_pool,
            pool.severity,
            cfg.knn_k,
            num_classes,
        )
        residual = redundancy.kernel_ridge_cv(
            query.u_hat, query.rho_bar, "exp_dot", cfg.cv_folds, rng
        ).residuals
        figures[space].knn_margin = scores["knn_margin"]
        figures[space].residual = residual
        for name, score in scores.items():
            if numpy.ptp(score) == 0:
                continue  # e.g. severity without corrupted images
            tables[DIFFICULTY].append(
                {
                    "space": space,
                    "score": name,
                    "spearman(rho_bar)": format_estimate(
                        metrics.spearman(query.rho_bar, score, nb, rng)
                    ),
                    "spearman(residual)": format_estimate(
                        metrics.spearman(residual, score, nb, rng)
                    ),
                    "partial spearman(rho_bar | class)": format_estimate(
                        metrics.partial_spearman(
                            query.rho_bar, score, pool.label, nb, rng
                        )
                    ),
                    "N": int(pool.label.size),
                }
            )

        # 5. Downstream, for center and augmented-view summaries.
        for views in features.VIEW_SELECTIONS:
            bank_v = features.summarize(x_train, views)
            test_v = features.summarize(x_test, views)
            for kind in downstream.FEATURE_KINDS:
                xb, xq = downstream.feature_pair(bank_v, test_v, kind)
                probe = downstream.LinearProbe(xb, train.label, num_classes)
                votes = downstream.knn_votes(
                    xb, train.label, xq, cfg.knn_k, num_classes
                )
                tables[DOWNSTREAM].append(
                    {
                        "space": space,
                        "views": views,
                        "features": kind,
                        "linear probe acc": downstream.accuracy(
                            probe.predict(xq), test.label
                        ),
                        "k-NN acc": downstream.accuracy(
                            numpy.argmax(votes, axis=1), test.label
                        ),
                    }
                )
            probe = downstream.LinearProbe(bank_v.u_hat, train.label, num_classes)
            probs = probe.predict_proba(test_v.u_hat)
            errors = numpy.argmax(probs, axis=1) != test.label
            tables[SELECTIVE].append(
                {
                    "space": space,
                    "views": views,
                    "probe error": float(errors.mean()),
                    "AURC (radius)": metrics.aurc(test_v.rho_bar, errors),
                    "AURC (max softmax)": metrics.aurc(probs.max(axis=1), errors),
                    "AURC (-radius)": metrics.aurc(-test_v.rho_bar, errors),
                }
            )

    # 5b. OOD detection and DTD retrieval do not need the bank.
    for space in features.SPACES:
        for views in features.VIEW_SELECTIONS:
            test_v = features.summarize(features.space_array(test, space), views)
            for ood_name in dumps.OOD_SETS:
                ood = sets.get(ood_name)
                if ood is None:
                    continue
                ood_v = features.summarize(features.space_array(ood, space), views)
                tables[OOD].append(
                    {
                        "space": space,
                        "OOD set": ood_name,
                        "views": views,
                        "AUROC (radius, in-dist positive)": metrics.auroc(
                            numpy.concatenate([test_v.rho_bar, ood_v.rho_bar]),
                            numpy.arange(test_v.rho_bar.size + ood_v.rho_bar.size)
                            < test_v.rho_bar.size,
                        ),
                    }
                )
            dtd = sets.get(dumps.DTD_TEST)
            if dtd is not None:
                dtd_v = features.summarize(features.space_array(dtd, space), views)
                for kind in ("u_hat", "u_hat+rho"):
                    feats, _ = downstream.feature_pair(dtd_v, dtd_v, kind)
                    tables[RETRIEVAL].append(
                        {
                            "space": space,
                            "views": views,
                            "features": kind,
                            "DTD mAP": metrics.retrieval_map(feats, dtd.label),
                        }
                    )
    return AnalysisResult(tables=tables, figures=figures)
