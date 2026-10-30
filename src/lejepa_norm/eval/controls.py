"""Controls for the redundancy result: how much of it is real, how much artifact?

`redundancy` regresses the image-level mean log-radius `rho_bar` on the image-level
mean direction `u_hat`, both computed from the same augmented views. This module
asks, on the same images and with the same nested-CV kernel ridge:

* **Permutation null.** Shuffle the radius across images (directions untouched)
  and refit through the identical pipeline, hyperparameter search included. The
  null `R^2` should sit at 0; the p-value is `(1 + #{null >= observed}) / (1 + B)`.
* **Cross-view.** Predict the radius from directions of *other* views, so that no
  view contributes to both sides: first half of the views to second half and back,
  and a single view to another single view (and to itself, for reference).
* **F-test.** The classical overall F-test of the linear regression (linear
  dependence only; in-sample), next to the held-out numbers.
* **Reliability.** The squared correlation between the two half-view means of
  `rho_bar`: a proxy for the part of the radius that is stable across views, hence
  an upper limit for the cross-view `R^2` of the half-to-half setups.
"""

import dataclasses
import re

import numpy
import numpy.typing

from lejepa_norm import config
from lejepa_norm.eval import analysis
from lejepa_norm.eval import features
from lejepa_norm.eval import redundancy
from lejepa_norm.eval import shards

FloatArray = numpy.typing.NDArray[numpy.float64]
Row = analysis.Row

RELIABILITY_SETUP = "reference: split-half reliability r^2 of rho_bar"


@dataclasses.dataclass(frozen=True)
class Setup:
    """One regression of a radius target on a direction input.

    Attributes:
        name: Row label.
        u: Inputs, unit vectors `(N, d)`.
        rho: Target, `(N,)`.
    """

    name: str
    u: FloatArray
    rho: FloatArray


def view_setups(x: FloatArray) -> list[Setup]:
    """The regression setups of one space.

    Args:
        x: Augmented-view embeddings, float64 `(N, K, d)` with `K >= 2`.

    Returns:
        Same views (all K both sides), the two half-to-half cross-view setups
        (only if `K >= 4`), and single-view setups.
    """
    num_views = x.shape[1]
    rho = features.log_radius(x)
    unit = features.direction(x)

    def u_hat(views: list[int]) -> FloatArray:
        return features.direction(unit[:, views].mean(axis=1))

    everything = list(range(num_views))
    setups = [
        Setup(
            f"same views (all {num_views}): rho_bar ~ u_hat",
            u_hat(everything),
            rho.mean(axis=1),
        )
    ]
    half = num_views // 2
    first, second = everything[:half], everything[half:]
    if half >= 2:
        setups += [
            Setup(
                f"disjoint halves: rho_bar(views {second[0]}-{second[-1]})"
                f" ~ u_hat(views {first[0]}-{first[-1]})",
                u_hat(first),
                rho[:, second].mean(axis=1),
            ),
            Setup(
                f"disjoint halves: rho_bar(views {first[0]}-{first[-1]})"
                f" ~ u_hat(views {second[0]}-{second[-1]})",
                u_hat(second),
                rho[:, first].mean(axis=1),
            ),
        ]
    setups += [
        Setup("single view, same view: rho_0 ~ u_0", unit[:, 0], rho[:, 0]),
        Setup("single view, other view: rho_1 ~ u_0", unit[:, 0], rho[:, 1]),
    ]
    return setups


def split_half_reliability(x: FloatArray) -> float:
    """Squared correlation between the mean log-radius of the two view halves."""
    rho = features.log_radius(x)
    half = x.shape[1] // 2
    a, b = rho[:, :half].mean(axis=1), rho[:, half:].mean(axis=1)
    return float(numpy.corrcoef(a, b)[0, 1] ** 2)


def run_setup(
    setup: Setup,
    cfg: config.EvalConfig,
    rng: numpy.random.Generator,
) -> Row:
    """Held-out `R^2`, permutation null and F-test of one setup."""
    return run_setup_detailed(setup, cfg, rng)[0]


def run_setup_detailed(
    setup: Setup,
    cfg: config.EvalConfig,
    rng: numpy.random.Generator,
) -> tuple[Row, redundancy.RegressionResult]:
    """`run_setup` plus the held-out predictions behind its `R^2` (for bootstraps)."""
    model = redundancy.NestedKernelRidge(setup.u, "exp_dot", cfg.cv_folds, rng)
    test = redundancy.permutation_test(model, setup.rho, cfg.num_permutations, rng)
    ftest = redundancy.f_test(setup.u, setup.rho)
    null = test.null
    row: Row = {
        "setup": setup.name,
        "N": setup.rho.size,
        "r2 (nested CV, kernel)": test.observed,
        "null mean": float(null.mean()) if null.size else float("nan"),
        "null sd": float(null.std(ddof=1)) if null.size > 1 else float("nan"),
        "null max": float(null.max()) if null.size else float("nan"),
        "permutations": int(null.size),
        "permutation p": test.p_value,
        "F": ftest.f_statistic,
        "df": f"{ftest.df1}, {ftest.df2}",
        "F p": ftest.p_value,
        "r2 (in-sample, linear)": ftest.r2,
    }
    return row, model.fit(setup.rho)


def run_redundancy_controls(
    test: shards.EmbeddingShard,
    cfg: config.EvalConfig,
    seed: int = 0,
) -> list[Row]:
    """Runs all controls on the clean test images, for `z` and `h`.

    The image subset is the one `analysis.run_analyses` uses for the same `seed`
    (the first draw of the generator), so the same-views row reproduces the
    redundancy table up to the fold split.

    Args:
        test: The `stl10_test` shard (needs at least two augmented views).
        cfg: Analysis settings (`max_kernel_images`, `cv_folds`, `num_permutations`).
        seed: Seed for the subset, fold splits and permutations.

    Returns:
        One row per space and setup, followed by one reliability row per space.
    """
    rng = numpy.random.default_rng(seed)
    idx = analysis.subsample(test.label.size, cfg.max_kernel_images, rng)
    rows: list[Row] = []
    for space in features.SPACES:
        x = features.select_views(features.space_array(test, space), "augmented")[idx]
        if x.shape[1] < 2:
            raise ValueError("the controls need at least two augmented views")
        for setup in view_setups(x):
            rows.append({"space": space, **run_setup(setup, cfg, rng)})
        rows.append(
            {
                "space": space,
                "setup": RELIABILITY_SETUP,
                "r2 (nested CV, kernel)": split_half_reliability(x),
            }
        )
    return rows


R2 = "r2 (nested CV, kernel)"
_RUN_NAME = re.compile(r"\(run `([^`]+)`\)")


def run_name(section: str) -> str:
    """Run name from the heading of a generated controls section."""
    match = _RUN_NAME.search(section)
    if match is None:
        raise ValueError("no run name in the controls section heading")
    return match.group(1)


def _mean_std(values: list[float]) -> str:
    arr = numpy.asarray(values)
    if arr.size < 2:
        return f"{arr.mean():.4g}"
    return f"{arr.mean():.4g} ± {arr.std(ddof=1):.2g}"


def aggregate_controls(runs: dict[float, list[list[dict[str, str]]]]) -> list[Row]:
    """Aggregates the controls tables of several seeds per lambda.

    Args:
        runs: For each lambda, one parsed controls table (cells as text, as read
            back from a note) per seed. All tables must list the same rows.

    Returns:
        One row per lambda, space and setup: held-out `R^2` as mean ± std over
        seeds, the largest permutation-null maximum, the worst (largest)
        permutation p-value and the smallest F statistic over seeds. Reliability
        rows only have the `R^2` column.

    Raises:
        ValueError: The seeds of a lambda list different rows.
    """
    rows: list[Row] = []
    for lam, tables in sorted(runs.items()):
        keys = [(r["space"], r["setup"]) for r in tables[0]]
        for table in tables[1:]:
            if [(r["space"], r["setup"]) for r in table] != keys:
                raise ValueError(f"lambda={lam}: seeds list different rows")
        for i, (space, setup) in enumerate(keys):
            cells = [table[i] for table in tables]
            row: Row = {
                "lambda": lam,
                "space": space,
                "setup": setup,
                "seeds": len(tables),
                R2: _mean_std([float(c[R2]) for c in cells]),
            }
            if setup != RELIABILITY_SETUP:
                row["null max (max over seeds)"] = max(
                    float(c["null max"]) for c in cells
                )
                row["worst permutation p"] = max(
                    float(c["permutation p"]) for c in cells
                )
                row["min F"] = min(float(c["F"]) for c in cells)
            rows.append(row)
    return rows
