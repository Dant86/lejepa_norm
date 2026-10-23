"""Synthetic study: how much does SIGReg see of a loosely constrained radius?

For embeddings `z = r u` with a Gaussian-like direction `u` (uniform on the
sphere) and a radius whose spread is measured by `kappa = Var(||z||^2) / (2 d)`
(1 for `N(0, I_d)`), how large are SIGReg's value, its gradient on the radius and
its power to detect `kappa > 1`, compared with a direct radial test?

Radial profiles (both have `E r^2 = d` and `Var(r^2) = 2 d kappa` exactly):

* `scaled_chi`: `r^2 = (d / nu) chi^2_nu` with `nu = d / kappa`, i.e. `r` is a scaled
  `chi_nu`; `kappa = 1` is exactly `chi_d`, so `z ~ N(0, I_d)` and it is the null.
* `lognormal`: `log r` normal with `exp(4 sigma^2) - 1 = 2 kappa / d`. At `kappa = 1`
  this is *not* chi_d (the same two moments, a different shape).

**Moment relation.** With `u` uniform on the sphere, `E u_1^2 = 1 / d` and
`E u_1^4 = 3 / (d (d + 2))`, so the kurtosis of a 1-D projection `a^T z = r u_a` is

    3 E[r^4] / E[r^2]^2 * d / (d + 2) = 3 (d + 2 kappa) / (d + 2)    (E r^2 = d),

exactly 3 at `kappa = 1`. Dropping the factor `d / (d + 2)` (treating the projection
as an exact Gaussian scale mixture) gives `3 (1 + 2 kappa / d)`, which is 3.023 at
`kappa = 1` and 3.469 (not 3.442) at `kappa = 20`, `d = 256`: the excess kurtosis is
overstated by about 6% (0.469 against 0.442). See `projection_kurtosis`.
"""

# SciPy ships no type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import collections.abc
import dataclasses
import json
import logging
import math
import os
import pathlib
import typing
import zlib

import numpy
import numpy.typing
import scipy.special
import torch

from lejepa_norm import config
from lejepa_norm.eval import dumps
from lejepa_norm.eval import metrics
from lejepa_norm.eval import shards
from lejepa_norm.losses import sigreg

FloatArray = numpy.typing.NDArray[numpy.float64]

Family = typing.Literal["scaled_chi", "lognormal"]
FAMILIES: tuple[Family, ...] = ("scaled_chi", "lognormal")
KAPPAS = (1.0, 5.0, 15.0, 20.0, 26.0)
LEVEL = 0.05
LOG_EVERY = 25

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class StudyConfig:
    """Sizes of the synthetic study (defaults are the training setup).

    Attributes:
        dim: Embedding dimension `d`.
        batch_size: Samples per view and call (the per-view `N` of SIGReg).
        num_slices: Random directions per SIGReg call.
        num_points: Quadrature nodes of SIGReg.
        t_max: Integration range of SIGReg.
        num_views: Views averaged by the training-time SIGReg (independent
            synthetic samples here, which is optimistic for power: real views of
            the same images are correlated).
        reps: Repetitions per cell (each redraws samples and slice directions).
        grad_reps: Repetitions for the gradient decomposition.
        seed: Seed of both random generators.
    """

    dim: int = 256
    batch_size: int = 256
    num_slices: int = 1024
    num_points: int = 17
    t_max: float = 3.0
    num_views: int = 6
    reps: int = 200
    grad_reps: int = 20
    seed: int = 0


def study_config(cfg: config.ExperimentConfig) -> StudyConfig:
    """The study sizes implied by a training config (and its eval section)."""
    return StudyConfig(
        dim=cfg.model.embed_dim,
        batch_size=cfg.train.batch_size,
        num_slices=cfg.loss.num_slices,
        num_points=cfg.loss.num_points,
        t_max=cfg.loss.t_max,
        num_views=cfg.data.num_global_views + cfg.data.num_local_views,
        reps=cfg.eval.power_reps,
        grad_reps=cfg.eval.power_grad_reps,
        seed=cfg.seed,
    )


def projection_kurtosis(kappa: float, dim: int) -> float:
    """Exact kurtosis of a unit-direction projection of `r u`, with `E r^2 = d`.

    Args:
        kappa: `Var(||z||^2) / (2 d)`.
        dim: Dimension `d`.

    Returns:
        `3 (d + 2 kappa) / (d + 2)`; equals 3 at `kappa = 1`.
    """
    return 3.0 * (dim + 2.0 * kappa) / (dim + 2.0)


def sample_radius(
    family: Family, kappa: float, dim: int, n: int, rng: numpy.random.Generator
) -> FloatArray:
    """Draws `n` radii with `E r^2 = d` and `Var(r^2) = 2 d kappa`.

    Args:
        family: Radial profile.
        kappa: Target `Var(r^2) / (2 d)`; positive.
        dim: Dimension `d`.
        n: Number of radii.
        rng: Random generator.

    Returns:
        Float64 `(n,)`.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if family == "scaled_chi":
        nu = dim / kappa
        return numpy.sqrt(dim / nu * rng.chisquare(nu, size=n))
    sigma2 = math.log1p(2.0 * kappa / dim) / 4.0
    mu = 0.5 * math.log(dim) - sigma2
    return numpy.exp(mu + math.sqrt(sigma2) * rng.standard_normal(n))


def sample_embeddings(
    family: Family | None,
    kappa: float,
    dim: int,
    shape: tuple[int, ...],
    rng: numpy.random.Generator,
) -> torch.Tensor:
    """Draws `z = r u`, `u` uniform on the sphere; `family=None` gives `N(0, I)`.

    Args:
        family: Radial profile, or None for exact standard Gaussians.
        kappa: Radial spread (ignored when `family` is None).
        dim: Dimension `d`.
        shape: Leading shape, e.g. `(views, batch)`; the result appends `dim`.
        rng: Random generator.

    Returns:
        Float32 tensor `(*shape, dim)`.
    """
    g = rng.standard_normal((*shape, dim))
    if family is None:
        return torch.from_numpy(g.astype(numpy.float32))
    u = g / numpy.linalg.norm(g, axis=-1, keepdims=True)
    r = sample_radius(family, kappa, dim, int(numpy.prod(shape)), rng)
    return torch.from_numpy((u * r.reshape(shape)[..., None]).astype(numpy.float32))


def _sigreg(cfg: StudyConfig, generator: torch.Generator) -> sigreg.SIGReg:
    return sigreg.SIGReg(cfg.num_slices, cfg.num_points, cfg.t_max, generator)


def sigreg_values(
    family: Family | None,
    kappa: float,
    num_views: int,
    cfg: StudyConfig,
    rng: numpy.random.Generator,
    generator: torch.Generator,
    reps: int | None = None,
) -> FloatArray:
    """SIGReg values over repetitions (fresh samples and slices every time).

    Args:
        family: Radial profile, or None for the exact-Gaussian null.
        kappa: Radial spread.
        num_views: Independent views per call (the loss averages over views).
        cfg: Study sizes.
        rng: Sample generator.
        generator: Slice-direction generator of the SIGReg instance; its device
            is where SIGReg runs.
        reps: Overrides `cfg.reps`.

    Returns:
        Float64 `(reps,)`.
    """
    sig = _sigreg(cfg, generator)
    total = cfg.reps if reps is None else reps
    out: list[float] = []
    with torch.no_grad():
        for rep in range(total):
            z = sample_embeddings(
                family, kappa, cfg.dim, (num_views, cfg.batch_size), rng
            )
            out.append(float(sig(z.to(generator.device))))
            if (rep + 1) % LOG_EVERY == 0:
                logger.info("SIGReg, %d view(s): %d/%d reps", num_views, rep + 1, total)
    return numpy.asarray(out, dtype=numpy.float64)


def ks_statistic(radius: FloatArray, dim: int) -> float:
    """Kolmogorov-Smirnov distance between samples of `r` and the `chi_d` law."""
    r = numpy.sort(numpy.asarray(radius, dtype=numpy.float64))
    n = r.size
    cdf = scipy.special.gammainc(dim / 2.0, r**2 / 2.0)
    rank = numpy.arange(1, n + 1)
    return float(max((rank / n - cdf).max(), (cdf - (rank - 1) / n).max()))


def ks_values(
    family: Family | None,
    kappa: float,
    cfg: StudyConfig,
    rng: numpy.random.Generator,
) -> FloatArray:
    """KS distances to `chi_d` of `cfg.reps` batches of `batch_size` radii."""
    out = []
    for _ in range(cfg.reps):
        z = sample_embeddings(family, kappa, cfg.dim, (cfg.batch_size,), rng)
        out.append(ks_statistic(numpy.linalg.norm(z.numpy(), axis=-1), cfg.dim))
    return numpy.asarray(out, dtype=numpy.float64)


def power(null: FloatArray, alternative: FloatArray, level: float = LEVEL) -> float:
    """Fraction of `alternative` above the `1 - level` quantile of `null`."""
    return float(numpy.mean(alternative > numpy.quantile(null, 1.0 - level)))


def split_gradient(z: FloatArray, grad: FloatArray) -> dict[str, float]:
    """Splits a gradient `dL/dz` into radial and tangential parts per sample.

    Args:
        z: Samples `(N, d)`.
        grad: Gradient of a scalar with respect to `z`, same shape.

    Returns:
        `radial_rms`: root mean square of the radial component `g . u` (one
        degree of freedom per sample); `tangential_rms`: root mean square per
        tangential coordinate (`d - 1` per sample); `norm_ratio`: Frobenius norm of
        the radial part over that of the tangential part; `restoring_corr`:
        correlation of the radial component with `r - mean(r)` (positive means a
        descent step shrinks the spread of the radii).
    """
    n, d = z.shape
    r = numpy.linalg.norm(z, axis=-1)
    u = z / r[:, None]
    g_r = (grad * u).sum(axis=-1)
    g_tan = grad - g_r[:, None] * u
    radial_sq = float((g_r**2).sum())
    tangential_sq = float((g_tan**2).sum())
    corr = float(numpy.corrcoef(g_r, r)[0, 1]) if r.std() > 0 else float("nan")
    return {
        "radial_rms": math.sqrt(radial_sq / n),
        "tangential_rms": math.sqrt(tangential_sq / (n * (d - 1))),
        "norm_ratio": math.sqrt(radial_sq / tangential_sq),
        "restoring_corr": corr,
    }


def gradient_stats(
    family: Family | None,
    kappa: float,
    cfg: StudyConfig,
    rng: numpy.random.Generator,
    generator: torch.Generator,
) -> dict[str, float]:
    """Mean over `cfg.grad_reps` of `split_gradient` of single-view SIGReg."""
    sig = _sigreg(cfg, generator)
    rows: list[dict[str, float]] = []
    for _ in range(cfg.grad_reps):
        z = sample_embeddings(family, kappa, cfg.dim, (cfg.batch_size,), rng)
        z = z.to(generator.device).requires_grad_(True)
        sig(z).backward()  # pyright: ignore[reportUnknownMemberType]
        assert z.grad is not None
        rows.append(
            split_gradient(
                z.detach().cpu().numpy().astype(numpy.float64),
                z.grad.cpu().numpy().astype(numpy.float64),
            )
        )
    return {k: float(numpy.mean([row[k] for row in rows])) for k in rows[0]}


def kurtosis_rows(
    cfg: StudyConfig, rng: numpy.random.Generator, size: int = 20000
) -> list[dict[str, str | float]]:
    """Claimed vs exact projection kurtosis, with a Monte Carlo check.

    Args:
        cfg: Study sizes (only `dim` is used).
        rng: Random generator.
        size: Samples of `z` for the Monte Carlo estimate (one random unit
            direction per sample, so the draws are independent).

    Returns:
        One row per family and kappa.
    """
    rows: list[dict[str, str | float]] = []
    for family in FAMILIES:
        for kappa in KAPPAS:
            z = sample_embeddings(family, kappa, cfg.dim, (size,), rng).numpy()
            a = rng.standard_normal(z.shape)
            a /= numpy.linalg.norm(a, axis=-1, keepdims=True)
            x = (z * a).sum(axis=-1).astype(numpy.float64)
            x = x - x.mean()
            rows.append(
                {
                    "family": family,
                    "kappa": kappa,
                    "claimed 3(1+2k/d)": 3.0 * (1.0 + 2.0 * kappa / cfg.dim),
                    "exact 3(d+2k)/(d+2)": projection_kurtosis(kappa, cfg.dim),
                    "Monte Carlo": float(numpy.mean(x**4) / numpy.mean(x**2) ** 2),
                }
            )
    return rows


@dataclasses.dataclass(frozen=True)
class SyntheticStudy:
    """Results of the synthetic part, as table rows and plot series.

    Attributes:
        power_rows: SIGReg value and power by family, kappa and number of views,
            plus the KS rows.
        gradient_rows: Gradient decomposition by family and kappa.
        null_mean: Mean single-view SIGReg under N(0, I).
        null_std: Its standard deviation.
        null_views: `(mean, std)` of SIGReg averaged over `cfg.num_views`.
        num_views: Views of the multi-view rows.
    """

    power_rows: list[dict[str, str | float]]
    gradient_rows: list[dict[str, str | float]]
    null_mean: float
    null_std: float
    null_views: tuple[float, float]
    num_views: int


@dataclasses.dataclass(frozen=True)
class CellResult:
    """Raw output of one synthetic cell (one independent unit of work).

    Attributes:
        name: Unit name (see `cell_units`).
        family: Radial profile, or None for exact `N(0, I)`.
        kappa: Radial spread.
        sigreg: Single-view SIGReg values over repetitions, key 1, and the
            `num_views`-view average, key `cfg.num_views`.
        ks: KS distances to `chi_d` over repetitions.
        gradient: Output of `gradient_stats`.
    """

    name: str
    family: Family | None
    kappa: float
    sigreg: dict[int, FloatArray]
    ks: FloatArray
    gradient: dict[str, float]


NULL_UNIT = "null"


def cell_units() -> dict[str, tuple[Family | None, float]]:
    """All synthetic units by name: the `N(0, I)` null and family x kappa cells."""
    units: dict[str, tuple[Family | None, float]] = {NULL_UNIT: (None, 1.0)}
    for family in FAMILIES:
        for kappa in KAPPAS:
            units[f"{family}_k{kappa:g}"] = (family, kappa)
    return units


def real_unit(run_name: str) -> str:
    """Unit name of the real-embedding analysis of one run."""
    return f"real_{run_name}"


def unit_seed(cfg: StudyConfig, name: str) -> int:
    """Seed of one unit: fixed by the study seed and the unit name only.

    Units therefore give the same result whatever the order or the process they
    run in, which is what makes a fan-out over SLURM jobs reproducible.
    """
    return (cfg.seed * 1_000_003 + zlib.crc32(name.encode())) % 2**32


def _generators(
    cfg: StudyConfig, name: str, device: torch.device
) -> tuple[numpy.random.Generator, torch.Generator]:
    seed = unit_seed(cfg, name)
    return numpy.random.default_rng(seed), torch.Generator(device).manual_seed(seed)


def run_cell(name: str, cfg: StudyConfig, device: torch.device) -> CellResult:
    """Runs one synthetic unit: SIGReg values, KS distances and the gradient split.

    Args:
        name: A key of `cell_units`.
        cfg: Study sizes.
        device: Where SIGReg runs (sampling stays on the CPU).

    Returns:
        The raw results of the cell.
    """
    family, kappa = cell_units()[name]
    rng, generator = _generators(cfg, name, device)
    sigreg_by_views: dict[int, FloatArray] = {}
    for views in (1, cfg.num_views):
        sigreg_by_views[views] = sigreg_values(
            family, kappa, views, cfg, rng, generator
        )
        logger.info("%s: SIGReg with %d view(s) done", name, views)
    ks = ks_values(family, kappa, cfg, rng)
    gradient = gradient_stats(family, kappa, cfg, rng, generator)
    logger.info("%s: done", name)
    return CellResult(name, family, kappa, sigreg_by_views, ks, gradient)


def _atomic_write(
    path: pathlib.Path, write: collections.abc.Callable[[str], None]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(str(tmp))
    os.replace(tmp, path)


def shard_path(shard_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Where a unit's result lives (`.npz` for cells, `.json` for real runs)."""
    suffix = ".json" if name.startswith("real_") else ".npz"
    return shard_dir / f"{name}{suffix}"


def _cfg_json(cfg: StudyConfig) -> str:
    return json.dumps(dataclasses.asdict(cfg), sort_keys=True)


def save_cell(shard_dir: pathlib.Path, cell: CellResult, cfg: StudyConfig) -> None:
    """Writes a cell atomically together with the study config that produced it."""
    arrays: dict[str, typing.Any] = {
        f"sigreg_{views}": values for views, values in cell.sigreg.items()
    }
    arrays["ks"] = cell.ks
    arrays["gradient_json"] = numpy.array(json.dumps(cell.gradient))
    arrays["cfg_json"] = numpy.array(_cfg_json(cfg))

    def write(tmp: str) -> None:
        with open(tmp, "wb") as handle:
            numpy.savez(handle, **arrays)

    _atomic_write(shard_path(shard_dir, cell.name), write)


def load_cell(
    shard_dir: pathlib.Path, name: str, cfg: StudyConfig
) -> CellResult | None:
    """Loads a cell; None if it is missing or was made with a different config."""
    path = shard_path(shard_dir, name)
    if not path.exists():
        return None
    with numpy.load(path) as data:
        if str(data["cfg_json"]) != _cfg_json(cfg):
            return None
        family, kappa = cell_units()[name]
        sigreg_by_views = {
            int(key.removeprefix("sigreg_")): numpy.asarray(data[key])
            for key in data.files
            if key.startswith("sigreg_")
        }
        return CellResult(
            name,
            family,
            kappa,
            sigreg_by_views,
            numpy.asarray(data["ks"]),
            json.loads(str(data["gradient_json"])),
        )


def save_real(
    shard_dir: pathlib.Path,
    name: str,
    row: dict[str, str | float],
    cfg: StudyConfig,
    real_reps: int,
) -> None:
    """Writes the raw row of a real-embedding unit atomically."""
    payload = {"cfg": _cfg_json(cfg), "real_reps": real_reps, "row": row}

    def write(tmp: str) -> None:
        pathlib.Path(tmp).write_text(json.dumps(payload))

    _atomic_write(shard_path(shard_dir, name), write)


def load_real(
    shard_dir: pathlib.Path, name: str, cfg: StudyConfig, real_reps: int
) -> dict[str, str | float] | None:
    """Loads a real unit's raw row; None if missing or made with other settings."""
    path = shard_path(shard_dir, name)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload["cfg"] != _cfg_json(cfg) or payload["real_reps"] != real_reps:
        return None
    return payload["row"]


def assemble(
    cells: collections.abc.Mapping[str, CellResult], cfg: StudyConfig
) -> SyntheticStudy:
    """Combines all synthetic cells into tables (power is against the null cell).

    Args:
        cells: Results of every key of `cell_units`.
        cfg: Study sizes.

    Returns:
        The tables and the null summary.
    """
    null = cells[NULL_UNIT]
    views_list = (1, cfg.num_views)
    power_rows: list[dict[str, str | float]] = []
    gradient_rows: list[dict[str, str | float]] = [
        {"family": "N(0, I)", "kappa": 1.0, **null.gradient}
    ]
    for name, (family, kappa) in cell_units().items():
        if family is None:
            continue
        cell = cells[name]
        for views in views_list:
            vals, base = cell.sigreg[views], null.sigreg[views]
            power_rows.append(
                {
                    "family": family,
                    "kappa": kappa,
                    "statistic": f"SIGReg, {views} view(s)",
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)),
                    "shift / null std": float(
                        (vals.mean() - base.mean()) / base.std(ddof=1)
                    ),
                    "power": power(base, vals),
                }
            )
        power_rows.append(
            {
                "family": family,
                "kappa": kappa,
                "statistic": "KS vs chi_d",
                "mean": float(cell.ks.mean()),
                "std": float(cell.ks.std(ddof=1)),
                "shift / null std": float(
                    (cell.ks.mean() - null.ks.mean()) / null.ks.std(ddof=1)
                ),
                "power": power(null.ks, cell.ks),
            }
        )
        gradient_rows.append({"family": family, "kappa": kappa, **cell.gradient})
    return SyntheticStudy(
        power_rows,
        gradient_rows,
        float(null.sigreg[1].mean()),
        float(null.sigreg[1].std(ddof=1)),
        (
            float(null.sigreg[cfg.num_views].mean()),
            float(null.sigreg[cfg.num_views].std(ddof=1)),
        ),
        cfg.num_views,
    )


def run_synthetic(
    cfg: StudyConfig, device: torch.device | None = None
) -> SyntheticStudy:
    """Runs every synthetic unit in this process (small studies and tests).

    The SLURM fan-out runs the same units, one per job (`run_cell`).
    """
    dev = torch.device("cpu") if device is None else device
    return assemble({name: run_cell(name, cfg, dev) for name in cell_units()}, cfg)


def real_row(
    name: str,
    shard: shards.EmbeddingShard,
    cfg: StudyConfig,
    training_dist: float | None,
    device: torch.device,
    real_reps: int = 20,
) -> dict[str, str | float]:
    """SIGReg value, kappa and KS distance of one run's real `z` (z only).

    Single-view SIGReg is evaluated on `real_reps` random batches of
    `cfg.batch_size` test images, on the center view and averaged over the
    augmented views; the synthetic scaled-chi value at the run's own center-view
    kappa shows how much of it radial spread alone would explain. The gradient of
    the center-view SIGReg with respect to `z` is split into radial and tangential
    parts (`split_gradient`) on the same batches. The standardization against the
    null is added later by `standardize_real_row`.

    Args:
        name: Run name.
        shard: STL-10 test embeddings (`z` is `(N, K + 1, d)`, view 0 = center).
        cfg: Study sizes.
        training_dist: Last epoch mean of the logged training SIGReg, if known.
        device: Where SIGReg runs.
        real_reps: Random batches.

    Returns:
        One table row (without the null-standardized column).
    """
    rng, generator = _generators(cfg, real_unit(name), device)
    z = shard.z.astype(numpy.float64)
    n, num_views, dim = z.shape
    center = z[:, 0]
    sig = _sigreg(cfg, generator)
    center_vals: list[float] = []
    aug_vals: list[float] = []
    ks_vals: list[float] = []
    batch_kappas: list[float] = []
    grads: list[dict[str, float]] = []
    for rep in range(real_reps):
        idx = rng.choice(n, size=min(cfg.batch_size, n), replace=False)
        batch = torch.from_numpy(z[idx].astype(numpy.float32)).to(device)
        center_z = batch[:, 0].clone().requires_grad_(True)
        value = sig(center_z)
        value.backward()  # pyright: ignore[reportUnknownMemberType]
        assert center_z.grad is not None
        center_vals.append(float(value.detach()))
        grads.append(
            split_gradient(
                center_z.detach().cpu().numpy().astype(numpy.float64),
                center_z.grad.cpu().numpy().astype(numpy.float64),
            )
        )
        with torch.no_grad():
            aug_vals.append(
                float(
                    numpy.mean([float(sig(batch[:, k])) for k in range(1, num_views)])
                )
            )
        ks_vals.append(ks_statistic(numpy.linalg.norm(center[idx], axis=-1), dim))
        batch_kappas.append(metrics.kappa(center[idx]))
        logger.info("%s: real batch %d/%d", name, rep + 1, real_reps)
    kappa_center = metrics.kappa(center)
    synthetic = sigreg_values(
        "scaled_chi",
        max(kappa_center, 1e-3),
        1,
        cfg,
        rng,
        generator,
        reps=real_reps,
    )
    center_sigreg = float(numpy.mean(center_vals))
    r = numpy.linalg.norm(center, axis=-1)
    return {
        "run": name,
        "kappa (center views)": kappa_center,
        "kappa (batches of N)": float(numpy.mean(batch_kappas)),
        "E||z||^2 / d": float(numpy.mean(r**2) / dim),
        "std(||z||) / mean(||z||)": float(r.std() / r.mean()),
        "SIGReg, center view": center_sigreg,
        "SIGReg, augmented views": float(numpy.mean(aug_vals)),
        "synthetic scaled-chi at same kappa": float(synthetic.mean()),
        "KS vs chi_d (batch of N)": float(numpy.mean(ks_vals)),
        "grad radial rms": float(numpy.mean([g["radial_rms"] for g in grads])),
        "grad tangential rms": float(numpy.mean([g["tangential_rms"] for g in grads])),
        "grad restoring corr": float(
            numpy.nanmean([g["restoring_corr"] for g in grads])
        ),
        "training SIGReg (last epoch)": "" if training_dist is None else training_dist,
    }


def standardize_real_row(
    row: dict[str, str | float], study: SyntheticStudy
) -> dict[str, str | float]:
    """Adds `(center - null mean) / null std` to a raw real row (after `kappa`)."""
    out: dict[str, str | float] = {}
    for key, value in row.items():
        out[key] = value
        if key == "SIGReg, augmented views":
            out["(center - null mean) / null std"] = (
                float(row["SIGReg, center view"]) - study.null_mean
            ) / study.null_std
    return out


def load_real_shard(embed_dir: str, run_name: str) -> shards.EmbeddingShard | None:
    """Loads the `stl10_test` dump of a run (None if missing)."""
    return dumps.load_set(dumps.set_dir(embed_dir, run_name, dumps.STL10_TEST))


def last_training_dist(records: list[dict[str, typing.Any]]) -> float | None:
    """Last epoch-mean `loss_dist` (training SIGReg) in a metrics log, if any."""
    values = [
        float(rec["loss_dist_epoch_mean"])
        for rec in records
        if rec.get("kind") == "epoch" and "loss_dist_epoch_mean" in rec
    ]
    return values[-1] if values else None
