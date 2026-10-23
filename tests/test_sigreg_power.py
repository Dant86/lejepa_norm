"""Synthetic SIGReg power study: moment relations, null recovery, workflow."""

import dataclasses
import importlib.util
import pathlib
import types

import numpy
import pytest
import torch

from lejepa_norm import paths
from lejepa_norm import reporting
from lejepa_norm.eval import dumps
from lejepa_norm.eval import shards
from lejepa_norm.eval import sigreg_power as sp
from lejepa_norm.losses import sigreg

REPO = pathlib.Path(__file__).parent.parent


@pytest.mark.parametrize("family", sp.FAMILIES)
def test_radius_profiles_match_the_requested_moments(family: sp.Family) -> None:
    rng = numpy.random.default_rng(0)
    dim, kappa = 64, 8.0
    r2 = sp.sample_radius(family, kappa, dim, 400_000, rng) ** 2
    assert r2.mean() == pytest.approx(dim, rel=0.01)
    assert r2.var() / (2 * dim) == pytest.approx(kappa, rel=0.06)


@pytest.mark.parametrize("family", sp.FAMILIES)
def test_projection_kurtosis_is_the_exact_formula_not_the_naive_one(
    family: sp.Family,
) -> None:
    # Small d so that the d / (d + 2) factor (0.8) separates the two formulas.
    dim, kappa = 8, 5.0
    cfg = sp.StudyConfig(dim=dim)
    rng = numpy.random.default_rng(1)
    z = sp.sample_embeddings(family, kappa, dim, (400_000,), rng).numpy()
    a = rng.standard_normal(z.shape)
    a /= numpy.linalg.norm(a, axis=-1, keepdims=True)
    x = (z * a).sum(axis=-1).astype(numpy.float64)
    measured = numpy.mean(x**4) / numpy.mean(x**2) ** 2
    exact = sp.projection_kurtosis(kappa, cfg.dim)
    naive = 3 * (1 + 2 * kappa / dim)
    assert exact == pytest.approx(5.4)
    assert abs(measured - exact) < 0.15
    assert abs(measured - naive) > 0.8
    assert sp.projection_kurtosis(1.0, 256) == pytest.approx(3.0)  # Gaussian


def test_kappa_one_recovers_the_gaussian_null() -> None:
    cfg = sp.StudyConfig(dim=16, batch_size=64, num_slices=64, reps=60)
    rng = numpy.random.default_rng(0)
    gen = torch.Generator().manual_seed(0)
    null = sp.sigreg_values(None, 1.0, 1, cfg, rng, gen)
    chi = sp.sigreg_values("scaled_chi", 1.0, 1, cfg, rng, gen)
    se = numpy.sqrt(null.var(ddof=1) / null.size + chi.var(ddof=1) / chi.size)
    assert abs(null.mean() - chi.mean()) < 4 * se
    # The null mean is the analytic noise floor of the repo's SIGReg.
    floor = sigreg.SIGReg(cfg.num_slices, cfg.num_points, cfg.t_max).noise_floor
    assert abs(null.mean() - floor) < 4 * null.std(ddof=1) / numpy.sqrt(null.size)


def test_ks_statistic_is_small_for_chi_and_large_for_spread_radii() -> None:
    cfg = sp.StudyConfig(dim=64, batch_size=256, reps=100)
    rng = numpy.random.default_rng(0)
    null = sp.ks_values(None, 1.0, cfg, rng)
    wide = sp.ks_values("scaled_chi", 20.0, cfg, rng)
    assert sp.power(null, wide) > 0.9
    assert sp.power(null, sp.ks_values("scaled_chi", 1.0, cfg, rng)) < 0.2


def test_gradient_split_is_an_orthogonal_decomposition() -> None:
    rng = numpy.random.default_rng(0)
    z = rng.standard_normal((50, 12))
    r = numpy.linalg.norm(z, axis=1)
    # A purely radial gradient: all tangential parts vanish.
    stats = sp.split_gradient(z, 2.0 * z / r[:, None])
    assert stats["radial_rms"] == pytest.approx(2.0)
    assert stats["tangential_rms"] == pytest.approx(0.0, abs=1e-12)
    # A purely tangential one: orthogonal to z by construction.
    g = rng.standard_normal(z.shape)
    g -= (g * z).sum(1, keepdims=True) / (r**2)[:, None] * z
    stats = sp.split_gradient(z, g)
    assert stats["radial_rms"] == pytest.approx(0.0, abs=1e-12)
    assert stats["norm_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_synthetic_study_shows_sigreg_rising_with_kappa() -> None:
    cfg = sp.StudyConfig(
        dim=32, batch_size=64, num_slices=64, num_views=2, reps=40, grad_reps=3
    )
    study = sp.run_synthetic(cfg)
    rows = [
        r
        for r in study.power_rows
        if r["family"] == "scaled_chi" and r["statistic"] == "SIGReg, 1 view(s)"
    ]
    means = [float(r["mean"]) for r in rows]
    assert [r["kappa"] for r in rows] == list(sp.KAPPAS)
    assert means[-1] > means[0]
    assert means[0] == pytest.approx(study.null_mean, abs=4 * study.null_std)
    ks = [r for r in study.power_rows if r["statistic"] == "KS vs chi_d"]
    assert all(float(r["power"]) > 0.9 for r in ks if float(r["kappa"]) >= 15)
    assert {r["family"] for r in study.gradient_rows} == {"N(0, I)", *sp.FAMILIES}


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app_power", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_units_are_reproducible_and_shards_guard_against_stale_config(
    tmp_path: pathlib.Path,
) -> None:
    cfg = sp.StudyConfig(
        dim=16, batch_size=32, num_slices=16, num_views=2, reps=6, grad_reps=2
    )
    cpu = torch.device("cpu")
    first = sp.run_cell("scaled_chi_k5", cfg, cpu)
    again = sp.run_cell("scaled_chi_k5", cfg, cpu)
    other = sp.run_cell("lognormal_k5", cfg, cpu)
    numpy.testing.assert_array_equal(first.sigreg[1], again.sigreg[1])
    assert not numpy.array_equal(first.sigreg[1], other.sigreg[1])
    assert sp.load_cell(tmp_path, first.name, cfg) is None
    sp.save_cell(tmp_path, first, cfg)
    loaded = sp.load_cell(tmp_path, first.name, cfg)
    assert loaded is not None
    numpy.testing.assert_array_equal(loaded.sigreg[2], first.sigreg[2])
    assert loaded.gradient == first.gradient
    stale = dataclasses.replace(cfg, reps=7)
    assert sp.load_cell(tmp_path, first.name, stale) is None


def test_app_writes_synthetic_and_real_sections(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rng = numpy.random.default_rng(0)
    n, k, d = 300, 3, 32
    z = rng.standard_normal((n, k + 1, d)).astype(numpy.float32)
    shard = shards.EmbeddingShard(
        z=z,
        h=numpy.zeros((n, k + 1, 4), dtype=numpy.float32),
        image_id=numpy.arange(n),
        label=numpy.zeros(n, dtype=numpy.int64),
        corruption=numpy.array([""] * n),
        severity=numpy.zeros(n, dtype=numpy.int8),
    )
    embed = tmp_path / "embed"
    out = dumps.set_dir(embed, "toy_s0", dumps.STL10_TEST)
    out.mkdir(parents=True)
    shards.save_shard(out / "part-0.npz", shard)
    for var, sub in (
        (paths.DATA_DIR_VAR, "data"),
        (paths.RUNS_DIR_VAR, "runs"),
        (paths.EMBED_DIR_VAR, "embed"),
        (paths.NOTES_DIR_VAR, "notes"),
    ):
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.setenv(paths.KALEIDO_CHROME_VAR, "")
    monkeypatch.chdir(tmp_path)
    common = [
        "--sigreg-power",
        "--config",
        str(REPO / "configs" / "base.yaml"),
        "--runs-prefix",
        "toy",
        "--set",
        "model.embed_dim=32",
        "--set",
        "train.batch_size=64",
        "--set",
        "loss.num_slices=32",
        "--set",
        "eval.power_reps=10",
        "--set",
        "eval.power_grad_reps=2",
        "--set",
        "eval.power_real_reps=3",
    ]
    app = _load_app()
    assert app.main([*common, "--power-list-units"]) == 0
    units = capsys.readouterr().out.split()
    assert units == [*sp.cell_units(), "real_toy_s0"]
    aggregate = [*common, "--note", "idea/power.md", "--no-figures"]
    with pytest.raises(FileNotFoundError, match="null"):
        app.main(aggregate)
    for unit in units:  # what the SLURM array does, one task per unit
        assert app.main([*common, "--power-unit", unit]) == 0
    shard_dir = tmp_path / "runs" / "sigreg_power" / "default"
    mtimes = {p.name: p.stat().st_mtime_ns for p in shard_dir.iterdir()}
    assert len(mtimes) == len(units)
    assert app.main([*common, "--power-unit", "null"]) == 0  # resume: skipped
    assert {p.name: p.stat().st_mtime_ns for p in shard_dir.iterdir()} == mtimes
    assert app.main(aggregate) == 0
    text = (tmp_path / "notes" / "idea" / "power.md").read_text()
    for name in ("setup", "kurtosis", "power", "gradient", "real"):
        assert f"BEGIN GENERATED: sigreg_power.{name}" in text
    note = tmp_path / "notes" / "idea" / "power.md"
    rows = reporting.parse_table(reporting.read_section(note, "sigreg_power.real"))
    assert [r["run"] for r in rows] == ["toy_s0"]
    assert 0.6 < float(rows[0]["kappa (center views)"]) < 1.6  # N(0, I) embeddings
    ratio = float(rows[0]["grad radial rms"]) / float(rows[0]["grad tangential rms"])
    assert 0.3 < ratio < 3.0  # null: no preferred direction
