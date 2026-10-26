"""SVHN control: low-level statistics vs the radius signal, end to end."""

import importlib.util
import pathlib
import types

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import image_stats
from lejepa_norm.eval import ood_radius
from lejepa_norm.eval import shards
from lejepa_norm.eval import svhn_control

REPO = pathlib.Path(__file__).parent.parent
D, SIZE = 8, 16
SIZES = {"train": 240, "test": 160, "dtd": 60, "svhn": 200}
LAYOUT = {
    "train": ("stl10", "train", dumps.STL10_TRAIN),
    "test": ("stl10", "test", dumps.STL10_TEST),
    "dtd": ("dtd", "test", dumps.DTD_TEST),
    "svhn": ("svhn", "test", dumps.SVHN_TEST),
}


def _images(contrast: numpy.ndarray, rng: numpy.random.Generator) -> numpy.ndarray:
    noise = rng.normal(size=(contrast.size, SIZE, SIZE, 3))
    pixels = 0.5 + contrast[:, None, None, None] * noise
    return numpy.clip(255 * pixels, 0, 255).astype(numpy.uint8)


def test_image_statistics_order_images_as_expected() -> None:
    rng = numpy.random.default_rng(0)
    flat = numpy.full((1, SIZE, SIZE, 3), 128, dtype=numpy.uint8)
    noisy = _images(numpy.array([0.3]), rng)
    ramp = numpy.broadcast_to(
        numpy.linspace(0, 255, SIZE).astype(numpy.uint8)[None, None, :, None],
        (1, SIZE, SIZE, 3),
    ).copy()
    stats = image_stats.compute(numpy.concatenate([flat, noisy, ramp]))
    col = {n: stats[:, i] for i, n in enumerate(image_stats.NAMES)}
    assert col["lum_std"][0] == pytest.approx(0.0, abs=1e-9)
    assert col["saturation"][0] == pytest.approx(0.0, abs=1e-9)
    assert col["colorfulness"][0] == pytest.approx(0.0, abs=1e-9)
    assert col["lum_mean"][0] == pytest.approx(128 / 255)
    # Noise has far more edge energy, high frequencies, Laplacian variance and bytes
    # than a ramp, which has more contrast than the flat image.
    for name in ("grad_energy", "hf_fraction", "laplacian_var", "jpeg_bytes"):
        assert col[name][1] > col[name][2]
    assert col["png_bytes"][1] > col["png_bytes"][0]
    assert col["lum_std"][2] > col["lum_std"][0]
    assert numpy.isfinite(image_stats.transform(stats)).all()
    with pytest.raises(ValueError):
        image_stats.compute(numpy.zeros((2, 4, 5, 3), dtype=numpy.uint8))


def _build_world(
    tmp_path: pathlib.Path, svhn_shift: float, svhn_contrast: tuple[float, float]
) -> paths.Paths:
    """Radius is driven by contrast (a low-level statistic) in every set; SVHN
    additionally has a log-radius offset `svhn_shift` that no statistic explains."""
    data, runs, embed = tmp_path / "data", tmp_path / "runs", tmp_path / "embed"
    rng = numpy.random.default_rng(0)
    contrast = {
        name: rng.uniform(0.05, 0.4, n)
        if name != "svhn"
        else rng.uniform(*svhn_contrast, n)
        for name, n in SIZES.items()
    }
    for name, c in contrast.items():
        dataset, split, _ = LAYOUT[name]
        (data / dataset).mkdir(parents=True, exist_ok=True)
        numpy.save(data / dataset / f"{split}_images.npy", _images(c, rng))
        labels = numpy.zeros(c.size, dtype=numpy.int64)
        numpy.save(data / dataset / f"{split}_labels.npy", labels)
    for seed in (0, 1):
        run = f"toy_lam0.05_s{seed}"
        cfg = config.ExperimentConfig(seed=seed)
        cfg.loss.lambda_ = 0.05
        cfg.eval.knn_k = 5
        cfg.eval.cv_folds = 3
        config.save_resolved(cfg, runs / run)
        z: dict[str, numpy.ndarray] = {}
        for name, c in contrast.items():
            u = features.direction(rng.normal(size=(c.size, D)))
            log_r = 1.0 + 3.0 * (c - 0.2) + 0.1 * rng.normal(size=c.size)
            if name == "svhn":
                log_r = log_r + svhn_shift
            z[name] = u * numpy.exp(log_r)[:, None]
            out = dumps.set_dir(embed, run, LAYOUT[name][2])
            out.mkdir(parents=True)
            full = numpy.repeat(z[name][:, None, :], 2, axis=1).astype(numpy.float32)
            shards.save_shard(
                out / "part-00000.npz",
                shards.EmbeddingShard(
                    z=full,
                    h=full.copy(),
                    image_id=numpy.arange(c.size, dtype=numpy.int64),
                    label=numpy.zeros(c.size, dtype=numpy.int64),
                    corruption=numpy.array([""] * c.size),
                    severity=numpy.zeros(c.size, dtype=numpy.int8),
                ),
            )
        scores = {
            ood: ood_radius.compute_scores(z["train"], z["test"], z[key], 5, seed)
            for ood, key in ((dumps.DTD_TEST, "dtd"), (dumps.SVHN_TEST, "svhn"))
        }
        ood_radius.save_unit(
            runs / run / "shift_ood.npz",
            ood_radius.OodUnit(run, 0.05, seed, SIZES["test"], scores),
        )
    return paths.Paths(data, runs, embed, tmp_path / "notes", None)


def _aggregate(locations: paths.Paths) -> svhn_control.LambdaResult:
    svhn_control.compute_image_stats(locations, lambda _: None)
    units: list[svhn_control.Unit] = []
    for seed in (0, 1):
        run = f"toy_lam0.05_s{seed}"
        run_cfg = config.load_config(locations.runs_dir / run / "config.resolved.yaml")
        out = svhn_control.run_unit(run_cfg, locations, run, lambda _: None)
        units.append(svhn_control.load_unit(out))
    raw = svhn_control.load_image_stats(locations.runs_dir)
    return svhn_control.evaluate(units, raw, 40, numpy.random.default_rng(0))[0]


def test_radius_signal_that_statistics_explain_is_removed_by_the_controls(
    tmp_path: pathlib.Path,
) -> None:
    # SVHN: low contrast, hence small radius through the same contrast -> radius law.
    result = _aggregate(_build_world(tmp_path, 0.0, (0.02, 0.12)))
    assert result.r2[svhn_control.R2_STATS_GBM][0] > 0.6
    svhn = result.ood[dumps.SVHN_TEST]
    assert svhn.auroc[svhn_control.RADIUS_LOW][0] > 0.9  # the radius separates SVHN
    # ... but the statistics reproduce it and the matched control removes it.
    assert svhn.auroc[svhn_control.STATS_PRED][0] > 0.85
    matched = result.matched[dumps.SVHN_TEST]
    assert matched.closed[0] > 0.6
    assert matched.auroc_matched[0] < matched.auroc_all[0] - 0.15
    residual = svhn.auroc[svhn_control.RES_LOW][0]
    assert residual < svhn.auroc[svhn_control.RADIUS_LOW][0] - 0.2


def test_radius_signal_that_statistics_miss_survives_every_control(
    tmp_path: pathlib.Path,
) -> None:
    # SVHN has the same contrast law as STL-10 but an unexplained log-radius offset.
    result = _aggregate(_build_world(tmp_path, -0.8, (0.05, 0.4)))
    svhn = result.ood[dumps.SVHN_TEST]
    assert svhn.auroc[svhn_control.RADIUS_LOW][0] > 0.9
    assert svhn.auroc[svhn_control.RES_LOW][0] > 0.9  # statistics do not remove it
    assert abs(svhn.auroc[svhn_control.STATS_PRED][0] - 0.5) < 0.12
    matched = result.matched[dumps.SVHN_TEST]
    assert matched.closed[0] < 0.3
    assert matched.auroc_matched[0] > 0.85
    rows = svhn_control.overall_rows(result)
    row = next(r for r in rows if r["OOD set"] == dumps.SVHN_TEST)
    assert row["matched AUROC reading"] == ood_radius.INFORMATIVE
    # The statistics still explain the ID law (R^2 high in-distribution).
    assert result.r2[svhn_control.R2_STATS_GBM][0] > 0.6


def test_difference_verdict_follows_the_preregistered_rule() -> None:
    def boot(center: float, half: float) -> numpy.ndarray:
        return numpy.concatenate([[center], center + numpy.linspace(-half, half, 200)])

    verdict = svhn_control.difference_verdict
    assert verdict(boot(0.05, 0.01), 0.01) == svhn_control.A_HIGHER
    assert verdict(boot(0.0, 0.005), 0.01) == svhn_control.EQUIVALENT
    assert verdict(boot(-0.05, 0.01), 0.01) == svhn_control.B_HIGHER
    assert verdict(boot(0.0, 0.05), 0.01) == svhn_control.INCONCLUSIVE


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_svhn_control_modes_write_a_note(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locations = _build_world(tmp_path, -0.8, (0.05, 0.4))
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(locations.data_dir))
    monkeypatch.setenv(paths.RUNS_DIR_VAR, str(locations.runs_dir))
    monkeypatch.setenv(paths.EMBED_DIR_VAR, str(locations.embed_dir))
    monkeypatch.setenv(paths.NOTES_DIR_VAR, str(locations.notes_dir))
    cfg = config.ExperimentConfig()
    cfg.eval.knn_k, cfg.eval.cv_folds, cfg.eval.num_bootstrap = 5, 3, 20
    config.save_resolved(cfg, tmp_path / "resolved")
    cfg_file = tmp_path / "resolved" / "config.resolved.yaml"
    app = _load_app()
    base = ["--config", str(cfg_file), "--svhn-control", "toy"]
    assert app.main([*base, "--svhn-control-stats"]) == 0
    for seed in (0, 1):
        assert app.main([*base, "--svhn-control-run", f"toy_lam0.05_s{seed}"]) == 0
    note = ["--note", "radius_usage/svhn_control.md", "--no-figures"]
    assert app.main([*base, *note]) == 0
    text = (
        locations.notes_dir / "radius_usage" / "svhn_control_lam0.05.md"
    ).read_text()
    for heading in ("(a) Held-out R^2", "Matched control", "Overall reading"):
        assert heading in text
