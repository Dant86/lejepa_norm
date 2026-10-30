"""Within-image regression of the radius on augmentation parameters (planted)."""

import json
import pathlib

import numpy
import pytest
import torch

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.data import gpu_views
from lejepa_norm.data import view_spec
from lejepa_norm.eval import aug_regression
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import shards
from lejepa_norm.eval import usage_aug_regression

N, K, SIZE = 1500, 8, 96
AREA, BLUR_GATE = 0, 13  # columns of `aug_regression.COLUMN_NAMES`


def _aug(seed: int = 0) -> numpy.ndarray:
    """Augmentation parameters drawn by the real sampler, `(N, K, 13)`."""
    spec = view_spec.global_spec(config.DataConfig())
    generator = torch.Generator().manual_seed(seed)
    views = [gpu_views.sample_params(spec, N, SIZE, SIZE, generator) for _ in range(K)]
    return torch.stack(
        [
            torch.stack([getattr(v, f).float() for f in shards.AUG_FIELDS], dim=1)
            for v in views
        ],
        dim=1,
    ).numpy()


def _radius(
    design: aug_regression.Design, rng: numpy.random.Generator, noise: float
) -> numpy.ndarray:
    """log-radius: image effect + 0.5 sd per sd of log area - 0.3 per blur gate."""
    alpha = rng.normal(scale=1.0, size=(N, 1))
    x = design.x
    return (
        alpha
        + 0.5 * x[..., AREA]
        - 0.3 * x[..., BLUR_GATE]
        + noise * rng.normal(size=(N, K))
    )


def test_recovers_planted_effects_and_block_shares() -> None:
    rng = numpy.random.default_rng(0)
    design = aug_regression.build_design(_aug(), SIZE, SIZE)
    # The two planted columns are weakly correlated through the sampler, not equal.
    y = _radius(design, rng, noise=0.2)
    out = aug_regression.analyze_regression(y, design, 200, 5, rng, seed=0)

    # Area enters through a spline, so check the effect curve's direction and the
    # blur coefficient (sigma is a separate column, so the gate effect is ~ -0.3).
    assert out.curve[-1] > out.curve[0] + 0.5
    assert out.beta[BLUR_GATE] == pytest.approx(-0.3, abs=0.06)
    low, high = out.beta_interval[BLUR_GATE]
    assert low < -0.3 < high
    # Unplanted columns are near zero.
    for name in ("flip", "brightness", "grayscale"):
        j = aug_regression.COLUMN_NAMES.index(name)
        assert abs(out.beta[j]) < 0.05
    # Crop and blur blocks carry the signal; color and flip do not.
    drops = {k: v[0] for k, v in out.block_drop.items()}
    assert drops["crop"] > 0.2 and drops["blur"] > 0.005
    assert drops["color"] < 0.01 and drops["flip"] < 0.01
    # The noise is 0.2, so the explained within-image share is high, and trees do
    # not find structure the (correct) linear model misses.
    assert out.within_r2[0] > 0.8
    assert out.cv_r2_linear == pytest.approx(out.within_r2[0], abs=0.02)
    assert out.cv_r2_trees < out.cv_r2_linear + 0.05
    # alpha_i has variance 1 vs ~0.3 within: most variance is between images.
    assert 0.1 < out.within_share < 0.4


def test_radius_unrelated_to_parameters_explains_nothing() -> None:
    rng = numpy.random.default_rng(1)
    design = aug_regression.build_design(_aug(1), SIZE, SIZE)
    y = rng.normal(size=(N, 1)) + 0.3 * rng.normal(size=(N, K))
    out = aug_regression.analyze_regression(y, design, 100, 5, rng, seed=0)
    assert out.within_r2[0] < 0.02
    assert out.cv_r2_linear < 0.01
    assert out.within_r2[1] <= out.within_r2[0] <= out.within_r2[2]


def test_residual_radius_predicts_view_quality_only_when_it_carries_it() -> None:
    rng = numpy.random.default_rng(2)
    design = aug_regression.build_design(_aug(2), SIZE, SIZE)
    xd = aug_regression.demean(design.x)
    p = aug_regression
    # Radius = parameter effect + a part not specified by the parameters.
    hidden = 0.5 * rng.normal(size=(N, K))
    y = _radius(design, rng, noise=0.0) + hidden
    yd = p.demean(y)

    def quality(depends_on_hidden: bool) -> aug_regression.ViewQualityResult:
        signal = (hidden if depends_on_hidden else xd[..., AREA]) / 0.5
        noise = rng.normal(size=(N, K))
        correct = (signal + noise) > 0
        angle = -0.1 * signal + 0.05 * rng.normal(size=(N, K))
        return p.analyze_view_quality(yd, xd, correct, angle, 100, 5, rng)

    carries = quality(True)
    proxy = quality(False)
    # The hidden part is in the residual: r keeps most of the AUC and adds to the angle.
    assert carries.retains
    assert carries.correct_auc["r"][0] > 0.58
    assert carries.angle_r2["r_increment"][1] > 0
    # If quality only follows the parameters, the residual has nothing to add.
    assert not proxy.retains
    assert proxy.correct_auc["yhat"][0] > 0.58
    assert abs(proxy.correct_auc["r"][0] - 0.5) < 0.03
    assert proxy.angle_r2["r_increment"][1] < 0.01


def test_design_neutralizes_gated_parameters() -> None:
    aug = _aug(3)
    design = aug_regression.build_design(aug, SIZE, SIZE)
    col = {n: design.raw[..., j] for j, n in enumerate(aug_regression.COLUMN_NAMES)}
    off = aug[..., shards.AUG_FIELDS.index("jitter")] == 0
    assert (col["brightness"][off] == 1).all() and (col["abs_hue"][off] == 0).all()
    gate = aug[..., shards.AUG_FIELDS.index("blur")] == 0
    assert (col["sigma"][gate] == 0).all()
    # Crop areas follow the global-view scale range of the training config.
    area = numpy.exp(col["log_area"])
    assert area.min() > 0.2 and area.max() <= 1.0


def test_within_auc_and_angle_helpers() -> None:
    # One image: correct views score higher than incorrect ones in 3 of 4 pairs.
    score = numpy.array([[3.0, 1.0, 2.0, 2.5]])
    correct = numpy.array([[True, False, True, False]])
    wins, pairs = aug_regression.within_auc_counts(score, correct)
    assert pairs.tolist() == [4.0]
    assert wins.tolist() == [3.0]
    z = numpy.zeros((1, 3, 4))
    z[0, :, 0] = [1.0, 2.0, 3.0]  # identical directions, different radii
    assert aug_regression.angle_to_mean_direction(z) == pytest.approx(0, abs=1e-6)


def test_unit_and_note_workflow_on_a_toy_dump(tmp_path: pathlib.Path) -> None:
    """run_unit on a dump with planted parameter effects, then the note sections."""
    rng = numpy.random.default_rng(5)
    run, d, classes = "toy_lam0.05_s0", 12, usage_aug_regression.NUM_CLASSES
    aug = _aug(4)
    design = aug_regression.build_design(aug, SIZE, SIZE)
    log_r = _radius(design, rng, noise=0.2)
    centers = rng.normal(size=(classes, d))
    labels = rng.integers(0, classes, size=N)
    direction = features.direction(centers[labels] + 0.5 * rng.normal(size=(N, d)))
    views = direction[:, None, :] * numpy.exp(log_r)[..., None]
    center = direction * numpy.exp(log_r.mean(axis=1))[:, None]
    z = numpy.concatenate([center[:, None, :], views], axis=1).astype(numpy.float32)

    runs, embed = tmp_path / "runs", tmp_path / "embed"
    cfg = config.ExperimentConfig(seed=0)
    cfg.loss.lambda_ = 0.05
    cfg.eval.num_bootstrap = 20
    config.save_resolved(cfg, runs / run)
    bank_dir = dumps.set_dir(embed, run, dumps.STL10_TRAIN)
    bank_dir.mkdir(parents=True)
    shards.save_shard(
        bank_dir / "part-00000.npz",
        shards.EmbeddingShard(
            z=z[:, :1],
            h=z[:, :1].copy(),
            image_id=numpy.arange(N, dtype=numpy.int64),
            label=labels.astype(numpy.int64),
            corruption=numpy.array([""] * N),
            severity=numpy.zeros(N, dtype=numpy.int8),
        ),
    )
    test_dir = dumps.set_dir(
        embed, usage_aug_regression.dump_name(run), dumps.STL10_TEST
    )
    test_dir.mkdir(parents=True)
    shards.save_shard(
        test_dir / "part-00000.npz",
        shards.EmbeddingShard(
            z=z,
            h=z.copy(),
            image_id=numpy.arange(N, dtype=numpy.int64),
            label=labels.astype(numpy.int64),
            corruption=numpy.array([""] * N),
            severity=numpy.zeros(N, dtype=numpy.int8),
            aug=aug.astype(numpy.float32),
        ),
    )
    locations = paths.Paths(tmp_path / "data", runs, embed, tmp_path / "notes", None)

    out = usage_aug_regression.run_unit(cfg, locations, run, lambda _: None)
    unit = usage_aug_regression.Unit(**json.loads(out.read_text()))
    assert unit.run == run and unit.lam == 0.05
    assert unit.within_r2[0] > 0.7  # the planted parameters explain the radius
    assert unit.block_drop["crop"][0] > unit.block_drop["color"][0]
    assert 0.0 <= unit.correct_auc["r"][0] <= 1.0

    written = usage_aug_regression.write_notes(
        cfg, locations, "toy", pathlib.Path("toy/aug.md"), figures=False
    )
    note = written[0].read_text()
    assert "analyze.aug_verdicts" in note and "P2 CV R2" in note
    assert "analyze.aug_blocks" in note and "toy_lam0.05_s0" in note
