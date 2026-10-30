"""View-direction study: crop content, nested models on planted effects, workflow."""

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false

import json
import pathlib

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.data import layout
from lejepa_norm.eval import aug_regression
from lejepa_norm.eval import dumps
from lejepa_norm.eval import features
from lejepa_norm.eval import shards
from lejepa_norm.eval import usage_aug_regression
from lejepa_norm.eval import usage_view_direction
from lejepa_norm.eval import view_direction
import test_aug_regression as toy

N, K, SIZE, D = toy.N, toy.K, toy.SIZE, 12


def _images(seed: int) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    return rng.integers(0, 256, size=(N, SIZE, SIZE, 3), dtype=numpy.uint8)


def _views(
    rng: numpy.random.Generator, log_r: numpy.ndarray, spread: float
) -> numpy.ndarray:
    """Views `(N, K, D)`: a shared image direction plus a view-specific offset."""
    base = rng.normal(size=(N, 1, D))
    return (
        features.direction(base + spread * rng.normal(size=(N, K, D)))
        * numpy.exp(log_r)[..., None]
    )


def test_crop_content_matches_brute_force() -> None:
    images, aug = _images(0), toy._aug(0)
    content = view_direction.crop_content(images, aug)
    fields = shards.AUG_FIELDS
    for i, v in [(0, 0), (7, 3), (400, 5)]:
        top, left, h, w = (
            aug[i, v, fields.index(f)] for f in ("top", "left", "height", "width")
        )
        r0, c0 = int(numpy.rint(top)), int(numpy.rint(left))
        r1, c1 = int(numpy.rint(top + h)), int(numpy.rint(left + w))
        rgb = images[i].astype(numpy.float64) / 255
        crop = rgb[r0:r1, c0:c1]
        lum = crop @ numpy.array([0.299, 0.587, 0.114])
        assert content[i, v, 0] == pytest.approx(lum.mean(), abs=1e-9)
        assert content[i, v, 1] == pytest.approx(lum.std(), abs=1e-6)
        shift = numpy.linalg.norm(crop.mean(axis=(0, 1)) - rgb.mean(axis=(0, 1)))
        assert content[i, v, 3] == pytest.approx(shift, abs=1e-9)


def test_direction_block_found_only_when_the_radius_depends_on_it() -> None:
    rng = numpy.random.default_rng(1)
    aug = toy._aug(1)
    design = aug_regression.build_design(aug, SIZE, SIZE)
    content = view_direction.crop_content(_images(1), aug)
    # Radius = image effect + parameter effect + direction effect (+ noise).
    z = _views(rng, numpy.zeros((N, K)), spread=0.6)
    w = rng.normal(size=D)
    off = features.direction(z) - features.direction(z).mean(axis=1, keepdims=True)
    from_direction = 2.0 * (off @ w)
    base = toy._radius(design, rng, noise=0.1)
    correct = rng.random((N, K)) < 0.5

    def run(y: numpy.ndarray) -> view_direction.DirectionResult:
        zz = z * numpy.exp(y)[..., None] / numpy.linalg.norm(z, axis=-1, keepdims=True)
        return view_direction.analyze_direction(
            y, design, zz, content, correct, 30, 5, rng
        )

    with_direction = run(base + from_direction)
    without = run(base)
    assert with_direction.increments["direction | params"][0] > 0.2
    assert without.increments["direction | params"][0] < 0.05
    # Content was not planted in either.
    assert abs(with_direction.increments["content | params+direction"][0]) < 0.03
    # Correctness is random here, so no residual predicts it.
    assert abs(with_direction.resid_auc["params+direction+content"][0] - 0.5) < 0.03


def test_unit_and_note_workflow_on_a_toy_dump(tmp_path: pathlib.Path) -> None:
    rng = numpy.random.default_rng(2)
    run, classes = "toy_lam0.05_s0", usage_aug_regression.NUM_CLASSES
    aug = toy._aug(2)
    design = aug_regression.build_design(aug, SIZE, SIZE)
    z = _views(rng, toy._radius(design, rng, noise=0.2), spread=0.5)
    labels = rng.integers(0, classes, size=N)
    center = z.mean(axis=1, keepdims=True)
    z_all = numpy.concatenate([center, z], axis=1).astype(numpy.float32)

    runs, embed, data = tmp_path / "runs", tmp_path / "embed", tmp_path / "data"
    cfg = config.ExperimentConfig(seed=0)
    cfg.loss.lambda_ = 0.05
    cfg.eval.num_bootstrap = 10
    config.save_resolved(cfg, runs / run)
    path = layout.images_path(data, "stl10", "test")
    path.parent.mkdir(parents=True)
    numpy.save(path, _images(2))

    def shard(
        z_part: numpy.ndarray, aug_part: numpy.ndarray | None
    ) -> shards.EmbeddingShard:
        return shards.EmbeddingShard(
            z=z_part,
            h=z_part.copy(),
            image_id=numpy.arange(N, dtype=numpy.int64),
            label=labels.astype(numpy.int64),
            corruption=numpy.array([""] * N),
            severity=numpy.zeros(N, dtype=numpy.int8),
            aug=aug_part,
        )

    bank = dumps.set_dir(embed, run, dumps.STL10_TRAIN)
    bank.mkdir(parents=True)
    shards.save_shard(bank / "part-00000.npz", shard(z_all[:, :1], None))
    test = dumps.set_dir(embed, usage_aug_regression.dump_name(run), dumps.STL10_TEST)
    test.mkdir(parents=True)
    shards.save_shard(test / "part-00000.npz", shard(z_all, aug.astype(numpy.float32)))
    locations = paths.Paths(data, runs, embed, tmp_path / "notes", None)

    out = usage_view_direction.run_unit(cfg, locations, run, lambda _: None)
    unit = usage_view_direction.Unit(**json.loads(out.read_text()))
    assert unit.run == run and unit.lam == 0.05
    assert unit.spread["within_mean"] < unit.spread["across_mean"]
    assert unit.r2["params"][3] > 0.5  # planted parameter effects

    written = usage_view_direction.write_notes(
        cfg, locations, "toy", pathlib.Path("t.md")
    )
    note = written[0].read_text()
    assert "analyze.dir_verdicts" in note and "V1 direction adds" in note
    assert "toy_lam0.05_s0" in note
