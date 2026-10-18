# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Data pipeline tests on tiny fake datasets (no network)."""

import os
import pathlib
import subprocess
import sys

import numpy as np
import PIL.Image
import pytest
import scipy.io
import torch

from lejepa_norm import config
from lejepa_norm.data import corruptions
from lejepa_norm.data import datasets
from lejepa_norm.data import layout
from lejepa_norm.data import prepare
from lejepa_norm.data import views

REPO = pathlib.Path(__file__).resolve().parents[1]
STL_SIZES = {"unlabeled": 6, "train": 4, "test": 5}
N_DTD = 4
N_SVHN = 3


def _random_images(rng: np.random.Generator, n: int, side: int) -> np.ndarray:
    return rng.integers(0, 256, size=(n, side, side, 3), dtype=np.uint8)


def _write_fake_raw(data_dir: pathlib.Path) -> dict[str, np.ndarray]:
    """Writes fake canonical sources in the formats torchvision downloads."""
    rng = np.random.default_rng(0)
    raw = data_dir / "raw"
    stl = raw / "stl10_binary"
    stl.mkdir(parents=True)
    stl_images: dict[str, np.ndarray] = {}
    for split, n in STL_SIZES.items():
        images = _random_images(rng, n, 96)
        stl_images[split] = images
        # Official binaries store images as (N, C, W, H).
        (stl / f"{split}_X.bin").write_bytes(images.transpose(0, 3, 2, 1).tobytes())
        if split != "unlabeled":
            labels = (np.arange(n) % 10 + 1).astype(np.uint8)  # 1-based
            (stl / f"{split}_y.bin").write_bytes(labels.tobytes())

    dtd = raw / "dtd" / "dtd"
    (dtd / "labels").mkdir(parents=True)
    names = ["banded/banded_1.jpg", "banded/banded_2.jpg", "dotted/dotted_1.jpg"]
    names.append("zigzagged/zigzagged_1.jpg")
    for name in names:
        path = dtd / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.fromarray(_random_images(rng, 1, 40)[0]).save(path)  # pyright: ignore[reportUnknownMemberType]
    (dtd / "labels" / "test1.txt").write_text("\n".join(names) + "\n")

    svhn = _random_images(rng, N_SVHN, 32)
    labels = np.array([[10], [1], [5]], dtype=np.uint8)
    scipy.io.savemat(
        raw / "test_32x32.mat", {"X": svhn.transpose(1, 2, 3, 0), "y": labels}
    )
    return stl_images


def _run_app(
    data_dir: pathlib.Path, *overrides: str
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "LEJEPA_NORM_DATA_DIR": str(data_dir),
        "LEJEPA_NORM_RUNS_DIR": str(data_dir / "runs"),
        "LEJEPA_NORM_EMBED_DIR": str(data_dir / "embed"),
        "LEJEPA_NORM_NOTES_DIR": str(data_dir / "notes"),
    }
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "apps" / "prepare_data.py"),
            "--stage",
            "build",
            "--env-file",
            os.devnull,
            "--set",
            *overrides,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_creates_layout_and_corruptions(tmp_path: pathlib.Path) -> None:
    stl_images = _write_fake_raw(tmp_path)
    result = _run_app(
        tmp_path,
        "data.corruptions=[gaussian_noise, pixelate]",
        "data.severities=[1, 3]",
        "data.corruption_seed=7",
        "data.num_workers=2",
    )
    assert result.returncode == 0, result.stderr

    for split, n in STL_SIZES.items():
        images = np.load(layout.images_path(tmp_path, "stl10", split))
        labels = np.load(layout.labels_path(tmp_path, "stl10", split))
        assert images.dtype == np.uint8 and images.shape == (n, 96, 96, 3)
        assert labels.dtype == np.int64 and labels.shape == (n,)
        np.testing.assert_array_equal(images, stl_images[split])
    assert (np.load(layout.labels_path(tmp_path, "stl10", "unlabeled")) == -1).all()
    train_labels = np.load(layout.labels_path(tmp_path, "stl10", "train"))
    np.testing.assert_array_equal(train_labels, np.arange(4))

    for dataset, n in (("dtd", N_DTD), ("svhn", N_SVHN)):
        images = np.load(layout.images_path(tmp_path, dataset, "test"))
        labels = np.load(layout.labels_path(tmp_path, dataset, "test"))
        assert images.dtype == np.uint8 and images.shape == (n, 96, 96, 3)
        assert labels.dtype == np.int64 and labels.shape == (n,)
    np.testing.assert_array_equal(
        np.load(layout.labels_path(tmp_path, "svhn", "test")), [0, 1, 5]
    )
    np.testing.assert_array_equal(
        np.load(layout.labels_path(tmp_path, "dtd", "test")), [0, 0, 1, 2]
    )

    test_images = stl_images["test"]
    for name in ("gaussian_noise", "pixelate"):
        for severity in (1, 3):
            path = layout.corrupted_images_path(tmp_path, name, severity)
            built = np.load(path)
            assert built.dtype == np.uint8 and built.shape == test_images.shape
            expected = np.stack(
                [
                    corruptions.corrupt(img, name, severity, seed=7, index=i)
                    for i, img in enumerate(test_images)
                ]
            )
            np.testing.assert_array_equal(built, expected)
    assert not layout.corrupted_images_path(tmp_path, "fog", 1).exists()
    assert (tmp_path / "stl10c" / "config.resolved.yaml").exists()

    # Readers over the layout, with the multi-view transform.
    cfg = config.DataConfig()
    ds = datasets.load_split(tmp_path, "stl10", "train", views.MultiViewTransform(cfg))
    item, label = ds[2]
    assert label == 2
    assert [tuple(v.shape) for v in item] == [(3, 96, 96)] * 2 + [(3, 48, 48)] * 4
    corrupted = datasets.load_corrupted(tmp_path, "pixelate", 3)
    assert len(corrupted) == STL_SIZES["test"]
    assert len(datasets.load_pretraining(tmp_path)) == 10


def test_corruption_deterministic_and_shape_preserving() -> None:
    image = _random_images(np.random.default_rng(1), 1, 96)[0]
    for name in corruptions.available():
        first = corruptions.corrupt(image, name, 3, seed=5, index=11)
        again = corruptions.corrupt(image, name, 3, seed=5, index=11)
        assert first.shape == image.shape and first.dtype == np.uint8, name
        np.testing.assert_array_equal(first, again, err_msg=name)
        assert not np.array_equal(first, image), name

    # Stochastic corruptions depend on seed and image index; deterministic ones
    # (e.g. contrast) do not.
    a = corruptions.corrupt(image, "gaussian_noise", 3, seed=5, index=11)
    b = corruptions.corrupt(image, "gaussian_noise", 3, seed=6, index=11)
    c = corruptions.corrupt(image, "gaussian_noise", 3, seed=5, index=12)
    assert not np.array_equal(a, b) and not np.array_equal(a, c)


def test_blur_corruptions_work_on_modern_skimage() -> None:
    """Regression: upstream used `multichannel=True`, removed from scikit-image."""
    image = _random_images(np.random.default_rng(2), 1, 96)[0]
    for name in ("gaussian_blur", "glass_blur"):
        out = corruptions.corrupt(image, name, 2)
        assert out.std() < image.std()


def test_corruption_name_resolution() -> None:
    cfg = config.DataConfig()
    assert corruptions.resolve_names(cfg) == cfg.corruptions
    full = config.DataConfig(full_corruption_set=True)
    assert set(cfg.corruptions) < set(corruptions.resolve_names(full))
    with pytest.raises(KeyError):
        corruptions.resolve_names(config.DataConfig(corruptions=("nope",)))


def test_multiview_count_dtype_and_normalized_range() -> None:
    image = PIL.Image.fromarray(_random_images(np.random.default_rng(3), 1, 96)[0])  # pyright: ignore[reportUnknownMemberType]
    cfg = config.DataConfig(num_global_views=1, num_local_views=2)
    out = views.MultiViewTransform(cfg)(image)
    assert len(out) == 3
    for view in out:
        assert view.dtype == torch.float32
        # Normalized [0, 1] pixels lie in [-mean/std, (1-mean)/std] per channel.
        lo = min(-m / s for m, s in zip(cfg.mean, cfg.std, strict=True))
        hi = max((1 - m) / s for m, s in zip(cfg.mean, cfg.std, strict=True))
        assert lo - 1e-4 <= float(view.min()) and float(view.max()) <= hi + 1e-4


def test_views_and_center_view_share_normalization() -> None:
    cfg = config.DataConfig(
        mean=(0.5, 0.4, 0.3), std=(0.25, 0.2, 0.1), num_local_views=1
    )
    # A black image stays exactly black under every augmentation (crop, flip,
    # jitter, grayscale, blur), so every view must equal -mean / std per channel.
    image = PIL.Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8))
    expected = torch.tensor([-2.0, -2.0, -3.0]).view(3, 1, 1)
    outputs = [views.CenterTransform(cfg)(image), *views.MultiViewTransform(cfg)(image)]
    for out in outputs:
        assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def test_build_resumes_after_interruption(tmp_path: pathlib.Path) -> None:
    """A rerun completes a killed build and keeps every array it finished."""
    _write_fake_raw(tmp_path)
    overrides = (
        "data.corruptions=[gaussian_noise, pixelate]",
        "data.severities=[1]",
        "data.num_workers=1",
    )
    first = _run_app(tmp_path, *overrides)
    assert first.returncode == 0, first.stderr
    stl_test = layout.images_path(tmp_path, "stl10", "test")
    noise = layout.corrupted_images_path(tmp_path, "gaussian_noise", 1)
    pixelate = layout.corrupted_images_path(tmp_path, "pixelate", 1)
    expected = {p: np.load(p) for p in (stl_test, noise, pixelate)}
    expected_labels = np.load(layout.labels_path(tmp_path, "stl10", "test"))

    # What a kill can leave behind: arrays never started, `.partial` files cut off
    # mid-write, and (at worst) labels written but not yet the images.
    for path in (stl_test, pixelate):
        path.unlink()
        path.with_name(f"{path.stem}.partial.npy").write_bytes(b"cut off")
    finished = noise.stat().st_mtime_ns

    second = _run_app(tmp_path, *overrides)
    assert second.returncode == 0, second.stderr
    assert noise.stat().st_mtime_ns == finished  # finished arrays are skipped
    for path, array in expected.items():
        np.testing.assert_array_equal(np.load(path), array)
    np.testing.assert_array_equal(
        np.load(layout.labels_path(tmp_path, "stl10", "test")), expected_labels
    )
    assert not list(tmp_path.rglob("*.partial.npy"))

    # The images file is the marker the build skips on, so a kill between the two
    # writes of a split must never leave images without labels.
    calls: list[pathlib.Path] = []

    def die_after_first_save(path: pathlib.Path, array: np.ndarray) -> None:
        if calls:
            raise KeyboardInterrupt  # killed before the second write
        calls.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    images = np.zeros((2, 96, 96, 3), dtype=np.uint8)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(prepare, "_save", die_after_first_save)
        with pytest.raises(KeyboardInterrupt):
            prepare._save_split(tmp_path, "fake", "x", images, np.zeros(2))  # pyright: ignore[reportPrivateUsage]
    assert not layout.images_path(tmp_path, "fake", "x").exists()
