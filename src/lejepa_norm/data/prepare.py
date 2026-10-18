# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Download canonical datasets and build the prepared layout (`data.layout`).

Two stages, mirroring `apps/prepare_data.py`:

* `download`: network access, intended for the login node. Fetches the canonical
  archives into `<data_dir>/raw/` with torchvision.
* `build`: offline and CPU-heavy (run under SLURM). Converts the raw files to
  uint8 `.npy` arrays plus int64 labels and generates STL-10-C from the STL-10
  test set. Existing outputs are skipped, so an interrupted job can be resubmitted.
"""

import concurrent.futures
import json
import os
import pathlib

import numpy as np
import PIL.Image
import scipy.io
import torchvision.datasets

from lejepa_norm import config
from lejepa_norm.data import corruptions
from lejepa_norm.data import layout

RAW_DIR_NAME = "raw"
IMAGE_SIZE = 96
_STL10_SIDE = 96
_CHUNK = 250


def raw_dir(data_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Directory holding downloaded canonical archives."""
    return pathlib.Path(data_dir) / RAW_DIR_NAME


def download(data_dir: str | os.PathLike[str]) -> None:
    """Fetches STL-10 (all splits), DTD and SVHN test into `<data_dir>/raw/`.

    Needs network access; run on the login node. Already-downloaded files are
    verified (md5) and not fetched again.
    """
    root = raw_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test", "unlabeled"):
        torchvision.datasets.STL10(str(root), split=split, download=True)
    torchvision.datasets.DTD(str(root), split="test", partition=1, download=True)
    torchvision.datasets.SVHN(str(root), split="test", download=True)


def _save(path: pathlib.Path, array: np.ndarray) -> None:
    """Writes `array` to `path` atomically (a half-written file never appears)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.partial.npy")
    np.save(partial, array)
    os.replace(partial, path)


def _save_split(
    data_dir: pathlib.Path,
    dataset: str,
    split: str,
    images: np.ndarray,
    labels: np.ndarray,
) -> None:
    if images.dtype != np.uint8 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"{dataset}/{split}: bad images {images.dtype} {images.shape}")
    # Labels first: the images file is the "split done" marker the build skips on,
    # so it must only appear once the labels are on disk.
    _save(layout.labels_path(data_dir, dataset, split), labels.astype(np.int64))
    _save(layout.images_path(data_dir, dataset, split), images)


def _resize(image: PIL.Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), PIL.Image.Resampling.BICUBIC
    )
    return np.asarray(resized, dtype=np.uint8)


def read_stl10_raw(raw: pathlib.Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Reads one STL-10 split from the official binaries (labels -1 if unlabeled).

    Returns:
        uint8 images `(N, 96, 96, 3)` and int64 labels `(N,)` in `0..9`.
    """
    folder = raw / "stl10_binary"
    flat = np.fromfile(folder / f"{split}_X.bin", dtype=np.uint8)
    # Binary layout is (N, C, W, H) column-major images; this is (N, H, W, C).
    images = flat.reshape(-1, 3, _STL10_SIDE, _STL10_SIDE).transpose(0, 3, 2, 1)
    if split == "unlabeled":
        labels = np.full(len(images), -1, dtype=np.int64)
    else:
        labels = np.fromfile(folder / f"{split}_y.bin", dtype=np.uint8).astype(np.int64)
        labels -= 1
    return np.ascontiguousarray(images), labels


def read_svhn_raw(raw: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads SVHN cropped digits (test) as 32x32 uint8 and labels (10 -> 0)."""
    mat = scipy.io.loadmat(str(raw / "test_32x32.mat"))
    images = np.ascontiguousarray(np.transpose(mat["X"], (3, 0, 1, 2)))
    labels = mat["y"].astype(np.int64).squeeze(-1)
    labels[labels == 10] = 0
    return images, labels


def read_dtd_raw(raw: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads the DTD partition-1 test split, resized to 96x96."""
    folder = raw / "dtd" / "dtd"
    names = (folder / "labels" / "test1.txt").read_text().split()
    classes = sorted({name.split("/")[0] for name in names})
    class_index = {cls: i for i, cls in enumerate(classes)}
    images = np.empty((len(names), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, name in enumerate(names):
        with PIL.Image.open(folder / "images" / name) as img:
            images[i] = _resize(img)
    labels = np.array([class_index[n.split("/")[0]] for n in names], dtype=np.int64)
    return images, labels


def build_stl10(data_dir: pathlib.Path) -> None:
    """Writes the three STL-10 splits (`unlabeled`, `train`, `test`)."""
    for split in ("unlabeled", "train", "test"):
        if layout.images_path(data_dir, "stl10", split).exists():
            continue
        images, labels = read_stl10_raw(raw_dir(data_dir), split)
        _save_split(data_dir, "stl10", split, images, labels)


def build_dtd(data_dir: pathlib.Path) -> None:
    """Writes DTD `test`, resized to 96x96."""
    if layout.images_path(data_dir, "dtd", "test").exists():
        return
    images, labels = read_dtd_raw(raw_dir(data_dir))
    _save_split(data_dir, "dtd", "test", images, labels)


def build_svhn(data_dir: pathlib.Path) -> None:
    """Writes SVHN `test`, resized from 32x32 to 96x96."""
    if layout.images_path(data_dir, "svhn", "test").exists():
        return
    small, labels = read_svhn_raw(raw_dir(data_dir))
    images = np.stack([_resize(PIL.Image.fromarray(img)) for img in small])  # pyright: ignore[reportUnknownMemberType]
    _save_split(data_dir, "svhn", "test", images, labels)


def _corrupt_chunk(
    source: str, name: str, severity: int, seed: int, start: int, stop: int
) -> np.ndarray:
    images = np.load(source, mmap_mode="r")
    return np.stack(
        [
            corruptions.corrupt(
                np.asarray(images[i]), name, severity, seed=seed, index=i
            )
            for i in range(start, stop)
        ]
    )


def build_stl10c(data_dir: pathlib.Path, cfg: config.DataConfig) -> None:
    """Generates STL-10-C from the STL-10 test images.

    Each (corruption, severity) is written to its own array, skipped if present.
    Per-image randomness depends only on `(corruption_seed, image index,
    corruption, severity)`, so the result does not depend on `num_workers`.
    """
    source = layout.images_path(data_dir, "stl10", "test")
    n = len(np.load(source, mmap_mode="r"))
    names = corruptions.resolve_names(cfg)
    for severity in cfg.severities:
        if severity not in corruptions.SEVERITIES:
            raise ValueError(f"severity {severity} not in {corruptions.SEVERITIES}")
    chunks = [(s, min(s + _CHUNK, n)) for s in range(0, n, _CHUNK)]
    with concurrent.futures.ProcessPoolExecutor(max(cfg.num_workers, 1)) as pool:
        for name in names:
            for severity in cfg.severities:
                out_path = layout.corrupted_images_path(data_dir, name, severity)
                if out_path.exists():
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                partial = out_path.with_name("images.partial.npy")
                out = np.lib.format.open_memmap(
                    partial,
                    mode="w+",
                    dtype=np.uint8,
                    shape=(n, IMAGE_SIZE, IMAGE_SIZE, 3),
                )
                futures = {
                    pool.submit(
                        _corrupt_chunk,
                        str(source),
                        name,
                        severity,
                        cfg.corruption_seed,
                        start,
                        stop,
                    ): start
                    for start, stop in chunks
                }
                for future, start in futures.items():
                    result = future.result()
                    out[start : start + len(result)] = result
                out.flush()
                del out
                os.replace(partial, out_path)
    meta = {
        "corruptions": list(names),
        "severities": list(cfg.severities),
        "corruption_seed": cfg.corruption_seed,
        "full_corruption_set": cfg.full_corruption_set,
        "num_images": n,
    }
    meta_path = pathlib.Path(data_dir) / layout.STL10_C_DIR / "meta.json"
    partial_meta = meta_path.with_name("meta.partial.json")
    partial_meta.write_text(json.dumps(meta, indent=2))
    os.replace(partial_meta, meta_path)


def build(data_dir: str | os.PathLike[str], cfg: config.DataConfig) -> None:
    """Builds every prepared dataset and STL-10-C from the raw downloads."""
    root = pathlib.Path(data_dir)
    build_stl10(root)
    build_dtd(root)
    build_svhn(root)
    build_stl10c(root, cfg)
