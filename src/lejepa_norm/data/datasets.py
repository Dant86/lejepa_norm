"""Dataset readers over the prepared layout (`data.layout`).

Arrays are read fully into RAM by default (`in_memory=True`). Memory-mapping looks
cheaper, but on the cluster's network filesystem each random read of a ~27 KB image
cost ~100 ms (measured with `scripts/bench_data.py`: 83 images/s through a
DataLoader vs. 1200 images/s from RAM). A sequential read of the whole array is fast
(~11 s for STL-10 `unlabeled`), and forked DataLoader workers share the array
copy-on-write, so memory is not multiplied. (Training and embedding no longer use
DataLoader workers: they gather uint8 batches and augment on the device, see
`data.batches` and `data.gpu_views`.) Pass `in_memory=False` for a read-only
memory map, e.g. for arrays too large for RAM.
"""

import collections.abc
import os
import pathlib
import typing

import numpy as np
import PIL.Image
import torch.utils.data

from lejepa_norm.data import batches
from lejepa_norm.data import layout


class ImageArrayDataset[T](torch.utils.data.Dataset[tuple[T, int]]):
    """Images from a uint8 `(N, H, W, 3)` array with int64 labels.

    Each item is `(transform(PIL image), label)`; without a transform the item is
    the PIL image itself. With `MultiViewTransform` the first element is a list of
    view tensors, which the default collate turns into one batched tensor per view.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: collections.abc.Callable[[PIL.Image.Image], T] | None = None,
    ) -> None:
        """Wraps `images` and `labels`; `transform` is applied to PIL images."""
        if len(images) != len(labels):
            raise ValueError(f"{len(images)} images but {len(labels)} labels")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Number of images."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[T, int]:
        """Returns `(transformed image, label)` for `index`."""
        image = PIL.Image.fromarray(np.asarray(self.images[index]))  # pyright: ignore[reportUnknownMemberType]
        item = image if self.transform is None else self.transform(image)
        return typing.cast(T, item), int(self.labels[index])


def _load_images(path: pathlib.Path, in_memory: bool) -> np.ndarray:
    return np.load(path, mmap_mode=None if in_memory else "r")


def load_split[T](
    data_dir: str | os.PathLike[str],
    dataset: str,
    split: str,
    transform: collections.abc.Callable[[PIL.Image.Image], T] | None = None,
    in_memory: bool = True,
) -> ImageArrayDataset[T]:
    """Opens `dataset`/`split` (e.g. `stl10`/`train`) from the prepared layout."""
    return ImageArrayDataset(
        _load_images(layout.images_path(data_dir, dataset, split), in_memory),
        np.load(layout.labels_path(data_dir, dataset, split)),
        transform,
    )


def load_corrupted[T](
    data_dir: str | os.PathLike[str],
    corruption: str,
    severity: int,
    transform: collections.abc.Callable[[PIL.Image.Image], T] | None = None,
    in_memory: bool = True,
) -> ImageArrayDataset[T]:
    """Opens STL-10-C (`corruption`, `severity`) with the STL-10 test labels."""
    return ImageArrayDataset(
        _load_images(
            layout.corrupted_images_path(data_dir, corruption, severity), in_memory
        ),
        np.load(layout.labels_path(data_dir, "stl10", "test")),
        transform,
    )


def load_pretraining_images(
    data_dir: str | os.PathLike[str], in_memory: bool = True
) -> batches.ArrayConcat:
    """STL-10 `unlabeled` + `train` images as one uint8 index space (no copy).

    Indices `0..N_unlabeled-1` are `unlabeled`, the rest `train`. Labels are not
    needed for pretraining.
    """
    return batches.ArrayConcat(
        [
            _load_images(layout.images_path(data_dir, "stl10", split), in_memory)
            for split in ("unlabeled", "train")
        ]
    )


def load_labeled_train(
    data_dir: str | os.PathLike[str], in_memory: bool = True
) -> tuple[batches.ArrayConcat, np.ndarray]:
    """STL-10 `train` (5000 labeled images) and its int64 labels.

    Used only by the supervised baseline; every other run pretrains on the labels-free
    `unlabeled` + `train` pool (`load_pretraining_images`).
    """
    images = batches.ArrayConcat(
        [_load_images(layout.images_path(data_dir, "stl10", "train"), in_memory)]
    )
    labels = np.load(layout.labels_path(data_dir, "stl10", "train")).astype(np.int64)
    return images, labels
