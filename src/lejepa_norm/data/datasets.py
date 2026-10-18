"""Dataset readers over the prepared layout (`data.layout`).

Arrays are opened as read-only memory maps, so many DataLoader workers share one
page cache instead of each loading the data.
"""

import collections.abc
import os
import typing

import numpy as np
import PIL.Image
import torch.utils.data

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


def load_split[T](
    data_dir: str | os.PathLike[str],
    dataset: str,
    split: str,
    transform: collections.abc.Callable[[PIL.Image.Image], T] | None = None,
) -> ImageArrayDataset[T]:
    """Opens `dataset`/`split` (e.g. `stl10`/`train`) from the prepared layout."""
    return ImageArrayDataset(
        np.load(layout.images_path(data_dir, dataset, split), mmap_mode="r"),
        np.load(layout.labels_path(data_dir, dataset, split)),
        transform,
    )


def load_corrupted[T](
    data_dir: str | os.PathLike[str],
    corruption: str,
    severity: int,
    transform: collections.abc.Callable[[PIL.Image.Image], T] | None = None,
) -> ImageArrayDataset[T]:
    """Opens STL-10-C (`corruption`, `severity`) with the STL-10 test labels."""
    return ImageArrayDataset(
        np.load(
            layout.corrupted_images_path(data_dir, corruption, severity),
            mmap_mode="r",
        ),
        np.load(layout.labels_path(data_dir, "stl10", "test")),
        transform,
    )


def load_pretraining[T](
    data_dir: str | os.PathLike[str],
    transform: collections.abc.Callable[[PIL.Image.Image], T] | None = None,
) -> torch.utils.data.ConcatDataset[tuple[T, int]]:
    """STL-10 `unlabeled` + `train` (labels present but unused in pretraining)."""
    return torch.utils.data.ConcatDataset(
        [
            load_split(data_dir, "stl10", "unlabeled", transform),
            load_split(data_dir, "stl10", "train", transform),
        ]
    )
