"""Embedding-dump shard schema shared by `apps/embed.py` and `apps/analyze.py`.

A shard holds embeddings for `N` images, each with `K + 1` views: view 0 is the
un-augmented center view, views 1..K are random augmentations. The analysis reads
shards only; it never re-runs the model.
"""

import dataclasses
import os
import pathlib

import numpy
import numpy.typing

CENTER_VIEW = 0
# Suffix of a shard being written (not matched by `*.npz`).
PARTIAL_SUFFIX = ".partial"
# Columns of `EmbeddingShard.aug`: the parameters of an augmented view
# (`data.gpu_views.ViewParams` fields without the jitter order), in this order.
AUG_FIELDS = (
    "top",
    "left",
    "height",
    "width",
    "flip",
    "jitter",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "grayscale",
    "blur",
    "sigma",
)


@dataclasses.dataclass
class EmbeddingShard:
    """Embeddings and metadata for N images.

    Attributes:
        z: Projector outputs, float32 `(N, K + 1, d_z)`.
        h: Backbone features, float32 `(N, K + 1, d_h)`.
        image_id: int64 `(N,)`, index into the source split.
        label: int64 `(N,)`; analysis only.
        corruption: str `(N,)`; empty string for clean images.
        severity: int8 `(N,)`; 0 for clean images.
        aug: Optional float32 `(N, K, len(AUG_FIELDS))`: the augmentation parameters
            of views 1..K (column `j` is `AUG_FIELDS[j]`; booleans as 0/1). Absent
            (None) in dumps written before this field existed.
    """

    z: numpy.typing.NDArray[numpy.float32]
    h: numpy.typing.NDArray[numpy.float32]
    image_id: numpy.typing.NDArray[numpy.int64]
    label: numpy.typing.NDArray[numpy.int64]
    corruption: numpy.typing.NDArray[numpy.str_]
    severity: numpy.typing.NDArray[numpy.int8]
    aug: numpy.typing.NDArray[numpy.float32] | None = None

    def __post_init__(self) -> None:
        """Validates that all arrays agree on N and K."""
        n, views = self.z.shape[:2]
        if self.z.ndim != 3 or self.h.ndim != 3 or self.h.shape[:2] != (n, views):
            raise ValueError("z and h must be (N, K + 1, d) with matching N, K")
        for name in ("image_id", "label", "corruption", "severity"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        if self.aug is not None and self.aug.shape != (
            n,
            views - 1,
            len(AUG_FIELDS),
        ):
            raise ValueError(
                f"aug must have shape ({n}, {views - 1}, {len(AUG_FIELDS)})"
            )


def save_shard(path: str | os.PathLike[str], shard: EmbeddingShard) -> None:
    """Writes `shard` as an uncompressed `.npz` file, atomically.

    The data goes to `<path>.partial` first and is then renamed, so a shard that
    exists is complete even if the writer was killed.
    """
    dest = pathlib.Path(path)
    partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
    with partial.open("wb") as f:
        arrays = dataclasses.asdict(shard)
        if shard.aug is None:
            del arrays["aug"]
        numpy.savez(f, **arrays)
    os.replace(partial, dest)


@dataclasses.dataclass
class CenterShard:
    """z of the center view only (the memory-light view of an `EmbeddingShard`).

    Attributes:
        z: Center-view projector outputs, float32 `(N, d_z)`.
        image_id: int64 `(N,)`.
        label: int64 `(N,)`.
        corruption: str `(N,)`; empty string for clean images.
        severity: int8 `(N,)`.
    """

    z: numpy.typing.NDArray[numpy.float32]
    image_id: numpy.typing.NDArray[numpy.int64]
    label: numpy.typing.NDArray[numpy.int64]
    corruption: numpy.typing.NDArray[numpy.str_]
    severity: numpy.typing.NDArray[numpy.int8]


def load_center_shard(path: str | os.PathLike[str], space: str = "z") -> CenterShard:
    """Reads only the center-view embedding and the metadata of a shard.

    `.npz` members are read lazily, so only the requested array is loaded and only
    its center slice is kept. `space` is `"z"` (default) or `"h"`; the chosen array
    is returned in the `z` field (only the E2 baselines are ever read as `"h"`).

    Raises:
        ValueError: `space` is neither `"z"` nor `"h"`.
    """
    if space not in ("z", "h"):
        raise ValueError(f"space must be 'z' or 'h', got {space!r}")
    with numpy.load(path, allow_pickle=False) as arrays:
        return CenterShard(
            z=numpy.ascontiguousarray(arrays[space][:, CENTER_VIEW]),
            image_id=arrays["image_id"],
            label=arrays["label"],
            corruption=arrays["corruption"],
            severity=arrays["severity"],
        )


def load_shard(path: str | os.PathLike[str]) -> EmbeddingShard:
    """Reads a shard written by `save_shard`, validating shapes."""
    with numpy.load(path, allow_pickle=False) as arrays:
        return EmbeddingShard(
            **{
                f.name: arrays[f.name]
                for f in dataclasses.fields(EmbeddingShard)
                if f.name in arrays
            }
        )


@dataclasses.dataclass
class ViewShard:
    """z of every view of N images, without the backbone feature `h`.

    Attributes:
        z: Projector outputs, float32 `(N, K + 1, d_z)`; view 0 is the center view.
        image_id: int64 `(N,)`.
        label: int64 `(N,)`.
        corruption: str `(N,)`; empty string for clean images.
        severity: int8 `(N,)`.
    """

    z: numpy.typing.NDArray[numpy.float32]
    image_id: numpy.typing.NDArray[numpy.int64]
    label: numpy.typing.NDArray[numpy.int64]
    corruption: numpy.typing.NDArray[numpy.str_]
    severity: numpy.typing.NDArray[numpy.int8]


def load_view_shard(path: str | os.PathLike[str]) -> ViewShard:
    """Reads `z` (all views) and the metadata of a shard, skipping `h`."""
    with numpy.load(path, allow_pickle=False) as arrays:
        return ViewShard(
            z=arrays["z"],
            image_id=arrays["image_id"],
            label=arrays["label"],
            corruption=arrays["corruption"],
            severity=arrays["severity"],
        )
