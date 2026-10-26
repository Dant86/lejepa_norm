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
    """

    z: numpy.typing.NDArray[numpy.float32]
    h: numpy.typing.NDArray[numpy.float32]
    image_id: numpy.typing.NDArray[numpy.int64]
    label: numpy.typing.NDArray[numpy.int64]
    corruption: numpy.typing.NDArray[numpy.str_]
    severity: numpy.typing.NDArray[numpy.int8]

    def __post_init__(self) -> None:
        """Validates that all arrays agree on N and K."""
        n, views = self.z.shape[:2]
        if self.z.ndim != 3 or self.h.ndim != 3 or self.h.shape[:2] != (n, views):
            raise ValueError("z and h must be (N, K + 1, d) with matching N, K")
        for name in ("image_id", "label", "corruption", "severity"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")


def save_shard(path: str | os.PathLike[str], shard: EmbeddingShard) -> None:
    """Writes `shard` as an uncompressed `.npz` file, atomically.

    The data goes to `<path>.partial` first and is then renamed, so a shard that
    exists is complete even if the writer was killed.
    """
    dest = pathlib.Path(path)
    partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
    with partial.open("wb") as f:
        numpy.savez(f, **dataclasses.asdict(shard))
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


def load_center_shard(path: str | os.PathLike[str]) -> CenterShard:
    """Reads only the center-view `z` and the metadata of a shard.

    `.npz` members are read lazily, so the backbone feature `h` (about two thirds
    of the file) is never loaded and only the center slice of `z` is kept.
    """
    with numpy.load(path, allow_pickle=False) as arrays:
        return CenterShard(
            z=numpy.ascontiguousarray(arrays["z"][:, CENTER_VIEW]),
            image_id=arrays["image_id"],
            label=arrays["label"],
            corruption=arrays["corruption"],
            severity=arrays["severity"],
        )


def load_shard(path: str | os.PathLike[str]) -> EmbeddingShard:
    """Reads a shard written by `save_shard`, validating shapes."""
    with numpy.load(path, allow_pickle=False) as arrays:
        return EmbeddingShard(
            **{f.name: arrays[f.name] for f in dataclasses.fields(EmbeddingShard)}
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
