"""Embedding-dump shard schema shared by `apps/embed.py` and `apps/analyze.py`.

A shard holds embeddings for `N` images, each with `K + 1` views: view 0 is the
un-augmented center view, views 1..K are random augmentations. The analysis reads
shards only; it never re-runs the model.
"""

import dataclasses
import os

import numpy
import numpy.typing

CENTER_VIEW = 0


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
    """Writes `shard` as an uncompressed `.npz` file."""
    numpy.savez(path, **dataclasses.asdict(shard))


def load_shard(path: str | os.PathLike[str]) -> EmbeddingShard:
    """Reads a shard written by `save_shard`, validating shapes."""
    with numpy.load(path, allow_pickle=False) as arrays:
        return EmbeddingShard(
            **{f.name: arrays[f.name] for f in dataclasses.fields(EmbeddingShard)}
        )
