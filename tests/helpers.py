"""Helpers shared by the workflow tests (importing registers `tiny_cnn`)."""

import pathlib

import numpy

from lejepa_norm.data import layout
import tiny_backbone

# Settings that make the real pipeline cheap: tiny model, small views.
TINY_MODEL = (
    f"model.backbone={tiny_backbone.NAME}",
    "model.projector_hidden=[32]",
    "model.embed_dim=16",
    "loss.num_slices=16",
    "data.num_workers=0",
    "data.global_size=32",
    "data.local_size=16",
)
# The prepared fake data these settings go with: global views (and the embedding
# center view) are the full image, as with the real 96 px STL-10 at global_size=96.
IMAGE_SIZE = 32


def write_fake_stl10(data_dir: pathlib.Path, sizes: dict[str, int]) -> None:
    """Writes fake prepared STL-10 splits of `IMAGE_SIZE` px."""
    rng = numpy.random.default_rng(0)
    for split, n in sizes.items():
        images = rng.integers(
            0, 256, size=(n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=numpy.uint8
        )
        labels = numpy.arange(n, dtype=numpy.int64) % 2
        if split == "unlabeled":
            labels[:] = -1
        path = layout.images_path(data_dir, "stl10", split)
        path.parent.mkdir(parents=True, exist_ok=True)
        numpy.save(path, images)
        numpy.save(layout.labels_path(data_dir, "stl10", split), labels)
