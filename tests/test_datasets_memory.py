import pathlib

import numpy

from lejepa_norm.data import datasets
from lejepa_norm.data import layout


def _write(root: pathlib.Path) -> numpy.ndarray:
    images = numpy.random.default_rng(0).integers(
        0, 256, size=(5, 8, 8, 3), dtype=numpy.uint8
    )
    path = layout.images_path(root, "stl10", "train")
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.save(path, images)
    numpy.save(layout.labels_path(root, "stl10", "train"), numpy.arange(5))
    return images


def test_arrays_are_read_into_ram_by_default(tmp_path: pathlib.Path) -> None:
    images = _write(tmp_path)
    in_ram: datasets.ImageArrayDataset[object] = datasets.load_split(
        tmp_path, "stl10", "train"
    )
    mapped: datasets.ImageArrayDataset[object] = datasets.load_split(
        tmp_path, "stl10", "train", in_memory=False
    )
    # Reading into RAM must not leave a file-backed map behind (random reads on
    # the cluster filesystem are what made training ~14x slower).
    assert not isinstance(in_ram.images, numpy.memmap)
    assert isinstance(mapped.images, numpy.memmap)
    numpy.testing.assert_array_equal(in_ram.images, images)
    numpy.testing.assert_array_equal(mapped.images, images)
