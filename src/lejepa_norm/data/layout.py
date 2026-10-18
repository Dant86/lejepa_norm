"""On-disk layout of prepared data under `LEJEPA_NORM_DATA_DIR`.

Written by `apps/prepare_data.py --stage build`, read by training and embedding.
Images are uint8 arrays of shape `(N, H, W, 3)`, stored as `.npy` so they can be
memory-mapped; labels are int64 arrays of shape `(N,)`.

    <data_dir>/<dataset>/<split>_images.npy
    <data_dir>/<dataset>/<split>_labels.npy
    <data_dir>/stl10c/<corruption>/<severity>/images.npy   # labels: stl10 test

Datasets: `stl10` (splits `unlabeled`, `train`, `test`), `dtd` (`test`),
`svhn` (`test`; resized to 96x96 at build time).
"""

import os
import pathlib

STL10_C_DIR = "stl10c"


def images_path(
    data_dir: str | os.PathLike[str], dataset: str, split: str
) -> pathlib.Path:
    """Path of the uint8 image array for `dataset`/`split`."""
    return pathlib.Path(data_dir) / dataset / f"{split}_images.npy"


def labels_path(
    data_dir: str | os.PathLike[str], dataset: str, split: str
) -> pathlib.Path:
    """Path of the int64 label array for `dataset`/`split`."""
    return pathlib.Path(data_dir) / dataset / f"{split}_labels.npy"


def corrupted_images_path(
    data_dir: str | os.PathLike[str], corruption: str, severity: int
) -> pathlib.Path:
    """Path of STL-10 test images under `corruption` at `severity` (1-5)."""
    return (
        pathlib.Path(data_dir)
        / STL10_C_DIR
        / corruption
        / str(severity)
        / ("images.npy")
    )
