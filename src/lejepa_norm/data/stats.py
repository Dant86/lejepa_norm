"""Per-channel mean/std of the pretraining images, for `DataConfig.mean/std`.

Computed over STL-10 `unlabeled` + `train` (the images the model is trained on;
`test` is never included) from the prepared uint8 arrays, in float64 and in chunks
over memory maps, so memory use stays flat. Statistics are on the `[0, 1]` pixel
scale and use the population standard deviation, matching what
`torchvision.transforms.Normalize` expects after `ToTensor`.
"""

import json
import os
import pathlib

import numpy as np

from lejepa_norm.data import layout

PRETRAINING_SPLITS = (("stl10", "unlabeled"), ("stl10", "train"))
STATS_FILE = "channel_stats.json"


def channel_stats(
    data_dir: str | os.PathLike[str], chunk_size: int = 4096
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Mean and std per RGB channel over the pretraining images.

    Args:
        data_dir: Prepared data directory (`LEJEPA_NORM_DATA_DIR`).
        chunk_size: Images read per step.

    Returns:
        `(mean, std)`, each a 3-tuple on the `[0, 1]` scale.
    """
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    pixels = 0
    for dataset, split in PRETRAINING_SPLITS:
        images = np.load(layout.images_path(data_dir, dataset, split), mmap_mode="r")
        for start in range(0, len(images), chunk_size):
            chunk = np.asarray(images[start : start + chunk_size], dtype=np.float64)
            chunk /= 255.0
            total += chunk.sum(axis=(0, 1, 2))
            total_sq += np.square(chunk).sum(axis=(0, 1, 2))
            pixels += chunk.shape[0] * chunk.shape[1] * chunk.shape[2]
    mean = total / pixels
    std = np.sqrt(np.maximum(total_sq / pixels - np.square(mean), 0.0))
    return (
        (float(mean[0]), float(mean[1]), float(mean[2])),
        (float(std[0]), float(std[1]), float(std[2])),
    )


def write_channel_stats(
    data_dir: str | os.PathLike[str],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> pathlib.Path:
    """Writes the statistics next to the prepared STL-10 arrays; returns the path."""
    path = pathlib.Path(data_dir) / "stl10" / STATS_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {"mean": list(mean), "std": list(std), "splits": ["unlabeled", "train"]},
            indent=2,
        )
    )
    tmp.replace(path)
    return path


def _fmt(values: tuple[float, float, float]) -> str:
    return "[" + ",".join(f"{v:.4f}" for v in values) + "]"


def override_args(
    mean: tuple[float, float, float], std: tuple[float, float, float]
) -> str:
    """The `--set` arguments that apply the statistics to a config."""
    return f"--set data.mean={_fmt(mean)} --set data.std={_fmt(std)}"
