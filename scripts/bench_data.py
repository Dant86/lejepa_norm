"""Diagnoses data-loading throughput: filesystem vs. CPU vs. DataLoader setup.

Run it where training runs (e.g. a short SLURM job) to see which stage limits
images/s. Reads `LEJEPA_NORM_*` paths from the environment / `.env`.

Usage: python scripts/bench_data.py [--workers N] [--batches B] [--env-file F]
"""

import argparse
import os
import pathlib
import time

import numpy
import torch.utils.data

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.data import datasets
from lejepa_norm.data import layout
from lejepa_norm.data import views

BATCH_SIZE = 256


def _timed_loader(
    dataset: torch.utils.data.Dataset[object], workers: int, batches: int
) -> tuple[float, float]:
    """Returns (seconds to first batch, steady-state images/s over later batches)."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
    )
    start = time.perf_counter()
    iterator = iter(loader)
    next(iterator)
    first = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(batches):
        next(iterator)
    return first, batches * BATCH_SIZE / (time.perf_counter() - start)


def main() -> None:
    """Runs all measurements and prints a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--env-file", type=pathlib.Path, default=None)
    args = parser.parse_args()
    data_dir = paths.load_paths(args.env_file).data_dir
    cfg = config.DataConfig()
    file = layout.images_path(data_dir, "stl10", "unlabeled")

    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    usable = len(affinity(0)) if affinity is not None else os.cpu_count()
    print(f"cpus: os={os.cpu_count()} usable={usable}")

    start = time.perf_counter()
    with open(file, "rb") as handle:
        nbytes = len(handle.read(512 * 2**20))
    print(f"sequential read: {nbytes / 2**20 / (time.perf_counter() - start):.0f} MB/s")

    images = numpy.load(file, mmap_mode="r")
    indices = numpy.random.default_rng(0).integers(0, len(images), 1000)
    for label in ("cold", "warm"):
        start = time.perf_counter()
        for i in indices:
            numpy.array(images[i])
        ms = (time.perf_counter() - start) / len(indices) * 1000
        print(f"memmap random read ({label}): {ms:.2f} ms/image")

    transform = views.MultiViewTransform(cfg)
    dataset = datasets.load_split(
        data_dir, "stl10", "unlabeled", transform, in_memory=False
    )
    sample = [dataset.images[i] for i in range(64)]  # pyright: ignore[reportAttributeAccessIssue]
    start = time.perf_counter()
    for array in sample:
        transform(datasets.PIL.Image.fromarray(array))  # pyright: ignore[reportUnknownMemberType]
    ms = (time.perf_counter() - start) / len(sample) * 1000
    print(f"augmentation, 1 process: {ms:.1f} ms/image ({1000 / ms:.0f} images/s)")

    first, rate = _timed_loader(dataset, args.workers, args.batches)
    print(
        f"DataLoader memmap, {args.workers} workers: first batch {first:.1f}s, "
        f"{rate:.0f} images/s"
    )
    n = BATCH_SIZE * (args.batches + 4)
    in_memory = datasets.ImageArrayDataset(
        numpy.array(images[:n]), numpy.full(n, -1, dtype=numpy.int64), transform
    )
    first, rate = _timed_loader(in_memory, args.workers, args.batches)
    print(
        f"DataLoader in RAM, {args.workers} workers: first batch {first:.1f}s, "
        f"{rate:.0f} images/s"
    )


if __name__ == "__main__":
    main()
