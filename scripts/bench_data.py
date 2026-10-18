"""Diagnoses data-loading throughput: filesystem vs. CPU vs. GPU augmentation.

Run it where training runs (e.g. a short SLURM job) to see which stage limits
images/s. Reads `LEJEPA_NORM_*` paths from the environment / `.env`.

Stages measured:
* filesystem reads and the PIL reference pipeline (`data.views`) in one process
  and through a DataLoader (skip with `--skip-cpu`);
* device-side augmentation (`data.gpu_views.MultiViewAugment`, all training
  views of a uint8 batch already on the GPU), when CUDA is available;
* with `--train-like`: the full training input path (in-RAM gather of random
  batches by the prefetch thread -> pinned host -> GPU -> augmentation), as in
  `train.loop`.

Usage: python scripts/bench_data.py [--workers N] [--batches B] [--env-file F]
           [--skip-cpu] [--gpu-batches G] [--train-like]
"""

import argparse
import os
import pathlib
import time

import numpy
import torch
import torch.utils.data

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.data import batches
from lejepa_norm.data import datasets
from lejepa_norm.data import gpu_views
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_augmentation(
    images: numpy.ndarray,
    cfg: config.DataConfig,
    num_batches: int,
    device: torch.device,
) -> float:
    """images/s of `MultiViewAugment` on uint8 batches already on `device`."""
    augment = gpu_views.MultiViewAugment(cfg)
    generator = torch.Generator(device=device).manual_seed(0)
    batch = torch.from_numpy(numpy.array(images[:BATCH_SIZE])).to(device)  # pyright: ignore[reportUnknownMemberType]
    for _ in range(5):  # warmup (kernel selection, allocator)
        augment(batch, generator)
    _sync(device)
    start = time.perf_counter()
    for _ in range(num_batches):
        augment(batch, generator)
    _sync(device)
    return num_batches * BATCH_SIZE / (time.perf_counter() - start)


def _train_like(
    data_dir: pathlib.Path,
    cfg: config.DataConfig,
    num_batches: int,
    device: torch.device,
) -> None:
    """Full input path of `train.loop`: gather -> pinned -> GPU -> augmentation."""
    start = time.perf_counter()
    images = datasets.load_pretraining_images(data_dir)
    print(
        f"train-like: loaded {len(images)} images into RAM in "
        f"{time.perf_counter() - start:.1f}s"
    )
    order = numpy.random.default_rng(0).permutation(len(images))
    index_batches = [
        order[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] for i in range(num_batches + 5)
    ]
    start = time.perf_counter()
    for indices in index_batches[:20]:
        images.gather(indices)
    rate = 20 * BATCH_SIZE / (time.perf_counter() - start)
    print(f"train-like: CPU gather only: {rate:.0f} images/s")
    augment = gpu_views.MultiViewAugment(cfg)
    generator = torch.Generator(device=device).manual_seed(0)
    pin = device.type == "cuda"
    with batches.Prefetcher(images, index_batches, pin_memory=pin) as prefetcher:
        for step, cpu_batch in enumerate(prefetcher):
            if step == 5:  # after warmup
                _sync(device)
                start = time.perf_counter()
            augment(cpu_batch.to(device, non_blocking=pin), generator)
    _sync(device)
    rate = num_batches * BATCH_SIZE / (time.perf_counter() - start)
    print(
        f"train-like: gather + transfer + augmentation on {device}: {rate:.0f} images/s"
    )


def main() -> None:
    """Runs all measurements and prints a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--env-file", type=pathlib.Path, default=None)
    parser.add_argument(
        "--skip-cpu", action="store_true", help="skip the PIL/DataLoader stages"
    )
    parser.add_argument("--gpu-batches", type=int, default=50)
    parser.add_argument(
        "--device",
        default=None,
        help="device for the augmentation stages (default: cuda if available; "
        "'cpu' runs them on the CPU, e.g. to check the script)",
    )
    parser.add_argument(
        "--train-like",
        action="store_true",
        help="also time the full training input path (loads the arrays into RAM)",
    )
    args = parser.parse_args()
    data_dir = paths.load_paths(args.env_file).data_dir
    cfg = config.DataConfig()
    file = layout.images_path(data_dir, "stl10", "unlabeled")

    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    usable = len(affinity(0)) if affinity is not None else os.cpu_count()
    print(f"cpus: os={os.cpu_count()} usable={usable}")
    cuda = torch.cuda.is_available()
    print(f"cuda: {torch.cuda.get_device_name() if cuda else 'not available'}")
    if not args.skip_cpu:
        _cpu_stages(data_dir, cfg, file, args.workers, args.batches)
    if args.device is None and not cuda:
        print(
            "GPU augmentation: skipped (no CUDA device); run on a GPU node, or "
            "pass --device cpu to time the same code on the CPU"
        )
        return
    device = torch.device(args.device or "cuda")
    images = numpy.load(file, mmap_mode="r")
    rate = _gpu_augmentation(images, cfg, args.gpu_batches, device)
    views_per_image = cfg.num_global_views + cfg.num_local_views
    print(
        f"augmentation on {device}, batch {BATCH_SIZE}, {views_per_image} views "
        f"(uint8 already on the device): {rate:.0f} images/s"
    )
    if args.train_like:
        _train_like(data_dir, cfg, args.gpu_batches, device)


def _cpu_stages(
    data_dir: pathlib.Path,
    cfg: config.DataConfig,
    file: pathlib.Path,
    workers: int,
    num_batches: int,
) -> None:
    """Filesystem reads and the PIL reference pipeline."""
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

    first, rate = _timed_loader(dataset, workers, num_batches)
    print(
        f"DataLoader memmap, {workers} workers: first batch {first:.1f}s, "
        f"{rate:.0f} images/s"
    )
    n = BATCH_SIZE * (num_batches + 4)
    in_memory = datasets.ImageArrayDataset(
        numpy.array(images[:n]), numpy.full(n, -1, dtype=numpy.int64), transform
    )
    first, rate = _timed_loader(in_memory, workers, num_batches)
    print(
        f"DataLoader in RAM, {workers} workers: first batch {first:.1f}s, "
        f"{rate:.0f} images/s"
    )


if __name__ == "__main__":
    main()
