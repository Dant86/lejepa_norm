"""Embedding dump: runs a frozen checkpoint over the evaluation sets.

For every image the model is applied to the un-augmented center view (view 0) and
to `eval.num_views` random global-view augmentations (views 1..K). Projector
outputs `z` and backbone features `h` are written as `.npz` shards in the layout
of `eval.dumps`, a few thousand images per shard, so STL-10-C never has to fit in
memory.

Views are computed on the device from uint8 batches (`data.gpu_views`; no
DataLoader workers). Augmentation randomness comes from a device generator seeded
per shard from `(seed, set, corruption, severity, first image index)`, and the
parameters of a whole shard are drawn before it is split into batches, so a rerun
on the same device type reproduces every shard exactly, independent of `--sets`,
`--max-images`, `eval.batch_size` and `data.num_workers`. The model runs in eval
mode (BatchNorm uses running statistics), under `torch.no_grad`, in fp32.

Shards are written atomically, so with `resume=True` an interrupted dump (e.g. a
preempted job) is completed by writing only the shards that are missing; the
result equals an uninterrupted run.
"""

import collections.abc
import dataclasses
import os
import pathlib
import zlib

import numpy
import torch
from torch import nn

from lejepa_norm import config
from lejepa_norm.data import batches
from lejepa_norm.data import datasets
from lejepa_norm.data import gpu_views
from lejepa_norm.data import layout
from lejepa_norm.data import view_spec
from lejepa_norm.eval import dumps
from lejepa_norm.eval import shards
from lejepa_norm.models import checkpoint

SHARD_SIZE = 4096


@dataclasses.dataclass
class _Source:
    """One image array of an evaluation set."""

    dataset: datasets.ImageArrayDataset[object]
    corruption: str
    severity: int


def stl10c_variants(data_dir: str | os.PathLike[str]) -> list[tuple[str, int]]:
    """Every `(corruption, severity)` present under the STL-10-C layout, sorted."""
    root = pathlib.Path(data_dir) / layout.STL10_C_DIR
    found: list[tuple[str, int]] = []
    for images in root.glob("*/*/images.npy"):
        if images.parent.name.isdigit():
            found.append((images.parent.parent.name, int(images.parent.name)))
    if not found:
        raise FileNotFoundError(f"no STL-10-C arrays under {root}")
    return sorted(found)


def _sources(
    eval_set: str, data_dir: str | os.PathLike[str]
) -> collections.abc.Iterator[_Source]:
    """Yields the image arrays of `eval_set`, loading each only when it is reached.

    Arrays are read into RAM (see `data.datasets`), so STL-10-C (50 arrays of about
    220 MB) must not be materialized all at once.
    """
    splits = {
        dumps.STL10_TRAIN: ("stl10", "train"),
        dumps.STL10_TEST: ("stl10", "test"),
        dumps.DTD_TEST: ("dtd", "test"),
        dumps.SVHN_TEST: ("svhn", "test"),
        dumps.PLACES_TEST: ("places365", "val"),
    }
    if eval_set in splits:
        dataset, name = splits[eval_set]
        yield _Source(datasets.load_split(data_dir, dataset, name), "", 0)
    elif eval_set == dumps.STL10_C:
        for name, severity in stl10c_variants(data_dir):
            yield _Source(
                datasets.load_corrupted(data_dir, name, severity),
                name,
                severity,
            )
    else:
        raise ValueError(f"unknown eval set {eval_set!r}; expected {dumps.EVAL_SETS}")


def _shard_seed(seed: int, *key: object) -> int:
    return zlib.crc32("/".join(str(k) for k in (seed, *key)).encode())


@torch.no_grad()
def _embed_shard(
    backbone: nn.Module,
    projector: nn.Module,
    images: numpy.ndarray,
    index: range,
    data_cfg: config.DataConfig,
    num_views: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embeds the center view and `num_views` augmentations of `images[index]`.

    The augmentation parameters of all images are drawn up front (view by view),
    so the result does not depend on `batch_size`.

    Returns:
        `z (N, K + 1, d_z)` and `h (N, K + 1, d_h)` on the CPU, center view first.
    """
    device = generator.device
    spec = view_spec.global_spec(data_cfg)
    height, width = images.shape[1:3]
    params = [
        gpu_views.sample_params(spec, len(index), height, width, generator)
        for _ in range(num_views)
    ]
    zs: list[torch.Tensor] = []
    hs: list[torch.Tensor] = []
    offset = 0
    for batch in batches.contiguous_batches(
        images, batch_size, device, index.start, index.stop
    ):
        rows = slice(offset, offset + batch.shape[0])
        offset += batch.shape[0]
        x = gpu_views.to_float(batch)
        view_list = [gpu_views.normalize(x, data_cfg)]
        view_list += [
            gpu_views.apply_view(x, p.select(rows), spec, data_cfg) for p in params
        ]
        h = torch.stack([backbone(view) for view in view_list], dim=1)  # (b, V, d)
        z = projector(h.flatten(0, 1)).reshape(*h.shape[:2], -1)
        zs.append(z.float().cpu())
        hs.append(h.float().cpu())
    return torch.cat(zs), torch.cat(hs)


def _existing_shards(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory.glob("*.npz"))


def _shard_name(part: int) -> str:
    return f"part-{part:05d}.npz"


def _check_resumable(
    path: pathlib.Path, index: range, corruption: str, severity: int
) -> None:
    """Fails if an existing shard is not the one this job would write."""
    with numpy.load(path, allow_pickle=False) as arrays:
        image_id = arrays["image_id"]
        corruptions = arrays["corruption"]
        severities = arrays["severity"]
    if (
        image_id.tolist() != list(index)
        or set(corruptions.tolist()) != {corruption}
        or set(severities.tolist()) != {severity}
    ):
        raise ValueError(
            f"{path} does not hold images {index.start}..{index.stop - 1} of"
            f" {corruption or 'clean'}/{severity}; it was written by a different"
            " job, so --resume cannot continue it (use --overwrite)"
        )


def embed_run(
    ckpt_path: str | os.PathLike[str],
    cfg: config.ExperimentConfig,
    data_dir: str | os.PathLike[str],
    embed_dir: str | os.PathLike[str],
    run_name: str,
    eval_sets: collections.abc.Sequence[str] = dumps.EVAL_SETS,
    max_images: int | None = None,
    overwrite: bool = False,
    shard_size: int = SHARD_SIZE,
    device: torch.device | None = None,
    resume: bool = False,
) -> dict[str, int]:
    """Writes the embedding dump of one checkpoint.

    The model, the view geometry and the input normalization come from the
    checkpoint's own config; `cfg` supplies `seed` and `eval.*`.

    Args:
        ckpt_path: Checkpoint written by `models.checkpoint.save`.
        cfg: Config of this embedding job.
        data_dir: Prepared data root (`data.layout`).
        embed_dir: Dump root (`eval.dumps`).
        run_name: Dump subdirectory name.
        eval_sets: Subset of `dumps.EVAL_SETS` to embed.
        max_images: Debug cap on images per image array (per corruption and
            severity for `stl10c`).
        overwrite: Replace existing shards; otherwise existing shards are an error.
        shard_size: Maximum images per shard.
        device: Defaults to CUDA when available, else CPU.
        resume: Keep existing shards and write only the missing ones (for a job
            that was interrupted; same checkpoint, config and arguments).

    Returns:
        Number of images in the dump per evaluation set (skipped shards
        included).

    Raises:
        FileExistsError: A requested set already has shards and neither
            `overwrite` nor `resume` is on.
        ValueError: Unknown evaluation set, both `overwrite` and `resume`, or
            (with `resume`) an existing shard that this job would not write.
    """
    unknown = [s for s in eval_sets if s not in dumps.EVAL_SETS]
    if unknown:
        raise ValueError(f"unknown eval sets {unknown}; expected {dumps.EVAL_SETS}")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    targets = {
        s: dumps.set_dir(embed_dir, run_name, s) for s in dict.fromkeys(eval_sets)
    }
    if not overwrite and not resume:
        for directory in targets.values():
            if _existing_shards(directory):
                raise FileExistsError(
                    f"{directory} already has shards; pass --overwrite to replace"
                )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = checkpoint.load(ckpt_path)
    backbone = ckpt.backbone.to(device).eval()
    projector = ckpt.projector.to(device).eval()
    data_cfg = ckpt.cfg.data

    counts: dict[str, int] = {}
    for name, directory in targets.items():
        directory.mkdir(parents=True, exist_ok=True)
        existing = set(_existing_shards(directory))
        if not resume:
            for stale in existing:
                stale.unlink()
            existing.clear()
        part = 0
        total = 0
        for source in _sources(name, data_dir):
            n = len(source.dataset)
            if max_images is not None:
                n = min(n, max_images)
            for start in range(0, n, shard_size):
                index = range(start, min(start + shard_size, n))
                path = directory / _shard_name(part)
                if path in existing:
                    _check_resumable(path, index, source.corruption, source.severity)
                    existing.discard(path)
                    part += 1
                    total += len(index)
                    continue
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    _shard_seed(
                        cfg.seed, name, source.corruption, source.severity, start
                    )
                )
                z, h = _embed_shard(
                    backbone,
                    projector,
                    source.dataset.images,
                    index,
                    data_cfg,
                    cfg.eval.num_views,
                    cfg.eval.batch_size,
                    generator,
                )
                labels = source.dataset.labels[index.start : index.stop]
                count = len(index)
                shards.save_shard(
                    path,
                    shards.EmbeddingShard(
                        z=z.numpy(),
                        h=h.numpy(),
                        image_id=numpy.arange(
                            index.start, index.stop, dtype=numpy.int64
                        ),
                        label=numpy.asarray(labels, dtype=numpy.int64),
                        corruption=numpy.array(
                            [source.corruption] * count, dtype=numpy.str_
                        ),
                        severity=numpy.full(count, source.severity, dtype=numpy.int8),
                    ),
                )
                part += 1
                total += count
        if existing:
            names = ", ".join(sorted(p.name for p in existing))
            raise ValueError(
                f"{directory} holds shards this job does not write ({names}); "
                "use --overwrite"
            )
        counts[name] = total
    return counts
