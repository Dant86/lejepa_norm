"""Untrained (random-init) baseline checkpoints in the standard checkpoint format.

The E2 baseline asks what an *untrained* network of the LeJEPA architecture does to
the radius/direction structure of its embedding. This module builds the same
backbone and projector as a LeJEPA run (`configs/base.yaml`, same builders, same
initialization seed convention: `torch.manual_seed(seed)` and then backbone,
then projector) and writes them through `models.checkpoint.save`, so
`apps/embed.py` and every analysis treat them like any other run.

BatchNorm needs an explicit choice, because an untrained network has no meaningful
running statistics (they are at their `mean 0, var 1` initial values, i.e. the
eval-mode BatchNorm is the identity). Two modes (`BN_MODES`):

* `"calibrated"` (default): the running statistics are *calibrated* by one forward
  pass in train mode over augmented global views of unlabeled training images, with
  no gradient step and no weight change (`momentum=None`: cumulative average over
  all calibration batches, so the result does not depend on the batch order).
  Evaluated in eval mode this is the large-batch version of "train-mode BatchNorm"
  (batch statistics of the data distribution), and it is the same mode the trained
  checkpoints are embedded in (running statistics).
* `"identity"`: no calibration; eval-mode BatchNorm is the identity map (raw
  activations of the random network flow through).

Train-mode batch statistics at embedding time are deliberately not offered: the
embedding of an image would depend on the other images in its batch.
"""

import os
import typing
import zlib

import numpy
import torch
from torch import nn

from lejepa_norm import config
from lejepa_norm.data import batches
from lejepa_norm.data import datasets
from lejepa_norm.data import gpu_views
from lejepa_norm.data import view_spec
from lejepa_norm.models import backbones
from lejepa_norm.models import checkpoint
from lejepa_norm.models import projector as projector_lib

BN_CALIBRATED = "calibrated"
BN_IDENTITY = "identity"
BN_MODES = (BN_CALIBRATED, BN_IDENTITY)

DEFAULT_CALIBRATION_IMAGES = 16384
CALIBRATION_BATCH_SIZE = 256


def build_untrained(
    cfg: config.ExperimentConfig, seed: int
) -> tuple[nn.Module, nn.Module]:
    """Builds the backbone and projector of `cfg` with random weights.

    Args:
        cfg: Experiment config (`model` section is used).
        seed: Initialization seed (`torch.manual_seed` before building).

    Returns:
        `(backbone, projector)` in training mode.
    """
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    backbone, feat_dim = backbones.build_backbone(cfg.model)
    projector = projector_lib.build_projector(feat_dim, cfg.model)
    return backbone, projector


def _batch_norms(*modules: nn.Module) -> list[nn.BatchNorm1d | nn.BatchNorm2d]:
    return [
        m
        for module in modules
        for m in module.modules()
        if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d)
    ]


@torch.no_grad()
def calibrate_batch_norm(
    backbone: nn.Module,
    projector: nn.Module,
    images: batches.ArrayConcat | numpy.ndarray,
    data_cfg: config.DataConfig,
    seed: int,
    num_images: int = DEFAULT_CALIBRATION_IMAGES,
    batch_size: int = CALIBRATION_BATCH_SIZE,
    device: torch.device | None = None,
) -> int:
    """Sets every BatchNorm running statistic by a gradient-free forward pass.

    Statistics are reset and then averaged cumulatively (`momentum=None`) over
    `num_images` random training images, one augmented global view each, with the
    modules in train mode (batch statistics inside the pass). Weights, including
    the BatchNorm affine parameters, do not change. The modules are left in train
    mode with their original momenta restored.

    Args:
        backbone: Backbone (modified in place).
        projector: Projector (modified in place).
        images: uint8 `(N, H, W, 3)` index space of the calibration images.
        data_cfg: Data config (view geometry, normalization).
        seed: Seed of the image choice and of the augmentations.
        num_images: Number of calibration images (rounded down to whole batches).
        batch_size: Images per forward batch.
        device: Defaults to CUDA when available, else CPU.

    Returns:
        Number of calibration batches.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norms = _batch_norms(backbone, projector)
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
    backbone.to(device).train()
    projector.to(device).train()
    rng = numpy.random.default_rng(zlib.crc32(f"{seed}/bn-calibration".encode()))
    count = min(num_images, len(images))
    chosen = numpy.sort(rng.choice(len(images), size=count, replace=False))
    generator = torch.Generator(device=device)
    generator.manual_seed(zlib.crc32(f"{seed}/bn-calibration/augment".encode()))
    spec = view_spec.global_spec(data_cfg)
    num_batches = max(count // batch_size, 1)
    for b in range(num_batches):
        index = chosen[b * batch_size : (b + 1) * batch_size]
        if isinstance(images, batches.ArrayConcat):
            raw = images.gather(index)
        else:
            raw = images[index]
        batch = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
            numpy.ascontiguousarray(raw)
        ).to(device)
        view = gpu_views.augment(batch, spec, data_cfg, generator)
        projector(backbone(view))
    for m, momentum in zip(norms, momenta, strict=True):
        m.momentum = momentum
    return num_batches


def write_checkpoint(
    path: str | os.PathLike[str],
    cfg: config.ExperimentConfig,
    seed: int,
    bn_mode: typing.Literal["calibrated", "identity"],
    data_dir: str | os.PathLike[str] | None = None,
    num_calibration_images: int = DEFAULT_CALIBRATION_IMAGES,
    device: torch.device | None = None,
) -> None:
    """Writes an untrained checkpoint (epoch 0) through the checkpoint contract.

    Args:
        path: Destination file.
        cfg: Experiment config of the LeJEPA runs this baseline mirrors.
        seed: Initialization (and calibration) seed.
        bn_mode: `"calibrated"` or `"identity"` (see the module docstring).
        data_dir: Prepared data root; required for `"calibrated"` (STL-10
            unlabeled and train images are the calibration pool).
        num_calibration_images: Calibration images for `"calibrated"`.
        device: Device of the calibration pass.

    Raises:
        ValueError: Unknown `bn_mode`, or `"calibrated"` without `data_dir`.
    """
    if bn_mode not in BN_MODES:
        raise ValueError(f"bn_mode must be one of {BN_MODES}, got {bn_mode!r}")
    backbone, projector = build_untrained(cfg, seed)
    if bn_mode == BN_CALIBRATED:
        if data_dir is None:
            raise ValueError("bn_mode='calibrated' needs data_dir")
        calibrate_batch_norm(
            backbone,
            projector,
            datasets.load_pretraining_images(data_dir),
            cfg.data,
            seed,
            num_calibration_images,
            device=device,
        )
    backbone.cpu()
    projector.cpu()
    checkpoint.save(path, backbone, projector, cfg, epoch=0)
