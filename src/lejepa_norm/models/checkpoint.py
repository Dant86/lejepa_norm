"""Checkpoint format shared by training (writes) and embedding (reads).

A checkpoint is a single `torch.save` file holding only tensors and plain Python
types (loadable with `weights_only=True`):

    {"backbone": state_dict, "projector": state_dict,
     "config": config.to_dict(cfg), "epoch": int}

The config is stored so a checkpoint can be rebuilt without any other input.
Optimizer state is not part of this format.
"""

import dataclasses
import os
import pathlib
import time

import torch
from torch import nn

from lejepa_norm import config
from lejepa_norm.models import backbones
from lejepa_norm.models import projector as projector_lib


@dataclasses.dataclass
class Checkpoint:
    """A rebuilt model pair with the config and epoch it was saved at."""

    backbone: nn.Module
    projector: nn.Module
    cfg: config.ExperimentConfig
    epoch: int


SAVE_ATTEMPTS = 3
SAVE_RETRY_SECONDS = 5.0


def save(
    path: str | os.PathLike[str],
    backbone: nn.Module,
    projector: nn.Module,
    cfg: config.ExperimentConfig,
    epoch: int,
) -> None:
    """Writes a checkpoint atomically (temp file, then rename).

    Args:
        path: Destination file; parent directories are created.
        backbone: Trained backbone.
        projector: Trained projector.
        cfg: The resolved config of the run.
        epoch: Number of completed epochs.

    Raises:
        RuntimeError: the write failed on every one of `SAVE_ATTEMPTS` attempts.
    """
    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    payload = {
        "backbone": backbone.state_dict(),
        "projector": projector.state_dict(),
        "config": config.to_dict(cfg),
        "epoch": epoch,
    }
    for attempt in range(SAVE_ATTEMPTS):
        try:
            torch.save(payload, tmp)
            break
        except (RuntimeError, OSError):
            # Transient write errors on the shared filesystem (seen once in a run).
            tmp.unlink(missing_ok=True)
            if attempt == SAVE_ATTEMPTS - 1:
                raise
            time.sleep(SAVE_RETRY_SECONDS * (attempt + 1))
    tmp.replace(dest)


def load(path: str | os.PathLike[str]) -> Checkpoint:
    """Rebuilds the models described by a checkpoint on CPU and loads the weights.

    The returned modules are in training mode; call `.eval()` before embedding.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    cfg = config.from_dict(blob["config"])
    backbone, feat_dim = backbones.build_backbone(cfg.model)
    projector = projector_lib.build_projector(feat_dim, cfg.model)
    backbone.load_state_dict(blob["backbone"])
    projector.load_state_dict(blob["projector"])
    return Checkpoint(backbone, projector, cfg, int(blob["epoch"]))
