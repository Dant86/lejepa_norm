"""Test-only tiny CNN backbone, registered as `tiny_cnn`.

Importing this module registers the backbone with the production registry
(`lejepa_norm.models.backbones.register`), so tests (and the subprocess drivers of
the resume tests) can select it with `model.backbone=tiny_cnn`. Everything else in
the pipeline stays real; only the model is small enough to train in milliseconds.
"""

from torch import nn

from lejepa_norm import config
from lejepa_norm.models import backbones

NAME = "tiny_cnn"
FEAT_DIM = 32


@backbones.register(NAME)
def _build(cfg: config.ModelConfig) -> tuple[nn.Module, int]:
    if cfg.reduced_stem:
        raise ValueError("tiny_cnn has no reduced-stem option")
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, FEAT_DIM, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(FEAT_DIM),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return model, FEAT_DIM
