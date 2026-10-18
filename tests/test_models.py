import typing

import pytest
import timm
import torch

from lejepa_norm import config
from lejepa_norm.models import backbones
from lejepa_norm.models import projector


@pytest.mark.parametrize(("reduced_stem", "spatial"), [(False, 3), (True, 12)])
def test_resnet18_backbone_and_projector(reduced_stem: bool, spatial: int) -> None:
    cfg = config.ModelConfig(backbone="resnet18", reduced_stem=reduced_stem)
    backbone, feat_dim = backbones.build_backbone(cfg)
    head = projector.build_projector(feat_dim, cfg)
    assert feat_dim == 512

    images = torch.randn(4, 3, 96, 96)
    h = backbone(images)
    z = head(h)
    assert h.shape == (4, feat_dim)
    assert z.shape == (4, 256)
    # The stem choice sets the resolution of the last feature map.
    features = typing.cast(timm.models.ResNet, backbone).forward_features(images)
    assert features.shape[-2:] == (spatial, spatial)


def test_projector_layout_follows_config() -> None:
    cfg = config.ModelConfig(projector_hidden=(64, 32), embed_dim=16)
    head = projector.build_projector(10, cfg)
    linears = [m for m in head if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (10, 64),
        (64, 32),
        (32, 16),
    ]
    assert sum(isinstance(m, torch.nn.BatchNorm1d) for m in head) == 2
    assert isinstance(head[-1], torch.nn.Linear)  # Linear output, no activation.


def test_build_backbone_falls_back_to_timm_and_rejects_unknown_names() -> None:
    _, feat_dim = backbones.build_backbone(config.ModelConfig(backbone="resnet10t"))
    assert feat_dim == 512
    with pytest.raises(ValueError):
        backbones.build_backbone(config.ModelConfig(backbone="not_a_model"))
