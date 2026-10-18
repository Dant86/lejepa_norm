import pathlib

import torch

from lejepa_norm import config
from lejepa_norm.models import backbones
from lejepa_norm.models import checkpoint
from lejepa_norm.models import projector


def test_checkpoint_round_trip_rebuilds_identical_models(
    tmp_path: pathlib.Path,
) -> None:
    cfg = config.ExperimentConfig(
        model=config.ModelConfig(
            reduced_stem=True, projector_hidden=(32,), embed_dim=16
        ),
        loss=config.LossConfig(lambda_=0.1),
    )
    backbone, feat_dim = backbones.build_backbone(cfg.model)
    head = projector.build_projector(feat_dim, cfg.model)
    path = tmp_path / "ckpt" / "epoch_0003.pt"
    checkpoint.save(path, backbone, head, cfg, epoch=3)
    assert not path.with_name(path.name + ".tmp").exists()

    loaded = checkpoint.load(path)
    assert loaded.epoch == 3
    assert loaded.cfg == cfg
    x = torch.randn(2, 3, 32, 32)
    for module in (backbone, head, loaded.backbone, loaded.projector):
        module.eval()
    with torch.no_grad():
        expected = head(backbone(x))
        actual = loaded.projector(loaded.backbone(x))
    assert torch.allclose(expected, actual)
