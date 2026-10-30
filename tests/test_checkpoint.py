import pathlib

import pytest
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


def test_save_retries_a_transient_write_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config.ExperimentConfig(
        model=config.ModelConfig(
            reduced_stem=True, projector_hidden=(32,), embed_dim=16
        )
    )
    backbone, feat_dim = backbones.build_backbone(cfg.model)
    head = projector.build_projector(feat_dim, cfg.model)
    monkeypatch.setattr(checkpoint, "SAVE_RETRY_SECONDS", 0.0)
    real_save = torch.save
    calls = {"n": 0}

    def flaky(obj: object, f: pathlib.Path) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            f.write_bytes(b"partial")
            raise RuntimeError("unexpected pos")
        real_save(obj, f)

    monkeypatch.setattr(torch, "save", flaky)
    path = tmp_path / "last.pt"
    checkpoint.save(path, backbone, head, cfg, epoch=1)
    assert calls["n"] == 2
    assert checkpoint.load(path).epoch == 1

    def always_fails(obj: object, f: pathlib.Path) -> None:
        raise RuntimeError("disk")

    monkeypatch.setattr(torch, "save", always_fails)
    with pytest.raises(RuntimeError):
        checkpoint.save(tmp_path / "other.pt", backbone, head, cfg, epoch=2)
