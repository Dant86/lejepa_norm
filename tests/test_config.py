import pathlib

import pytest

from lejepa_norm import config

BASE = pathlib.Path(__file__).parent.parent / "configs" / "base.yaml"


def test_base_config_matches_design_defaults() -> None:
    cfg = config.load_config(BASE)
    assert cfg.model.embed_dim == 256
    assert cfg.model.projector_hidden == (2048, 2048)
    assert cfg.loss.variant == "A"
    assert cfg.loss.lambda_ == 0.05
    assert cfg.train.lr == 5e-4


def test_overrides_round_trip_through_resolved_config(tmp_path: pathlib.Path) -> None:
    cfg = config.load_config(
        BASE,
        [
            "train.lr=1e-3",
            "loss.lambda=0.1",
            "loss.variant=C",
            "data.global_scale=[0.2, 1.0]",
            "seed=3",
        ],
    )
    assert cfg.train.lr == 1e-3
    assert cfg.loss.lambda_ == 0.1
    assert cfg.loss.variant == "C"
    assert cfg.data.global_scale == (0.2, 1.0)
    assert cfg.seed == 3

    resolved = config.save_resolved(cfg, tmp_path / "run")
    assert config.load_config(resolved) == cfg


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("train.lrr=1", KeyError),
        ("nonsense.x=1", KeyError),
        ("loss.variant=Z", ValueError),
        ("train.epochs=many", TypeError),
        ("train.lr", ValueError),
    ],
)
def test_bad_overrides_are_rejected(override: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        config.load_config(BASE, [override])
