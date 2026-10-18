"""Backbone registry and factory.

Backbones are built by name through `build_backbone(cfg)`, which returns the module
and its feature dimension. Every backbone is created from scratch (no pretrained
weights, no network access). Names without a dedicated builder fall back to any
`timm` architecture, so swapping backbones is a config change.
"""

import collections.abc
import typing

import timm
from torch import nn

from lejepa_norm import config

BackboneBuilder = collections.abc.Callable[[config.ModelConfig], tuple[nn.Module, int]]

_REGISTRY: dict[str, BackboneBuilder] = {}


def register(
    name: str,
) -> collections.abc.Callable[[BackboneBuilder], BackboneBuilder]:
    """Registers a backbone builder under `name` (decorator).

    Args:
        name: Value of `ModelConfig.backbone` that selects this builder.

    Returns:
        A decorator that registers and returns the builder unchanged.

    Raises:
        ValueError: `name` is already registered.
    """

    def decorator(builder: BackboneBuilder) -> BackboneBuilder:
        if name in _REGISTRY:
            raise ValueError(f"backbone {name!r} is already registered")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available() -> list[str]:
    """Names with a dedicated builder (any other `timm` name also works)."""
    return sorted(_REGISTRY)


def _create_timm(name: str) -> nn.Module:
    """Creates a headless `timm` model with random weights."""
    return timm.create_model(name, pretrained=False, num_classes=0)


def _feat_dim(model: nn.Module) -> int:
    return int(typing.cast(int, model.num_features))


def _reduce_resnet_stem(model: nn.Module) -> None:
    """Replaces the ImageNet stem (7x7/2 conv + maxpool) with a 3x3/1 conv.

    At 96px the standard stem downsamples 4x before the first stage, leaving a 3x3
    map after the last stage; the reduced stem keeps a 12x12 map.
    """
    conv1 = typing.cast(nn.Conv2d, model.conv1)
    model.conv1 = nn.Conv2d(
        conv1.in_channels,
        conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()


def _build_timm_resnet(cfg: config.ModelConfig) -> tuple[nn.Module, int]:
    model = _create_timm(cfg.backbone)
    if cfg.reduced_stem:
        _reduce_resnet_stem(model)
    return model, _feat_dim(model)


register("resnet18")(_build_timm_resnet)
register("resnet34")(_build_timm_resnet)
register("resnet50")(_build_timm_resnet)


def build_backbone(cfg: config.ModelConfig) -> tuple[nn.Module, int]:
    """Builds the backbone named by `cfg.backbone`.

    Args:
        cfg: Model config; `backbone` and `reduced_stem` are used.

    Returns:
        `(module, feat_dim)`: the module maps images `(B, 3, H, W)` to pooled
        features `(B, feat_dim)`.

    Raises:
        ValueError: The name is unknown, or `reduced_stem` is requested for a
            backbone without a reduced-stem option.
    """
    builder = _REGISTRY.get(cfg.backbone)
    if builder is not None:
        return builder(cfg)
    if not timm.is_model(cfg.backbone):
        raise ValueError(
            f"unknown backbone {cfg.backbone!r}; registered: {available()}, "
            "or any timm model name"
        )
    if cfg.reduced_stem:
        raise ValueError(f"reduced_stem is not supported for {cfg.backbone!r}")
    model = _create_timm(cfg.backbone)
    return model, _feat_dim(model)
