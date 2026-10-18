"""Projector MLP mapping backbone features `h` to embeddings `z`."""

from torch import nn

from lejepa_norm import config


def build_projector(feat_dim: int, cfg: config.ModelConfig) -> nn.Sequential:
    """Builds `feat_dim -> *projector_hidden -> embed_dim`.

    Hidden layers are Linear + BatchNorm1d + GELU; the output layer is a plain
    Linear (no normalization or activation), so `z` is unconstrained.

    Args:
        feat_dim: Backbone feature dimension.
        cfg: Model config; `projector_hidden` and `embed_dim` are used.

    Returns:
        The projector, mapping `(B, feat_dim)` to `(B, embed_dim)`.
    """
    layers: list[nn.Module] = []
    in_dim = feat_dim
    for hidden in cfg.projector_hidden:
        layers += [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU()]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, cfg.embed_dim))
    return nn.Sequential(*layers)
