"""Variant A (LeJEPA): `(1 - lambda) * invariance + lambda * SIGReg`.

Balestriero & LeCun, arXiv:2511.08544. SIGReg is computed per view (N is the
per-view batch size) and averaged over views, as in the paper and the official
implementation; the views are not pooled into one sample.
"""

import torch

from lejepa_norm import config
from lejepa_norm.losses import baselines
from lejepa_norm.losses import batch as batch_lib
from lejepa_norm.losses import diagnostics
from lejepa_norm.losses import invariance
from lejepa_norm.losses import sigreg


class LeJEPALoss:
    """Variant A loss; conforms to `lejepa_norm.losses.batch.Loss`."""

    def __init__(
        self, cfg: config.LossConfig, generator: torch.Generator | None = None
    ) -> None:
        """Initializes the loss.

        Args:
            cfg: Loss config; `lambda_`, `num_slices` and `num_points` are used.
            generator: Optional RNG for the SIGReg directions.

        Raises:
            ValueError: `lambda_` is outside `[0, 1]`.
        """
        if not 0.0 <= cfg.lambda_ <= 1.0:
            raise ValueError(f"lambda must be in [0, 1], got {cfg.lambda_}")
        self.lambda_ = cfg.lambda_
        self.sigreg = sigreg.SIGReg(
            num_slices=cfg.num_slices,
            num_points=cfg.num_points,
            t_max=cfg.t_max,
            generator=generator,
        )

    def __call__(
        self, batch: batch_lib.ViewBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes the loss.

        Args:
            batch: Embeddings of all views of a batch.

        Returns:
            The fp32 scalar loss and detached diagnostics: `inv`, `dist` (SIGReg),
            `total` and `kappa` (on all views concatenated; `dist` is per view).
        """
        flat = batch.z.reshape(-1, batch.z.shape[-1])
        inv = invariance.invariance_loss(batch)
        dist = self.sigreg(batch.z)
        total = (1.0 - self.lambda_) * inv + self.lambda_ * dist
        stats = {
            "inv": inv.detach(),
            "dist": dist.detach(),
            "total": total.detach(),
            "kappa": diagnostics.kappa(flat),
        }
        return total, stats


def build_loss(cfg: config.LossConfig, embed_dim: int | None = None) -> batch_lib.Loss:
    """Builds the loss selected by `cfg.variant`.

    Args:
        cfg: Loss config.
        embed_dim: Embedding width (`model.embed_dim`); only the supervised
            baseline's head needs it.

    Returns:
        A callable conforming to `lejepa_norm.losses.batch.Loss`.

    Raises:
        NotImplementedError: Variants B and C (milestone M4).
    """
    if cfg.variant == "A":
        return LeJEPALoss(cfg)
    if cfg.variant in baselines.BASELINES:
        return baselines.build(cfg, embed_dim)
    raise NotImplementedError(f"loss variant {cfg.variant!r} is not implemented yet")
