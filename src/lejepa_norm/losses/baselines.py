"""E2 baseline losses: SimCLR, VICReg and a supervised cross-entropy head.

All three consume the same `ViewBatch` as LeJEPA and see the *unnormalized*
projector output `z`, so the embedding radius is whatever the loss induces:

* `SimCLRLoss`: NT-Xent on the first two global views. `z` is L2-normalized
  *inside* the loss, which therefore never sees (nor constrains) `||z||`; the radius
  of the stored `z` is a built-in nuisance reference. Temperature
  `loss.temperature` (default 0.2). Diagnostics: `inv` = the positive-pair term
  `-mean(s_pos) / tau`, `dist` = the mean log-sum-exp over the 2N - 1 candidates, so
  `total = inv + dist` is the NT-Xent loss exactly. Local views are ignored (use
  `data.num_local_views=0`).
* `VICRegLoss` (Bardes et al., arXiv:2105.04906): `sim * MSE(z1, z2) +
  std * mean_view(mean_d relu(1 - std_d)) + cov * mean_view(sum_{i != j} C_ij^2 / d)`
  with coefficients 25 / 25 / 1 (`loss.sim_coeff`, `std_coeff`, `cov_coeff`), on the
  first two global views, no radial constraint. `inv` = the MSE, `dist` = the
  weighted variance + covariance terms.
* `SupervisedLoss`: cross-entropy of a linear head on `z` (the projector trained
  jointly with the head), averaged over all views, labels from `ViewBatch.labels`.
  The head is exposed as `module` and is optimized and checkpointed (`resume.pt`) by
  the training loop. `inv` = the cross-entropy, `dist` = 0, `acc` = head accuracy.
"""

import torch
from torch import nn
from torch.nn import functional

from lejepa_norm import config
from lejepa_norm.losses import batch as batch_lib

BASELINES = ("simclr", "vicreg", "supervised")


def _two_global_views(batch: batch_lib.ViewBatch) -> tuple[torch.Tensor, torch.Tensor]:
    views = batch.global_z.float()
    if views.shape[0] < 2:
        raise ValueError("this loss needs at least two global views")
    return views[0], views[1]


def nt_xent(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """NT-Xent terms for two views of `N` images.

    Args:
        z1: `(N, d)` embeddings of view 1 (normalized here).
        z2: `(N, d)` embeddings of view 2.
        temperature: Softmax temperature.

    Returns:
        `(positive, log_sum_exp)`: the mean `-s_pos / tau` and the mean log-sum-exp
        over the other 2N - 1 samples; their sum is the NT-Xent loss.
    """
    n = z1.shape[0]
    z = functional.normalize(torch.cat([z1, z2]).float(), dim=-1)
    logits = z @ z.T / temperature
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    partner = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    pos = logits.gather(1, partner[:, None]).squeeze(1)
    return -pos.mean(), torch.logsumexp(logits, dim=1).mean()


class SimCLRLoss:
    """SimCLR NT-Xent; conforms to `lejepa_norm.losses.batch.Loss`."""

    def __init__(self, cfg: config.LossConfig) -> None:
        """Stores the temperature (must be positive)."""
        if cfg.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {cfg.temperature}")
        self.temperature = cfg.temperature

    def __call__(
        self, batch: batch_lib.ViewBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns the NT-Xent loss of the first two global views and diagnostics."""
        z1, z2 = _two_global_views(batch)
        with torch.autocast(device_type=z1.device.type, enabled=False):
            pos, lse = nt_xent(z1, z2, self.temperature)
        total = pos + lse
        stats = {"inv": pos.detach(), "dist": lse.detach(), "total": total.detach()}
        return total, stats


def _variance_term(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return functional.relu(1.0 - std).mean()


def _covariance_term(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    centered = z - z.mean(dim=0)
    cov = centered.T @ centered / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.square().sum() / d


class VICRegLoss:
    """VICReg (invariance, variance, covariance); conforms to `batch.Loss`."""

    def __init__(self, cfg: config.LossConfig) -> None:
        """Stores the three coefficients."""
        self.sim = cfg.sim_coeff
        self.std = cfg.std_coeff
        self.cov = cfg.cov_coeff

    def __call__(
        self, batch: batch_lib.ViewBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns the VICReg loss of the first two global views and diagnostics."""
        z1, z2 = _two_global_views(batch)
        with torch.autocast(device_type=z1.device.type, enabled=False):
            inv = functional.mse_loss(z1, z2)
            var = (_variance_term(z1) + _variance_term(z2)) / 2
            cov = (_covariance_term(z1) + _covariance_term(z2)) / 2
            reg = self.std * var + self.cov * cov
            total = self.sim * inv + reg
        stats = {
            "inv": inv.detach(),
            "dist": reg.detach(),
            "total": total.detach(),
            "var": var.detach(),
            "cov": cov.detach(),
        }
        return total, stats


class SupervisedLoss:
    """Cross-entropy of a linear head on `z`; conforms to `batch.Loss`."""

    def __init__(self, cfg: config.LossConfig, embed_dim: int | None = None) -> None:
        """Creates the head.

        Args:
            cfg: `num_classes` is used.
            embed_dim: Input width of the head (`model.embed_dim`).
        """
        if embed_dim is None:
            raise ValueError("the supervised loss needs the embedding dimension")
        self.module = nn.Linear(embed_dim, cfg.num_classes)

    def __call__(
        self, batch: batch_lib.ViewBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns the mean cross-entropy over all views and the head accuracy."""
        if batch.labels is None:
            raise ValueError("the supervised loss needs ViewBatch.labels")
        with torch.autocast(device_type=batch.z.device.type, enabled=False):
            logits = self.module(batch.z.float())
            target = batch.labels.expand(logits.shape[0], -1)
            loss = functional.cross_entropy(logits.flatten(0, 1), target.flatten())
        acc = (logits.argmax(dim=-1) == target).float().mean()
        zero = torch.zeros((), device=loss.device)
        stats = {
            "inv": loss.detach(),
            "dist": zero,
            "total": loss.detach(),
            "acc": acc.detach(),
        }
        return loss, stats


def build(cfg: config.LossConfig, embed_dim: int | None = None) -> batch_lib.Loss:
    """Builds the baseline selected by `cfg.variant`.

    Args:
        cfg: Loss config with `variant` in `BASELINES`.
        embed_dim: Embedding width (needed by `supervised`).

    Returns:
        The loss.

    Raises:
        ValueError: `variant` is not a baseline.
    """
    if cfg.variant == "simclr":
        return SimCLRLoss(cfg)
    if cfg.variant == "vicreg":
        return VICRegLoss(cfg)
    if cfg.variant == "supervised":
        return SupervisedLoss(cfg, embed_dim)
    raise ValueError(f"{cfg.variant!r} is not a baseline loss")
