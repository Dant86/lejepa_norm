"""Detached scalar diagnostics shared by the loss variants.

`kappa` pooled over all views mixes views that may sit at different radii (global
and local crops), while SIGReg acts per view. Apply `kappa` to the samples of one
view type (e.g. `ViewBatch.global_z` flattened to `(views * batch, d)`) to measure
the radial spread the loss actually controls.
"""

import torch


@torch.no_grad()
def kappa(z: torch.Tensor) -> torch.Tensor:
    """Radius dispersion `Var(||z||^2) / (2 d)`.

    Equals 1 in expectation for `z ~ N(0, I_d)` (where `||z||^2 ~ chi^2_d`) and 0
    when every sample has the same norm.

    Args:
        z: Samples, shape `(N, d)`.

    Returns:
        Detached fp32 scalar tensor.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be (N, d), got {tuple(z.shape)}")
    sq_norm = z.float().square().sum(dim=-1)
    return sq_norm.var() / (2 * z.shape[-1])
