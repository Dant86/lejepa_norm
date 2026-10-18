"""LeJEPA invariance (prediction) loss.

Every view's embedding is pulled towards the mean embedding of the global views of
the same image (Balestriero & LeCun, arXiv:2511.08544). The center is not detached.
"""

import torch

from lejepa_norm.losses import batch


def invariance_loss(views: batch.ViewBatch) -> torch.Tensor:
    """Mean squared error between each view's `z` and the global-view mean.

    Averaged over views (global and local), images and embedding coordinates.
    Computed in fp32 regardless of the input dtype or an enclosing autocast region.

    Args:
        views: Embeddings of all views of a batch.

    Returns:
        Scalar tensor.
    """
    z = views.z
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        center = z[views.is_global].mean(dim=0, keepdim=True)
        return (z - center).square().mean()
