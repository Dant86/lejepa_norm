"""Shared loss interface: every loss maps a `ViewBatch` to a scalar plus diagnostics.

Keeping this contract fixed lets the training loop stay variant-agnostic (A, B, C).
"""

import dataclasses
import typing

import torch


@dataclasses.dataclass(frozen=True)
class ViewBatch:
    """Embeddings of every view of a batch of images.

    Attributes:
        z: Projector outputs, shape `(num_views, batch_size, dim)`. View `v` of
            image `i` is `z[v, i]`.
        is_global: Bool tensor of shape `(num_views,)`; True for global views,
            False for local views.
    """

    z: torch.Tensor
    is_global: torch.Tensor

    def __post_init__(self) -> None:
        """Validates shapes and that a global view exists."""
        if self.z.ndim != 3:
            raise ValueError(f"z must be (views, batch, dim), got {self.z.shape}")
        if self.is_global.shape != (self.z.shape[0],):
            raise ValueError("is_global must have one entry per view")
        if self.is_global.dtype != torch.bool:
            raise ValueError("is_global must be a bool tensor")
        if not bool(self.is_global.any()):
            raise ValueError("at least one global view is required")

    @property
    def global_z(self) -> torch.Tensor:
        """Global-view embeddings, `(num_global, batch_size, dim)`."""
        return self.z[self.is_global]

    @property
    def local_z(self) -> torch.Tensor:
        """Local-view embeddings, `(num_local, batch_size, dim)`."""
        return self.z[~self.is_global]


class Loss(typing.Protocol):
    """A loss variant: returns a scalar loss and detached scalar diagnostics."""

    def __call__(
        self, batch: ViewBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes the loss for `batch`."""
        ...
