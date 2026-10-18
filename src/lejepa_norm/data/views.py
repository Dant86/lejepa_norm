# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Multi-view augmentation on PIL images: the REFERENCE implementation.

Training and embedding use the batched GPU pipeline in `data.gpu_views`; this
torchvision/PIL version is kept as the reference it is tested against (see
`research_notes/gpu_augmentation/`). Both read their parameters from
`data.view_spec`.

Every view is a random resized crop + horizontal flip + color jitter
(0.4, 0.4, 0.2, 0.1; p=0.8) + random grayscale (p=0.2). Global views are 96px with
crop scale 0.3-1.0 and additionally get a Gaussian blur; local views are 48px with
crop scale 0.05-0.3 and no blur. All sizes and counts come from `DataConfig`.
Every view ends with per-channel normalization using `DataConfig.mean/std`; the
un-augmented `CenterTransform` applies the same normalization, so training and
embedding see identically scaled inputs.
"""

import typing

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_functional

from lejepa_norm import config
from lejepa_norm.data import view_spec


def _view_transform(
    spec: view_spec.ViewSpec, cfg: config.DataConfig
) -> transforms.Compose:
    ops: list[torch.nn.Module] = [
        transforms.RandomResizedCrop(
            spec.size,
            scale=spec.scale,
            ratio=view_spec.CROP_RATIO,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=view_spec.FLIP_P),
        transforms.RandomApply(
            [transforms.ColorJitter(*view_spec.COLOR_JITTER)],
            p=view_spec.COLOR_JITTER_P,
        ),
        transforms.RandomGrayscale(p=view_spec.GRAYSCALE_P),
    ]
    if spec.blur:
        ops.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        spec.blur_kernel_size, sigma=view_spec.BLUR_SIGMA
                    )
                ],
                p=view_spec.BLUR_P,
            )
        )
    return transforms.Compose(
        [*ops, transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
    )


class MultiViewTransform:
    """Turns one PIL image into `num_global_views + num_local_views` tensors.

    Calling it returns a list: the global views first (each `(3, G, G)`), then the
    local views (each `(3, L, L)`).
    """

    def __init__(self, cfg: config.DataConfig) -> None:
        """Builds the global and local pipelines from `cfg`."""
        self.num_global_views = cfg.num_global_views
        self.num_local_views = cfg.num_local_views
        self._global = _view_transform(view_spec.global_spec(cfg), cfg)
        self._local = _view_transform(view_spec.local_spec(cfg), cfg)

    def __call__(self, image: object) -> list[torch.Tensor]:
        """Generates all views of `image` (a PIL image)."""
        out: list[torch.Tensor] = []
        for _ in range(self.num_global_views):
            out.append(typing.cast(torch.Tensor, self._global(image)))
        for _ in range(self.num_local_views):
            out.append(typing.cast(torch.Tensor, self._local(image)))
        return out


class AugmentedViewTransform:
    """One random global-view augmentation: PIL image to `(3, G, G)` tensor.

    The same pipeline `MultiViewTransform` uses for its global views (crop, flip,
    color jitter, grayscale, blur, normalization). Each call draws fresh
    randomness from torch's global RNG.
    """

    def __init__(self, cfg: config.DataConfig) -> None:
        """Builds the global-view pipeline from `cfg`."""
        self._global = _view_transform(view_spec.global_spec(cfg), cfg)

    def __call__(self, image: object) -> torch.Tensor:
        """Returns one augmented global view of `image` (a PIL image)."""
        return typing.cast(torch.Tensor, self._global(image))


class CenterTransform:
    """Un-augmented view: PIL image to a normalized float tensor `(3, H, W)`."""

    def __init__(self, cfg: config.DataConfig) -> None:
        """Stores the normalization statistics from `cfg`."""
        self._mean = list(cfg.mean)
        self._std = list(cfg.std)

    def __call__(self, image: object) -> torch.Tensor:
        """Converts `image` (a PIL image) without any augmentation."""
        tensor = transforms_functional.to_tensor(image)  # pyright: ignore[reportArgumentType]
        return transforms_functional.normalize(tensor, self._mean, self._std)
