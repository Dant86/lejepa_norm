# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Multi-view augmentation for joint-embedding pretraining.

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

COLOR_JITTER = (0.4, 0.4, 0.2, 0.1)
COLOR_JITTER_P = 0.8
GRAYSCALE_P = 0.2
BLUR_P = 0.5
BLUR_SIGMA = (0.1, 2.0)


def _view_transform(
    size: int, scale: tuple[float, float], blur: bool, cfg: config.DataConfig
) -> transforms.Compose:
    ops: list[torch.nn.Module] = [
        transforms.RandomResizedCrop(
            size,
            scale=scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(*COLOR_JITTER)], p=COLOR_JITTER_P
        ),
        transforms.RandomGrayscale(p=GRAYSCALE_P),
    ]
    if blur:
        # Odd kernel of about 10% of the image side, as in SimCLR.
        kernel = 2 * (size // 20) + 1
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel, sigma=BLUR_SIGMA)], p=BLUR_P
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
        self._global = _view_transform(
            cfg.global_size, cfg.global_scale, blur=True, cfg=cfg
        )
        self._local = _view_transform(
            cfg.local_size, cfg.local_scale, blur=False, cfg=cfg
        )

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
        self._global = _view_transform(
            cfg.global_size, cfg.global_scale, blur=True, cfg=cfg
        )

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
