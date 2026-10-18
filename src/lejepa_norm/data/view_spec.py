"""Augmentation parameters shared by the PIL reference and the GPU pipeline.

`data.views` (torchvision on PIL images, the reference) and `data.gpu_views`
(batched tensors on the training device) both read their constants from here, so
the two definitions cannot drift apart. Per view: random resized crop (area
fraction `ViewSpec.scale`, aspect ratio `CROP_RATIO`), horizontal flip
(`FLIP_P`), color jitter `COLOR_JITTER` with probability `COLOR_JITTER_P`,
grayscale (`GRAYSCALE_P`) and, for global views only, a Gaussian blur
(`BLUR_P`, sigma uniform in `BLUR_SIGMA`, kernel `ViewSpec.blur_kernel_size`).
"""

import dataclasses

from lejepa_norm import config

CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)
CROP_ATTEMPTS = 10  # torchvision's RandomResizedCrop tries 10 times, then centers.
FLIP_P = 0.5
# (brightness, contrast, saturation, hue): factors are drawn uniformly from
# [1 - x, 1 + x] for the first three and from [-x, x] for hue.
COLOR_JITTER = (0.4, 0.4, 0.2, 0.1)
COLOR_JITTER_P = 0.8
GRAYSCALE_P = 0.2
BLUR_P = 0.5
BLUR_SIGMA = (0.1, 2.0)


@dataclasses.dataclass(frozen=True)
class ViewSpec:
    """Geometry of one kind of view.

    Attributes:
        size: Output side length in pixels.
        scale: Range of the crop area as a fraction of the image area.
        blur: Whether the view gets the random Gaussian blur.
    """

    size: int
    scale: tuple[float, float]
    blur: bool

    @property
    def blur_kernel_size(self) -> int:
        """Odd kernel of about 10% of the view side, as in SimCLR."""
        return 2 * (self.size // 20) + 1


def global_spec(cfg: config.DataConfig) -> ViewSpec:
    """Global views: `global_size` px, `global_scale` crops, with blur."""
    return ViewSpec(cfg.global_size, cfg.global_scale, blur=True)


def local_spec(cfg: config.DataConfig) -> ViewSpec:
    """Local views: `local_size` px, `local_scale` crops, no blur."""
    return ViewSpec(cfg.local_size, cfg.local_scale, blur=False)
