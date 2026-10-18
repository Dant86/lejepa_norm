# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false
"""Batched, vectorized view augmentation on the training device.

The same augmentation as the PIL reference in `data.views` (parameters shared via
`data.view_spec`), applied to a whole uint8 batch at once with independent
random parameters per sample and no per-sample Python loop:

1. Random resized crop: area fraction and log-uniform aspect ratio, up to
   `CROP_ATTEMPTS` tries per sample, center-crop fallback (torchvision's
   `RandomResizedCrop.get_params`, vectorized). Crop + resize is Pillow's
   bicubic resampling (a = -0.5, support widened when downscaling, i.e.
   antialiased) written as two per-sample interpolation matrices, so the result
   matches PIL up to Pillow's 8-bit intermediate rounding (`torchvision.ops.
   roi_align` was not used: it is bilinear, see the pipeline comparison note).
2. Horizontal flip.
3. Color jitter, gated per sample with `COLOR_JITTER_P`: brightness, contrast,
   saturation and hue in a random order per sample (four stages; at each stage
   every sample applies the op its permutation names, selected with
   `torch.where`). The ops follow `torchvision.transforms.v2.functional` on float
   tensors. Hue uses torchvision's batched RGB<->HSV conversion
   (`torchvision.transforms.v2.functional._color._rgb_to_hsv` / `_hsv_to_rgb`).
   These are PRIVATE helpers; the torchvision version is pinned in `uv.lock` and
   the op-level parity tests against the public `adjust_hue` guard against drift.
4. Grayscale (ITU-R 601 weights, as torchvision's tensor path).
5. Gaussian blur (global views only): per-sample sigma, separable depthwise
   convolution with reflect padding (cuDNN's default TF32 for fp32 convolutions
   is switched off around it, so CUDA results are full fp32).
6. Normalization with `DataConfig.mean/std`.

All math is fp32 and the pixels stay continuous (no uint8 rounding between
steps, unlike PIL). All randomness comes from the `torch.Generator` the caller
passes (on the device of the images), drawn in a fixed order, so the output
depends only on the generator state, never on DataLoader workers. The same seed
gives different streams on CPU and CUDA generators. Works on any device.
"""

import contextlib
import dataclasses
import math
import typing

import torch
import torch.nn.functional as nn_functional
import torchvision.transforms.v2.functional as tv_functional
import torchvision.transforms.v2.functional._color as tv_color

from lejepa_norm import config
from lejepa_norm.data import view_spec

# Indices of the color-jitter ops in `ViewParams.order`.
BRIGHTNESS, CONTRAST, SATURATION, HUE = range(4)


@dataclasses.dataclass(frozen=True)
class ViewParams:
    """Random parameters of one view for a batch; every field has length B.

    Attributes:
        top: Crop top row (int64).
        left: Crop left column (int64).
        height: Crop height (int64).
        width: Crop width (int64).
        flip: Horizontal flip (bool).
        jitter: Color jitter is applied (bool).
        order: `(B, 4)` permutation of the jitter ops (`BRIGHTNESS`, ...);
            `order[b, k]` is applied at stage k.
        brightness: Brightness factor.
        contrast: Contrast factor.
        saturation: Saturation factor.
        hue: Hue shift (fraction of a turn).
        grayscale: Converted to grayscale (bool).
        blur: Blurred (bool; always False for views without blur).
        sigma: Blur sigma.
    """

    top: torch.Tensor
    left: torch.Tensor
    height: torch.Tensor
    width: torch.Tensor
    flip: torch.Tensor
    jitter: torch.Tensor
    order: torch.Tensor
    brightness: torch.Tensor
    contrast: torch.Tensor
    saturation: torch.Tensor
    hue: torch.Tensor
    grayscale: torch.Tensor
    blur: torch.Tensor
    sigma: torch.Tensor

    def __len__(self) -> int:
        """Batch size."""
        return int(self.top.shape[0])

    def select(self, index: int | slice | torch.Tensor) -> "ViewParams":
        """The parameters of a subset of the samples."""
        return ViewParams(
            **{f.name: getattr(self, f.name)[index] for f in dataclasses.fields(self)}
        )


def _uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    generator: torch.Generator,
) -> torch.Tensor:
    u = torch.rand(shape, generator=generator, device=generator.device)
    return low + (high - low) * u


def sample_crops(
    batch: int,
    height: int,
    width: int,
    scale: tuple[float, float],
    generator: torch.Generator,
    ratio: tuple[float, float] = view_spec.CROP_RATIO,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples `RandomResizedCrop` boxes for a batch, vectorized.

    Each sample makes `CROP_ATTEMPTS` draws of (area fraction, log aspect ratio)
    and uses the first one whose rounded box fits; samples without a fitting draw
    get torchvision's center-crop fallback.

    Returns:
        `(top, left, crop_height, crop_width)`, int64 tensors of length `batch`.
    """
    attempts = view_spec.CROP_ATTEMPTS
    area = _uniform((batch, attempts), *scale, generator) * (height * width)
    log_ratio = _uniform(
        (batch, attempts), math.log(ratio[0]), math.log(ratio[1]), generator
    )
    u_top = torch.rand(batch, generator=generator, device=generator.device)
    u_left = torch.rand(batch, generator=generator, device=generator.device)
    aspect = torch.exp(log_ratio)
    w = torch.round(torch.sqrt(area * aspect)).long()
    h = torch.round(torch.sqrt(area / aspect)).long()
    valid = (w > 0) & (w <= width) & (h > 0) & (h <= height)
    first = valid.int().argmax(dim=1, keepdim=True)  # first fitting attempt
    found = valid.any(dim=1)
    w = w.gather(1, first)[:, 0]
    h = h.gather(1, first)[:, 0]
    # Center-crop fallback (torchvision's rule for the image aspect ratio).
    in_ratio = width / height
    if in_ratio < min(ratio):
        fallback_w, fallback_h = width, int(round(width / min(ratio)))
    elif in_ratio > max(ratio):
        fallback_w, fallback_h = int(round(height * max(ratio))), height
    else:
        fallback_w, fallback_h = width, height
    w = torch.where(found, w, fallback_w)
    h = torch.where(found, h, fallback_h)
    # Uniform integer offsets in [0, size - crop], like torch.randint.
    top = torch.floor(u_top * (height - h + 1)).long().clamp(max=height - h)
    left = torch.floor(u_left * (width - w + 1)).long().clamp(max=width - w)
    top = torch.where(found, top, (height - fallback_h) // 2)
    left = torch.where(found, left, (width - fallback_w) // 2)
    return top, left, h, w


def sample_params(
    spec: view_spec.ViewSpec,
    batch: int,
    height: int,
    width: int,
    generator: torch.Generator,
) -> ViewParams:
    """Draws the random parameters of one view for `batch` images.

    The draws happen in a fixed order (every field is drawn even when unused), so
    the generator advances by the same amount for every view of the same size.

    Args:
        spec: View geometry (`view_spec.global_spec` / `local_spec`).
        batch: Number of images.
        height: Input image height.
        width: Input image width.
        generator: Source of all randomness; tensors live on its device.

    Returns:
        The parameters, on the generator's device.
    """
    top, left, crop_h, crop_w = sample_crops(
        batch, height, width, spec.scale, generator
    )

    def coin(p: float) -> torch.Tensor:
        return torch.rand(batch, generator=generator, device=generator.device) < p

    flip = coin(view_spec.FLIP_P)
    jitter = coin(view_spec.COLOR_JITTER_P)
    order = torch.rand(
        (batch, 4), generator=generator, device=generator.device
    ).argsort(dim=1)
    b, c, s, h = view_spec.COLOR_JITTER
    brightness = _uniform((batch,), 1 - b, 1 + b, generator)
    contrast = _uniform((batch,), 1 - c, 1 + c, generator)
    saturation = _uniform((batch,), 1 - s, 1 + s, generator)
    hue = _uniform((batch,), -h, h, generator)
    grayscale = coin(view_spec.GRAYSCALE_P)
    blur = coin(view_spec.BLUR_P) & spec.blur
    sigma = _uniform((batch,), *view_spec.BLUR_SIGMA, generator)
    return ViewParams(
        top=top,
        left=left,
        height=crop_h,
        width=crop_w,
        flip=flip,
        jitter=jitter,
        order=order,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        grayscale=grayscale,
        blur=blur,
        sigma=sigma,
    )


def to_float(images: torch.Tensor) -> torch.Tensor:
    """uint8 `(B, H, W, 3)` or `(B, 3, H, W)` to float32 `(B, 3, H, W)` in [0, 1].

    A tensor whose dimension 1 has size 3 is read as channels-first, otherwise a
    last dimension of size 3 means channels-last.
    """
    if images.dtype != torch.uint8 or images.ndim != 4:
        raise ValueError(
            f"expected a uint8 (B, H, W, 3) or (B, 3, H, W) batch, got "
            f"{images.dtype} {tuple(images.shape)}"
        )
    if images.shape[1] != 3:
        if images.shape[-1] != 3:
            raise ValueError(f"no RGB channel axis in {tuple(images.shape)}")
        images = images.permute(0, 3, 1, 2)
    return images.float().div(255.0).contiguous()


def _bicubic(x: torch.Tensor) -> torch.Tensor:
    """Pillow's bicubic kernel (a = -0.5)."""
    a = -0.5
    x = x.abs()
    near = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    far = (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return torch.where(x < 1.0, near, torch.where(x < 2.0, far, 0.0))


def _resample_matrix(
    start: torch.Tensor, length: torch.Tensor, in_size: int, out_size: int
) -> torch.Tensor:
    """Per-sample `(B, out, in)` weights of Pillow's bicubic resampling.

    Pillow crops first, then resamples the crop `[start, start + length)` to
    `out_size` pixels: output pixel `o` is centered at `(o + 0.5) * scale` in crop
    coordinates (`scale = length / out_size`), the kernel is stretched by
    `max(scale, 1)` (antialiasing when downscaling), only crop pixels inside the
    kernel support contribute, and the weights are normalized to sum to 1.
    """
    device = start.device
    scale = (length.float() / out_size)[:, None, None]  # (B, 1, 1)
    filter_scale = scale.clamp(min=1.0)
    support = 2.0 * filter_scale
    center = (
        torch.arange(out_size, device=device).float()[None, :, None] + 0.5
    ) * scale
    x = (
        torch.arange(in_size, device=device).float()[None, None, :]
        - start.float()[:, None, None]
    )  # input pixel index in crop coordinates
    lo = torch.floor(center - support + 0.5).clamp(min=0.0)
    hi = torch.minimum(
        torch.floor(center + support + 0.5), length.float()[:, None, None]
    )
    weights = _bicubic((x - center + 0.5) / filter_scale)
    weights = torch.where((x >= lo) & (x < hi), weights, 0.0)
    total = weights.sum(dim=2, keepdim=True)
    return weights / torch.where(total == 0, 1.0, total)


def resized_crop(
    images: torch.Tensor,
    top: torch.Tensor,
    left: torch.Tensor,
    height: torch.Tensor,
    width: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Crops each image to its own box and resizes it to `size` x `size`.

    Pillow's bicubic algorithm in float (see `_resample_matrix`), without
    clamping: the bicubic overshoot is kept (Pillow saturates at 0 and 255).

    Args:
        images: Float `(B, C, H, W)`.
        top: Per-sample crop top row.
        left: Per-sample crop left column.
        height: Per-sample crop height.
        width: Per-sample crop width.
        size: Output side length.

    Returns:
        `(B, C, size, size)`.
    """
    rows = _resample_matrix(top, height, images.shape[-2], size)  # (B, S, H)
    cols = _resample_matrix(left, width, images.shape[-1], size)  # (B, S, W)
    horizontal = torch.matmul(images, cols.transpose(1, 2)[:, None])  # (B,C,H,S)
    return torch.matmul(rows[:, None], horizontal)


def _factor(values: torch.Tensor) -> torch.Tensor:
    return values.float().reshape(-1, 1, 1, 1)


def _blend(
    image: torch.Tensor, other: torch.Tensor, ratio: torch.Tensor
) -> torch.Tensor:
    """`ratio * image + (1 - ratio) * other`, clamped to [0, 1] (as torchvision)."""
    r = _factor(ratio)
    return (image * r + other * (1.0 - r)).clamp(0.0, 1.0)


def grayscale(images: torch.Tensor) -> torch.Tensor:
    """Luma `(B, 1, H, W)` with torchvision's weights (0.2989, 0.587, 0.114)."""
    return tv_functional.rgb_to_grayscale(images, num_output_channels=1)


def adjust_brightness(images: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Per-sample `adjust_brightness` on float `(B, 3, H, W)`."""
    return (images * _factor(factor)).clamp(0.0, 1.0)


def adjust_contrast(images: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Per-sample `adjust_contrast`: blend with the image's mean luma."""
    mean = grayscale(images).mean(dim=(-3, -2, -1), keepdim=True)
    return _blend(images, mean, factor)


def adjust_saturation(images: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Per-sample `adjust_saturation`: blend with the per-pixel luma."""
    return _blend(images, grayscale(images), factor)


def adjust_hue(images: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Per-sample `adjust_hue`: rotate the HSV hue by `shift` turns."""
    hsv = tv_color._rgb_to_hsv(images)
    h, s, v = hsv.unbind(dim=-3)
    h = (h + shift.float().reshape(-1, 1, 1)).remainder(1.0)
    return tv_color._hsv_to_rgb(torch.stack((h, s, v), dim=-3))


def color_jitter(images: torch.Tensor, params: ViewParams) -> torch.Tensor:
    """Applies the four jitter ops in each sample's own order (where gated on).

    Every stage computes all four ops for the whole batch and keeps, per sample,
    the one its permutation names at that stage (no data-dependent indexing, so
    no host synchronization).
    """
    ops: list[tuple[int, typing.Callable[[torch.Tensor], torch.Tensor]]] = [
        (BRIGHTNESS, lambda x: adjust_brightness(x, params.brightness)),
        (CONTRAST, lambda x: adjust_contrast(x, params.contrast)),
        (SATURATION, lambda x: adjust_saturation(x, params.saturation)),
        (HUE, lambda x: adjust_hue(x, params.hue)),
    ]
    out = images
    for stage in range(4):
        current = out
        for op, fn in ops:
            use = (params.jitter & (params.order[:, stage] == op)).reshape(-1, 1, 1, 1)
            out = torch.where(use, fn(current), out)
    return out


def _gaussian_kernels(sigma: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """`(B, k)` normalized 1D Gaussian kernels, as torchvision's `gaussian_blur`."""
    half = (kernel_size - 1) / 2
    x = torch.linspace(-half, half, kernel_size, device=sigma.device)
    pdf = torch.exp(-0.5 * (x[None, :] / sigma.float()[:, None]).square())
    return pdf / pdf.sum(dim=1, keepdim=True)


def gaussian_blur(
    images: torch.Tensor, sigma: torch.Tensor, kernel_size: int
) -> torch.Tensor:
    """Per-sample Gaussian blur: separable depthwise conv, reflect padding.

    Args:
        images: Float `(B, C, H, W)`.
        sigma: Per-sample standard deviation `(B,)`.
        kernel_size: Odd kernel side length.

    Returns:
        The blurred images, same shape.
    """
    b, c, height, width = images.shape
    kernels = _gaussian_kernels(sigma, kernel_size).repeat_interleave(c, dim=0)
    pad = kernel_size // 2
    flat = images.reshape(1, b * c, height, width)
    flat = nn_functional.pad(flat, (pad, pad, pad, pad), mode="reflect")
    with _ieee_fp32_conv(images.device):
        flat = nn_functional.conv2d(flat, kernels[:, None, None, :], groups=b * c)
        flat = nn_functional.conv2d(flat, kernels[:, None, :, None], groups=b * c)
    return flat.reshape(b, c, height, width)


@contextlib.contextmanager
def _ieee_fp32_conv(device: torch.device) -> typing.Generator[None]:
    """Full fp32 convolutions on CUDA (cuDNN uses TF32 for fp32 by default)."""
    if device.type != "cuda":
        yield
        return
    conv = torch.backends.cudnn.conv  # pyright: ignore[reportAttributeAccessIssue]
    previous = conv.fp32_precision
    conv.fp32_precision = "ieee"
    try:
        yield
    finally:
        conv.fp32_precision = previous


def normalize(images: torch.Tensor, cfg: config.DataConfig) -> torch.Tensor:
    """Per-channel `(x - mean) / std` with `cfg.mean/std`."""
    return tv_functional.normalize(images, list(cfg.mean), list(cfg.std))


def apply_view(
    images: torch.Tensor,
    params: ViewParams,
    spec: view_spec.ViewSpec,
    cfg: config.DataConfig,
) -> torch.Tensor:
    """Applies one view's parameters to float `(B, 3, H, W)` images in [0, 1].

    Returns:
        The normalized view `(B, 3, spec.size, spec.size)`.
    """
    out = resized_crop(
        images, params.top, params.left, params.height, params.width, spec.size
    ).clamp(0.0, 1.0)
    out = torch.where(params.flip.reshape(-1, 1, 1, 1), out.flip(-1), out)
    out = color_jitter(out, params)
    gray = grayscale(out).expand_as(out)
    out = torch.where(params.grayscale.reshape(-1, 1, 1, 1), gray, out)
    if spec.blur:
        blurred = gaussian_blur(out, params.sigma, spec.blur_kernel_size)
        out = torch.where(params.blur.reshape(-1, 1, 1, 1), blurred, out)
    return normalize(out, cfg)


@torch.no_grad()
def augment(
    images: torch.Tensor,
    spec: view_spec.ViewSpec,
    cfg: config.DataConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """One random view of every image in a uint8 batch.

    Args:
        images: uint8 `(B, H, W, 3)` or `(B, 3, H, W)` on the generator's device.
        spec: View geometry.
        cfg: Data config (normalization).
        generator: Source of randomness.

    Returns:
        float32 `(B, 3, spec.size, spec.size)`.
    """
    x = to_float(images)
    params = sample_params(spec, x.shape[0], x.shape[-2], x.shape[-1], generator)
    return apply_view(x, params, spec, cfg)


def augmented_view(
    images: torch.Tensor, cfg: config.DataConfig, generator: torch.Generator
) -> torch.Tensor:
    """One random global view (with blur) per image: `(B, 3, G, G)`.

    The embedding dump's augmented views.
    """
    return augment(images, view_spec.global_spec(cfg), cfg, generator)


@torch.no_grad()
def center_view(images: torch.Tensor, cfg: config.DataConfig) -> torch.Tensor:
    """Un-augmented, normalized view `(B, 3, H, W)` of a uint8 batch."""
    return normalize(to_float(images), cfg)


class MultiViewAugment:
    """All training views of a uint8 batch: global views first, then local."""

    def __init__(self, cfg: config.DataConfig) -> None:
        """Reads the view counts, geometry and normalization from `cfg`."""
        self._cfg = cfg
        self._specs = [view_spec.global_spec(cfg)] * cfg.num_global_views + [
            view_spec.local_spec(cfg)
        ] * cfg.num_local_views

    @torch.no_grad()
    def __call__(
        self, images: torch.Tensor, generator: torch.Generator
    ) -> list[torch.Tensor]:
        """Returns `num_global_views` `(B, 3, G, G)` then local `(B, 3, L, L)` views.

        Args:
            images: uint8 `(B, H, W, 3)` or `(B, 3, H, W)` on the generator's device.
            generator: Source of randomness (views are drawn in order).
        """
        x = to_float(images)
        out: list[torch.Tensor] = []
        for spec in self._specs:
            params = sample_params(
                spec, x.shape[0], x.shape[-2], x.shape[-1], generator
            )
            out.append(apply_view(x, params, spec, self._cfg))
        return out
