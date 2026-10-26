"""Per-image low-level statistics of the raw prepared images (no model involved).

The statistics describe the 96x96 uint8 RGB image exactly as the model sees it
(`data.gpu_views.center_view`: `to_float` to [0, 1], then a per-channel affine
normalization; the images on disk already have the model resolution, so no resize
happens). They are computed on the [0, 1] image; the per-channel normalization is an
affine map per channel, so the per-channel means / stds are affine transforms of the
statistics of the normalized image and every score built on them below is invariant
to it (standardization, Mahalanobis, ridge, trees).

Statistics (`NAMES`, 16 per image; `LOG_NAMES` are log-transformed before analysis
because they are positive and heavy-tailed):

* `lum_mean`, `lum_std` (contrast): mean / std of the luminance
  `Y = 0.299 R + 0.587 G + 0.114 B`;
* `r_mean`, `g_mean`, `b_mean`, `r_std`, `g_std`, `b_std`: per channel;
* `saturation`: mean HSV saturation `(max - min) / max` over pixels;
* `grad_energy`: mean squared finite-difference gradient of `Y`
  (`mean(dx^2) + mean(dy^2)`), the edge energy;
* `hf_fraction`: fraction of the non-DC power spectrum of `Y` at radial frequency
  above `HF_CUTOFF` cycles per pixel (Nyquist is 0.5);
* `laplacian_var`: variance of the 4-neighbour Laplacian of `Y` (blur measure; low =
  blurry);
* `colorfulness`: Hasler and Suesstrunk (2003), `sqrt(sd_rg^2 + sd_yb^2) + 0.3 *
  sqrt(mu_rg^2 + mu_yb^2)` with `rg = R - G`, `yb = (R + G) / 2 - B`, in [0, 1] units;
* `jpeg_bytes`, `png_bytes`: size of the image encoded with Pillow at JPEG quality
  `JPEG_QUALITY` / PNG `compress_level` `PNG_LEVEL` (entropy proxy).
"""

import io

import numpy
import numpy.typing
import PIL.Image

FloatArray = numpy.typing.NDArray[numpy.float64]
ImageArray = numpy.typing.NDArray[numpy.uint8]

HF_CUTOFF = 0.25
JPEG_QUALITY = 75
PNG_LEVEL = 6
NAMES = (
    "lum_mean",
    "lum_std",
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "saturation",
    "grad_energy",
    "hf_fraction",
    "laplacian_var",
    "colorfulness",
    "jpeg_bytes",
    "png_bytes",
)
LOG_NAMES = (
    "grad_energy",
    "hf_fraction",
    "laplacian_var",
    "jpeg_bytes",
    "png_bytes",
)
LOG_EPS = 1e-8
LUMA = numpy.array([0.299, 0.587, 0.114])


def _radial_mask(size: int) -> numpy.typing.NDArray[numpy.bool_]:
    freq = numpy.fft.fftfreq(size)
    radius = numpy.sqrt(freq[:, None] ** 2 + freq[None, :] ** 2)
    return radius > HF_CUTOFF


def _jpeg_size(image: ImageArray) -> int:
    buffer = io.BytesIO()
    PIL.Image.fromarray(image).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.tell()


def _png_size(image: ImageArray) -> int:
    buffer = io.BytesIO()
    PIL.Image.fromarray(image).save(buffer, format="PNG", compress_level=PNG_LEVEL)
    return buffer.tell()


def compute(images: ImageArray, chunk: int = 256) -> FloatArray:
    """The `NAMES` statistics of uint8 `(N, H, W, 3)` images, `(N, len(NAMES))`.

    Raises:
        ValueError: `images` is not a uint8 `(N, H, W, 3)` array with H == W.
    """
    if (
        images.dtype != numpy.uint8
        or images.ndim != 4
        or images.shape[-1] != 3
        or images.shape[1] != images.shape[2]
    ):
        raise ValueError(f"expected uint8 (N, S, S, 3) images, got {images.shape}")
    mask = _radial_mask(images.shape[1])
    out = numpy.empty((images.shape[0], len(NAMES)))
    for start in range(0, images.shape[0], chunk):
        raw = images[start : start + chunk]
        x = raw.astype(numpy.float64) / 255.0
        lum = x @ LUMA
        rgb_mean = x.mean(axis=(1, 2))
        rgb_std = x.std(axis=(1, 2))
        top, bottom = x.max(axis=-1), x.min(axis=-1)
        sat = numpy.where(top > 0, (top - bottom) / numpy.maximum(top, 1e-12), 0.0)
        dx = numpy.diff(lum, axis=2)
        dy = numpy.diff(lum, axis=1)
        grad = numpy.square(dx).mean(axis=(1, 2)) + numpy.square(dy).mean(axis=(1, 2))
        power = numpy.abs(numpy.fft.fft2(lum - lum.mean(axis=(1, 2))[:, None, None]))
        power = numpy.square(power)
        total = power.sum(axis=(1, 2))
        hf = (power * mask).sum(axis=(1, 2)) / numpy.maximum(total, 1e-12)
        lap = (
            lum[:, 1:-1, 1:-1] * -4.0
            + lum[:, :-2, 1:-1]
            + lum[:, 2:, 1:-1]
            + lum[:, 1:-1, :-2]
            + lum[:, 1:-1, 2:]
        )
        rg = x[..., 0] - x[..., 1]
        yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
        color = numpy.sqrt(
            rg.std(axis=(1, 2)) ** 2 + yb.std(axis=(1, 2)) ** 2
        ) + 0.3 * numpy.sqrt(rg.mean(axis=(1, 2)) ** 2 + yb.mean(axis=(1, 2)) ** 2)
        jpeg = [_jpeg_size(im) for im in raw]
        png = [_png_size(im) for im in raw]
        columns = [
            lum.mean(axis=(1, 2)),
            lum.std(axis=(1, 2)),
            *rgb_mean.T,
            *rgb_std.T,
            sat.mean(axis=(1, 2)),
            grad,
            hf,
            lap.var(axis=(1, 2)),
            color,
            numpy.asarray(jpeg, dtype=numpy.float64),
            numpy.asarray(png, dtype=numpy.float64),
        ]
        out[start : start + raw.shape[0]] = numpy.stack(columns, axis=1)
    return out


def transform(stats: FloatArray) -> FloatArray:
    """Log-transforms the `LOG_NAMES` columns (the scale used by every analysis)."""
    out = stats.copy()
    for name in LOG_NAMES:
        i = NAMES.index(name)
        out[:, i] = numpy.log(out[:, i] + LOG_EPS)
    return out
