# pyright: basic, reportIndexIssue=false, reportCallIssue=false
# pyright: reportArgumentType=false, reportOperatorIssue=false
# pyright: reportAttributeAccessIssue=false
# ruff: noqa: D103
"""Hendrycks-Dietterich corruptions, vendored from `imagecorruptions` 1.1.2.

Derived from https://github.com/bethgelab/imagecorruptions (Apache-2.0), see
NOTICE and LICENSE in this directory. Each corruption takes a PIL RGB image, a
severity in 1-5 and a `numpy.random.Generator`, and returns an image-like object
(array in [0, 255] or PIL image).
"""

import collections.abc
import io
import math

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import skimage.filters
import skimage.util

# /////////////// Corruption helpers ///////////////


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        axis = np.arange(-8, 8 + 1)
        truncate_radius = 1  # was cv2 ksize (3, 3)
    else:
        axis = np.arange(-radius, radius + 1)
        truncate_radius = 2  # was cv2 ksize (5, 5)
    xx, yy = np.meshgrid(axis, axis)
    aliased_disk = np.array((xx**2 + yy**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    # Supersample disk to antialias.
    return scipy.ndimage.gaussian_filter(
        aliased_disk,
        sigma=alias_blur,
        truncate=truncate_radius / alias_blur,
        mode="mirror",
    )


def plasma_fractal(rng, mapsize=256, wibbledecay=3):
    """Diamond-square heightmap, square array of side `mapsize` in [0, 1]."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        half = stepsize // 2
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[half:mapsize:stepsize, half:mapsize:stepsize] = wibbledmean(
            squareaccum
        )

    def filldiamonds():
        size = maparray.shape[0]
        half = stepsize // 2
        drgrid = maparray[half:size:stepsize, half:size:stepsize]
        ulgrid = maparray[0:size:stepsize, 0:size:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:size:stepsize, half:size:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[half:size:stepsize, 0:size:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2
    return scipy.ndimage.zoom(
        img[top0 : top0 + ch0, top1 : top1 + ch1],
        (zoom_factor, zoom_factor, 1),
        order=1,
    )


def _gauss_function(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def _motion_blur_kernel(width, sigma):
    k = _gauss_function(np.arange(width), 0, sigma)
    return k / np.sum(k)


def _shift(image, dx, dy):
    if dx < 0:
        shifted = np.roll(image, shift=image.shape[1] + dx, axis=1)
        shifted[:, dx:] = shifted[:, dx - 1 : dx]
    elif dx > 0:
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx : dx + 1]
    else:
        shifted = image

    if dy < 0:
        shifted = np.roll(shifted, shift=image.shape[0] + dy, axis=0)
        shifted[dy:, :] = shifted[dy - 1 : dy, :]
    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy : dy + 1, :]
    return shifted


def _motion_blur(x, radius, sigma, angle):
    width = radius * 2 + 1
    kernel = _motion_blur_kernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]:
            break  # simulated motion exceeded image borders
        blurred = blurred + kernel[i] * _shift(x, dx, dy)
    return blurred


def _rgb_to_gray(x):
    """ITU-R 601 luma, as `cv2.cvtColor(..., COLOR_RGB2GRAY)`."""
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def _gaussian(x, sigma):
    return skimage.filters.gaussian(x, sigma=sigma, channel_axis=-1)


# /////////////// Corruptions ///////////////


def gaussian_noise(x, severity, rng):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity, rng):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(rng.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity, rng):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    x = skimage.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c, rng=rng)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity, rng):
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + x * rng.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity, rng):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = _gaussian(np.array(x) / 255.0, c)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity, rng):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(_gaussian(np.array(x) / 255.0, c[0]) * 255)
    x_shape = np.array(x).shape

    # Locally shuffle pixels.
    for _ in range(c[2]):
        for h in range(x_shape[0] - c[1], c[1], -1):
            for w in range(x_shape[1] - c[1], c[1], -1):
                dx, dy = rng.integers(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(_gaussian(x / 255.0, c[0]), 0, 1) * 255


def defocus_blur(x, severity, rng):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = [
        scipy.ndimage.correlate(x[:, :, d], kernel, mode="mirror") for d in range(3)
    ]
    return np.clip(np.stack(channels, axis=-1), 0, 1) * 255


def motion_blur(x, severity, rng):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.array(x)
    angle = rng.uniform(-45, 45)
    x = _motion_blur(x, radius=c[0], sigma=c[1], angle=angle)
    return np.clip(x, 0, 255)


def zoom_blur(x, severity, rng):
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        zoom_layer = clipped_zoom(x, zoom_factor)[: x.shape[0], : x.shape[1], :]
        try:
            out += zoom_layer
        except ValueError:
            out[: zoom_layer.shape[0], : zoom_layer.shape[1]] += zoom_layer
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def fog(x, severity, rng):
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    shape = np.array(x).shape
    map_size = _next_power_of_2(int(np.max(shape)))

    x = np.array(x) / 255.0
    max_val = x.max()
    fractal = plasma_fractal(rng, mapsize=map_size, wibbledecay=c[1])
    x += c[0] * fractal[: shape[0], : shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def snow(x, severity, rng):
    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.0
    snow_layer = rng.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0
    snow_layer = np.clip(snow_layer.squeeze(), 0, 1)
    snow_layer = _motion_blur(
        snow_layer, radius=c[4], sigma=c[5], angle=rng.uniform(-135, -45)
    )

    # The snow layer is rounded and cropped to the image dims.
    snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.0
    snow_layer = snow_layer[..., np.newaxis]
    snow_layer = snow_layer[: x.shape[0], : x.shape[1], :]

    gray = _rgb_to_gray(x).reshape(x.shape[0], x.shape[1], 1)
    x = c[6] * x + (1 - c[6]) * np.maximum(x, gray * 1.5 + 0.5)
    try:
        return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    except ValueError:
        x[: snow_layer.shape[0], : snow_layer.shape[1]] += snow_layer + np.rot90(
            snow_layer, k=2
        )
        return np.clip(x, 0, 1) * 255


def contrast(x, severity, rng):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity, rng):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    x = skimage.color.rgb2hsv(np.array(x) / 255.0)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = skimage.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def saturate(x, severity, rng):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = skimage.color.rgb2hsv(np.array(x) / 255.0)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = skimage.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity, rng):
    c = [25, 18, 15, 10, 7][severity - 1]
    output = io.BytesIO()
    x.save(output, "JPEG", quality=c)
    return Image.open(output)


def pixelate(x, severity, rng):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    width, height = x.size
    x = x.resize((int(width * c), int(height * c)), Image.Resampling.BOX)
    return x.resize((width, height), Image.Resampling.NEAREST)


def elastic_transform(image, severity, rng):
    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    sigma = np.array(shape[:2]) * 0.01
    alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][severity - 1]
    max_dx = shape[0] * 0.005
    max_dy = shape[0] * 0.005

    dx = (
        skimage.filters.gaussian(
            rng.uniform(-max_dx, max_dx, size=shape[:2]),
            sigma,
            mode="reflect",
            truncate=3,
        )
        * alpha
    ).astype(np.float32)
    dy = (
        skimage.filters.gaussian(
            rng.uniform(-max_dy, max_dy, size=shape[:2]),
            sigma,
            mode="reflect",
            truncate=3,
        )
        * alpha
    ).astype(np.float32)

    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    warped = scipy.ndimage.map_coordinates(image, indices, order=1, mode="reflect")
    return np.clip(warped.reshape(shape), 0, 1) * 255


REGISTRY: dict[str, collections.abc.Callable[..., object]] = {
    f.__name__: f
    for f in (
        gaussian_noise,
        shot_noise,
        impulse_noise,
        defocus_blur,
        glass_blur,
        motion_blur,
        zoom_blur,
        snow,
        fog,
        brightness,
        contrast,
        elastic_transform,
        pixelate,
        jpeg_compression,
        speckle_noise,
        gaussian_blur,
        saturate,
    )
}
