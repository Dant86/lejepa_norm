"""Deterministic Hendrycks-Dietterich corruptions for STL-10-C.

See NOTICE and LICENSE in this directory: the corruption code is vendored from
`imagecorruptions` 1.1.2 (Apache-2.0), https://arxiv.org/abs/1903.12261.
"""

import zlib

import numpy as np
import PIL.Image

from lejepa_norm import config
from lejepa_norm.data.corruptions import functions

SEVERITIES = (1, 2, 3, 4, 5)


def available() -> tuple[str, ...]:
    """Names of all vendored corruptions (the full set)."""
    return tuple(functions.REGISTRY)


def resolve_names(cfg: config.DataConfig) -> tuple[str, ...]:
    """Returns the corruptions to generate.

    That is every vendored corruption if `full_corruption_set`, else the configured
    subset.

    Raises:
        KeyError: A configured corruption is not available.
    """
    if cfg.full_corruption_set:
        return available()
    unknown = sorted(set(cfg.corruptions) - set(functions.REGISTRY))
    if unknown:
        raise KeyError(f"unknown corruptions {unknown}; available: {available()}")
    return cfg.corruptions


def _rng(seed: int, index: int, name: str, severity: int) -> np.random.Generator:
    key = [seed, index, zlib.crc32(name.encode()), severity]
    return np.random.default_rng(np.random.SeedSequence(key))


def corrupt(
    image: np.ndarray,
    name: str,
    severity: int,
    *,
    seed: int = 0,
    index: int = 0,
) -> np.ndarray:
    """Corrupts one image deterministically.

    The random stream depends only on `(seed, index, name, severity)`, so results
    are reproducible and independent of processing order or parallelism.

    Args:
        image: uint8 RGB array of shape `(H, W, 3)`.
        name: Corruption name, see `available()`.
        severity: 1-5.
        seed: Base seed (`DataConfig.corruption_seed`).
        index: Index of the image in its dataset.

    Returns:
        uint8 array with the same shape as `image`.

    Raises:
        ValueError: `image` or `severity` is invalid.
        KeyError: `name` is unknown.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected uint8 (H, W, 3), got {image.dtype} {image.shape}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")
    if name not in functions.REGISTRY:
        raise KeyError(f"unknown corruption {name!r}; available: {available()}")
    out = functions.REGISTRY[name](
        PIL.Image.fromarray(image),  # pyright: ignore[reportUnknownMemberType]
        severity,
        _rng(seed, index, name, severity),
    )
    return np.asarray(out).astype(np.uint8)
