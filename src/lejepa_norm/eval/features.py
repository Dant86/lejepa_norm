"""Radius / direction decomposition and image-level summaries of shard views.

Notation follows the Technical Design: for a view embedding `z`, the radius is
`r = ||z||`, the direction `u = z / r` and the log-radius `rho = log r`. Over a set
of views of one image, `rho_bar` is the mean log-radius, `u_bar` the mean
direction and `u_hat = u_bar / ||u_bar||` its normalization.

Shards store view 0 as the un-augmented center view and views 1..K as random
augmentations (`shards.CENTER_VIEW`). Every function that summarizes views takes a
`ViewSelection`: `"center"` uses view 0 only (K = 1), `"augmented"` uses views
1..K. Statistics about augmentation noise (ICC, view consistency) need the
augmented views; the center view is a fixed, non-random view.

Every statistic is computable on both the projector output `z` and the backbone
feature `h`; `Space` names which one.
"""

import dataclasses
import typing

import numpy
import numpy.typing

from lejepa_norm.eval import shards

Space = typing.Literal["z", "h"]
ViewSelection = typing.Literal["center", "augmented"]
SPACES: tuple[Space, ...] = ("z", "h")
VIEW_SELECTIONS: tuple[ViewSelection, ...] = ("center", "augmented")

FloatArray = numpy.typing.NDArray[numpy.float64]

# Guards log(0) and division by zero for degenerate (all-zero) embeddings.
_EPS = 1e-12


def space_array(
    shard: shards.EmbeddingShard, space: Space
) -> numpy.typing.NDArray[numpy.floating[typing.Any]]:
    """Returns the `(N, K + 1, d)` embedding array of `shard` in `space`."""
    return shard.z if space == "z" else shard.h


def select_views(
    x: numpy.typing.NDArray[numpy.floating[typing.Any]], views: ViewSelection
) -> FloatArray:
    """Selects views from an `(N, K + 1, d)` array as float64 `(N, V, d)`.

    Args:
        x: Embeddings with the center view at index `shards.CENTER_VIEW`.
        views: `"center"` (V = 1) or `"augmented"` (V = K).

    Returns:
        The selected views in float64.

    Raises:
        ValueError: `"augmented"` was requested but the array has no augmented
            views.
    """
    x64 = numpy.asarray(x, dtype=numpy.float64)
    if views == "center":
        return x64[:, shards.CENTER_VIEW : shards.CENTER_VIEW + 1]
    if x64.shape[1] < 2:
        raise ValueError("no augmented views (need K + 1 >= 2 views)")
    return numpy.delete(x64, shards.CENTER_VIEW, axis=1)


def radius(x: FloatArray) -> FloatArray:
    """Euclidean norm over the last axis."""
    return numpy.linalg.norm(x, axis=-1)


def direction(x: FloatArray) -> FloatArray:
    """Unit vectors `x / ||x||` over the last axis."""
    return x / numpy.maximum(radius(x), _EPS)[..., None]


def log_radius(x: FloatArray) -> FloatArray:
    """Log of the Euclidean norm over the last axis."""
    return numpy.log(numpy.maximum(radius(x), _EPS))


@dataclasses.dataclass(frozen=True)
class ImageSummary:
    """Image-level summaries of the selected views of N images.

    Attributes:
        rho: Per-view log-radius, `(N, V)`.
        rho_bar: Mean log-radius over views, `(N,)`.
        u_hat: Normalized mean direction, `(N, d)`.
        mean: Mean embedding over views, `(N, d)`.
        dim: Embedding dimension `d`.
    """

    rho: FloatArray
    rho_bar: FloatArray
    u_hat: FloatArray
    mean: FloatArray
    dim: int


def summarize(
    x: numpy.typing.NDArray[numpy.floating[typing.Any]], views: ViewSelection
) -> ImageSummary:
    """Computes `rho`, `rho_bar` and `u_hat` from an `(N, K + 1, d)` array.

    Args:
        x: Embeddings of one space from a shard.
        views: Which views to summarize over.

    Returns:
        The image-level summary.
    """
    sel = select_views(x, views)
    rho = log_radius(sel)
    return ImageSummary(
        rho=rho,
        rho_bar=rho.mean(axis=1),
        u_hat=direction(direction(sel).mean(axis=1)),
        mean=sel.mean(axis=1),
        dim=sel.shape[-1],
    )


def mean_log_radius(
    x: numpy.typing.NDArray[numpy.floating[typing.Any]],
    views: ViewSelection = "augmented",
    chunk: int = 4096,
) -> FloatArray:
    """`rho_bar` of an `(N, K + 1, d)` array, computed in chunks of images.

    Equivalent to `summarize(x, views).rho_bar` but never materializes a float64
    copy of the whole array (STL-10-C has hundreds of thousands of images).
    """
    parts = [
        log_radius(select_views(x[i : i + chunk], views)).mean(axis=1)
        for i in range(0, x.shape[0], chunk)
    ]
    return numpy.concatenate(parts) if parts else numpy.zeros(0)


def view_consistency(
    x: numpy.typing.NDArray[numpy.floating[typing.Any]],
) -> FloatArray:
    """Mean pairwise cosine similarity of `u` across the augmented views.

    Args:
        x: `(N, K + 1, d)` embeddings; the center view is excluded.

    Returns:
        `(N,)` mean over the `K (K - 1) / 2` distinct view pairs.

    Raises:
        ValueError: Fewer than two augmented views.
    """
    u = direction(select_views(x, "augmented"))
    k = u.shape[1]
    if k < 2:
        raise ValueError("view consistency needs at least two augmented views")
    # ||sum_k u_k||^2 = K + 2 * sum_{j<k} u_j . u_k for unit vectors.
    total = numpy.square(u.sum(axis=1)).sum(axis=-1)
    return (total - k) / (k * (k - 1))
