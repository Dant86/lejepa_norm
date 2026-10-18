"""Per-image difficulty scores and the paired radius shift under corruption.

Difficulty scores (labels used for analysis only) are computed per query image:
corruption severity, cosine k-NN margin of `u_hat` against a labeled bank, view
consistency, and the entropy of a linear probe on `u_hat`. The paired radius
shift `delta_rho(x, c, s) = rho_bar(corrupt_{c,s}(x)) - rho_bar(x)` compares each
corrupted image with its own clean version, which removes class and identity
effects by design.
"""

import dataclasses
import typing

import numpy
import numpy.typing

from lejepa_norm.eval import downstream
from lejepa_norm.eval import features
from lejepa_norm.eval import metrics

FloatArray = numpy.typing.NDArray[numpy.float64]


def paired_radius_shift(
    clean_rho_bar: numpy.typing.ArrayLike,
    clean_image_id: numpy.typing.ArrayLike,
    corrupt_rho_bar: numpy.typing.ArrayLike,
    corrupt_image_id: numpy.typing.ArrayLike,
) -> FloatArray:
    """Per corrupted image, `rho_bar(corrupted) - rho_bar(clean twin)`.

    Args:
        clean_rho_bar: `(M,)` mean log-radius of the clean images.
        clean_image_id: `(M,)` unique ids of the clean images.
        corrupt_rho_bar: `(N,)` mean log-radius of the corrupted images.
        corrupt_image_id: `(N,)` id of each corrupted image's clean twin.

    Returns:
        `(N,)` paired shifts.

    Raises:
        ValueError: Clean ids are not unique, or a corrupted image has no clean
            twin.
    """
    ids = numpy.asarray(clean_image_id).ravel()
    order = numpy.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if numpy.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("clean image ids must be unique")
    query = numpy.asarray(corrupt_image_id).ravel()
    pos = numpy.clip(numpy.searchsorted(sorted_ids, query), 0, ids.size - 1)
    if ids.size == 0 or numpy.any(sorted_ids[pos] != query):
        raise ValueError("some corrupted images have no clean twin")
    clean = numpy.asarray(clean_rho_bar, dtype=numpy.float64).ravel()[order]
    return numpy.asarray(corrupt_rho_bar, dtype=numpy.float64).ravel() - clean[pos]


@dataclasses.dataclass(frozen=True)
class ShiftRow:
    """Mean paired radius shift for one (corruption, severity) cell."""

    corruption: str
    severity: int
    mean: metrics.Estimate


def shift_by_severity(
    delta: numpy.typing.ArrayLike,
    corruption: numpy.typing.ArrayLike,
    severity: numpy.typing.ArrayLike,
    num_bootstrap: int,
    rng: numpy.random.Generator,
) -> list[ShiftRow]:
    """Mean `delta_rho` with bootstrap CI per corruption type and severity."""
    d = numpy.asarray(delta, dtype=numpy.float64).ravel()
    c = numpy.asarray(corruption).ravel()
    s = numpy.asarray(severity).ravel()
    rows: list[ShiftRow] = []
    for name in sorted(set(c.tolist())):
        for level in sorted(set(s[c == name].tolist())):
            cell = d[(c == name) & (s == level)]
            est = metrics.bootstrap(
                lambda idx, cell=cell: float(cell[idx].mean()),
                cell.size,
                num_bootstrap,
                rng,
            )
            rows.append(ShiftRow(str(name), int(level), est))
    return rows


def scores(
    bank: features.ImageSummary,
    bank_labels: numpy.typing.ArrayLike,
    query: features.ImageSummary,
    query_labels: numpy.typing.ArrayLike,
    query_views: numpy.typing.NDArray[numpy.floating[typing.Any]],
    query_severity: numpy.typing.ArrayLike,
    k: int,
    num_classes: int,
) -> dict[str, FloatArray]:
    """Computes every difficulty score for the query images.

    Args:
        bank: Summary of the labeled bank (augmented or center views).
        bank_labels: `(M,)` bank labels.
        query: Summary of the query images.
        query_labels: `(N,)` query labels (analysis only).
        query_views: `(N, K + 1, d)` raw query embeddings, for view consistency.
        query_severity: `(N,)` corruption severity (0 = clean).
        k: Neighbours for the k-NN margin.
        num_classes: Number of classes.

    Returns:
        Score name to `(N,)` array: `severity`, `knn_margin`, `view_consistency`,
        `probe_entropy`. Larger margin / consistency means easier; larger
        severity / entropy means harder.
    """
    votes = downstream.knn_votes(bank.u_hat, bank_labels, query.u_hat, k, num_classes)
    probe = downstream.LinearProbe(bank.u_hat, bank_labels, num_classes)
    return {
        "severity": numpy.asarray(query_severity, dtype=numpy.float64).ravel(),
        "knn_margin": downstream.vote_margin(votes, query_labels),
        "view_consistency": features.view_consistency(query_views),
        "probe_entropy": downstream.entropy(probe.predict_proba(query.u_hat)),
    }
