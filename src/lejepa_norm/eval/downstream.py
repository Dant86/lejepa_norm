"""Frozen-feature classifiers: cosine k-NN and a linear (logistic) probe.

Both are fitted on a labeled bank (STL-10 `train`) and applied to queries
(STL-10 `test`, possibly corrupted). Besides accuracy they expose per-query
quantities used as difficulty scores: the k-NN vote margin and the entropy of the
probe's softmax.
"""

# scikit-learn ships no type stubs; its calls are confined to `LinearProbe`.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import typing
import warnings

import numpy
import numpy.typing
import sklearn.exceptions
import sklearn.linear_model
import sklearn.neural_network

from lejepa_norm.eval import features

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.int64]
FeatureKind = typing.Literal["embedding", "u_hat", "u_hat+rho"]
FEATURE_KINDS: tuple[FeatureKind, ...] = ("embedding", "u_hat", "u_hat+rho")


def feature_pair(
    bank: features.ImageSummary,
    query: features.ImageSummary,
    kind: FeatureKind,
) -> tuple[FloatArray, FloatArray]:
    """Builds bank and query feature matrices of one kind.

    Args:
        bank: Summary of the labeled bank images.
        query: Summary of the query images.
        kind: `"embedding"` (mean embedding over the selected views), `"u_hat"`,
            or `"u_hat+rho"` (`u_hat` with `rho_bar` appended as a column,
            z-scored with the bank's statistics so that it is on the scale of a
            unit-vector coordinate for cosine k-NN).

    Returns:
        `(bank_x, query_x)`.
    """
    if kind == "embedding":
        return bank.mean, query.mean
    if kind == "u_hat":
        return bank.u_hat, query.u_hat
    mu = float(bank.rho_bar.mean())
    sd = float(bank.rho_bar.std()) or 1.0
    return (
        numpy.hstack([bank.u_hat, ((bank.rho_bar - mu) / sd)[:, None]]),
        numpy.hstack([query.u_hat, ((query.rho_bar - mu) / sd)[:, None]]),
    )


def _unit(x: numpy.typing.ArrayLike) -> FloatArray:
    arr = numpy.asarray(x, dtype=numpy.float64)
    return arr / numpy.maximum(numpy.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def knn_votes(
    bank: numpy.typing.ArrayLike,
    bank_labels: numpy.typing.ArrayLike,
    query: numpy.typing.ArrayLike,
    k: int,
    num_classes: int,
    chunk: int = 1024,
) -> FloatArray:
    """Cosine k-NN vote shares of each class for each query.

    Args:
        bank: `(M, d)` labeled features.
        bank_labels: `(M,)` integer labels in `[0, num_classes)`.
        query: `(N, d)` query features.
        k: Number of neighbours (capped at M).
        num_classes: Number of classes.
        chunk: Queries processed per block (bounds memory to `chunk * M`).

    Returns:
        `(N, num_classes)` fraction of the `k` neighbours in each class.
    """
    b = _unit(bank)
    q = _unit(query)
    labels = numpy.asarray(bank_labels, dtype=numpy.int64).ravel()
    k = min(k, b.shape[0])
    votes = numpy.zeros((q.shape[0], num_classes), dtype=numpy.float64)
    for start in range(0, q.shape[0], chunk):
        sims = q[start : start + chunk] @ b.T
        nearest = numpy.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = numpy.repeat(numpy.arange(nearest.shape[0]), k)
        numpy.add.at(votes[start : start + chunk], (rows, labels[nearest].ravel()), 1)
    return votes / k


def vote_margin(
    votes: numpy.typing.ArrayLike, labels: numpy.typing.ArrayLike
) -> FloatArray:
    """Vote share of the true class minus the best other class, `(N,)`."""
    v = numpy.asarray(votes, dtype=numpy.float64)
    y = numpy.asarray(labels, dtype=numpy.int64).ravel()
    rows = numpy.arange(v.shape[0])
    true_share = v[rows, y]
    others = v.copy()
    others[rows, y] = -numpy.inf
    return true_share - others.max(axis=1)


def accuracy(
    predictions: numpy.typing.ArrayLike, labels: numpy.typing.ArrayLike
) -> float:
    """Fraction of correct predictions."""
    p = numpy.asarray(predictions).ravel()
    y = numpy.asarray(labels).ravel()
    return float((p == y).mean())


def entropy(probs: numpy.typing.ArrayLike) -> FloatArray:
    """Shannon entropy (nats) of each row of a probability matrix."""
    p = numpy.asarray(probs, dtype=numpy.float64)
    logp = numpy.log(numpy.clip(p, 1e-300, None))
    return -(p * logp).sum(axis=1)


class LinearProbe:
    """Multinomial logistic regression on standardized frozen features."""

    def __init__(
        self,
        x: numpy.typing.ArrayLike,
        y: numpy.typing.ArrayLike,
        num_classes: int,
        c: float = 1.0,
        max_iter: int = 1000,
        center: bool = True,
        fit_intercept: bool = True,
    ) -> None:
        """Fits the probe.

        Args:
            x: `(M, d)` training features.
            y: `(M,)` integer labels in `[0, num_classes)`.
            num_classes: Number of classes; columns of `predict_proba`.
            c: Inverse L2 regularization strength.
            max_iter: L-BFGS iterations.
            center: Subtract the training mean of every feature (the features
                are always divided by their training standard deviation).
            fit_intercept: Fit a class intercept. With neither centering nor an
                intercept the logits are linear homogeneous in `x`, so the
                predicted class is invariant to a rescaling of `x`.
        """
        arr = numpy.asarray(x, dtype=numpy.float64)
        self._mean = arr.mean(axis=0) if center else numpy.zeros(arr.shape[1])
        self._std = numpy.where(arr.std(axis=0) > 0, arr.std(axis=0), 1.0)
        self._num_classes = num_classes
        self._model = sklearn.linear_model.LogisticRegression(
            C=c, max_iter=max_iter, fit_intercept=fit_intercept
        )
        self._model.fit(
            (arr - self._mean) / self._std, numpy.asarray(y, dtype=numpy.int64)
        )

    def predict_proba(self, x: numpy.typing.ArrayLike) -> FloatArray:
        """Softmax probabilities, `(N, num_classes)` (zero for unseen classes)."""
        arr = (numpy.asarray(x, dtype=numpy.float64) - self._mean) / self._std
        seen = numpy.asarray(self._model.classes_, dtype=numpy.int64)
        out = numpy.zeros((arr.shape[0], self._num_classes), dtype=numpy.float64)
        out[:, seen] = numpy.asarray(self._model.predict_proba(arr))
        return out

    def affine(self) -> tuple[FloatArray, FloatArray]:
        """The logits as an affine map of the raw features: `x @ W.T + b`.

        Returns:
            `(W, b)` of shapes `(C, d)` and `(C,)` over the classes seen in
            training, with the standardization folded in (the logits equal
            `log` of `predict_proba` up to a per-row constant).
        """
        coef = numpy.asarray(self._model.coef_, dtype=numpy.float64)
        weights = coef / self._std
        bias = numpy.asarray(self._model.intercept_, dtype=numpy.float64) - (
            weights @ self._mean
        )
        return weights, bias

    def predict(self, x: numpy.typing.ArrayLike) -> IntArray:
        """Most probable class, `(N,)`."""
        return numpy.argmax(self.predict_proba(x), axis=1).astype(numpy.int64)


class MlpProbe:
    """One-hidden-layer MLP on standardized frozen features, fixed training budget.

    The number of epochs is fixed (no early stopping); `weight_decay` is the L2
    penalty (scikit-learn's `alpha`) and is chosen by the caller.
    """

    def __init__(
        self,
        x: numpy.typing.ArrayLike,
        y: numpy.typing.ArrayLike,
        num_classes: int,
        weight_decay: float,
        hidden: int = 256,
        epochs: int = 100,
        seed: int = 0,
    ) -> None:
        """Fits the MLP.

        Args:
            x: `(M, d)` training features.
            y: `(M,)` integer labels in `[0, num_classes)`.
            num_classes: Number of classes; columns of `predict_proba`.
            weight_decay: L2 penalty.
            hidden: Hidden units.
            epochs: Adam epochs (all of them are run).
            seed: Initialization and shuffling seed.
        """
        arr = numpy.asarray(x, dtype=numpy.float64)
        self._mean = arr.mean(axis=0)
        self._std = numpy.where(arr.std(axis=0) > 0, arr.std(axis=0), 1.0)
        self._num_classes = num_classes
        self._model = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(hidden,),
            alpha=weight_decay,
            max_iter=epochs,
            n_iter_no_change=epochs + 1,
            tol=0.0,
            random_state=seed,
        )
        with warnings.catch_warnings():
            # Reaching the fixed budget is intended, not a convergence failure.
            warnings.simplefilter("ignore", sklearn.exceptions.ConvergenceWarning)
            self._model.fit(
                (arr - self._mean) / self._std, numpy.asarray(y, dtype=numpy.int64)
            )

    def predict_proba(self, x: numpy.typing.ArrayLike) -> FloatArray:
        """Softmax probabilities, `(N, num_classes)` (zero for unseen classes)."""
        arr = (numpy.asarray(x, dtype=numpy.float64) - self._mean) / self._std
        seen = numpy.asarray(self._model.classes_, dtype=numpy.int64)
        out = numpy.zeros((arr.shape[0], self._num_classes), dtype=numpy.float64)
        out[:, seen] = numpy.asarray(self._model.predict_proba(arr))
        return out

    def predict(self, x: numpy.typing.ArrayLike) -> IntArray:
        """Most probable class, `(N,)` (unseen classes are never predicted)."""
        return numpy.argmax(self.predict_proba(x), axis=1).astype(numpy.int64)
