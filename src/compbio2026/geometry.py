"""Descriptors of the geometry of a population representation.

Every function here takes a ``(n_samples, n_features)`` matrix — trials by
flattened channel-time features, or trials by embedding coordinates — and
returns a scalar or a small array. They exist so that Stage 1 produces
*numbers* rather than a gallery of embeddings, and so that Stage 2 can ask
whether any of those numbers predicts decoding accuracy.

None of these is a ground truth. They are competing summaries of the same
object, and where they disagree is usually the interesting part.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness as _trustworthiness


def participation_ratio(X: np.ndarray) -> float:
    """Effective dimensionality: ``(Σλ)² / Σλ²`` over the covariance eigenvalues.

    Equals 1 when all variance is on one axis and ``n`` when it is spread evenly
    over ``n`` axes. Unlike "number of PCs to reach 90 % variance" it needs no
    threshold, which is why it is the default here.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.svd(Xc, full_matrices=False, compute_uv=False) ** 2
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float(lam.sum() ** 2 / np.sum(lam**2))


def variance_explained(X: np.ndarray, n_components: int = 20) -> np.ndarray:
    """Cumulative explained-variance ratio of the first ``n_components`` PCs."""
    n = min(n_components, min(X.shape) - 1)
    return np.cumsum(PCA(n_components=n).fit(X).explained_variance_ratio_)


def class_separation(X: np.ndarray, y: np.ndarray) -> float:
    """Ratio of between-class to within-class scatter (a Fisher-style index).

    Large means class means are far apart relative to the spread inside a class.
    Computed on the trace of the scatter matrices, so it is cheap and does not
    require inverting anything.
    """
    classes = np.unique(y)
    mu = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        between += Xc.shape[0] * np.sum((mc - mu) ** 2)
        within += np.sum((Xc - mc) ** 2)
    if within == 0:
        return np.inf
    return float(between / within)


def mean_pairwise_distance(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean distance between class centroids, and mean within-class radius.

    Returned separately rather than as a ratio so you can see which of the two
    moves when a preprocessing choice changes the geometry.
    """
    classes = np.unique(y)
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes])
    d = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    iu = np.triu_indices(len(classes), k=1)
    between = float(d[iu].mean()) if len(classes) > 1 else 0.0
    radii = [float(np.linalg.norm(X[y == c] - X[y == c].mean(axis=0), axis=1).mean()) for c in classes]
    return between, float(np.mean(radii))


def trustworthiness(X: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = 15) -> float:
    """How well an embedding preserves local neighbourhoods (1 is perfect).

    The point of computing this is that it lets you compare PCA, UMAP, Isomap
    and t-SNE on the same axis instead of by eye. It says nothing about whether
    the preserved structure is the structure you care about — pair it with
    :func:`class_separation` or with decoding accuracy.
    """
    n = min(n_neighbors, X.shape[0] // 2 - 1)
    return float(_trustworthiness(X, X_embedded, n_neighbors=max(n, 1)))


def summarize(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """All of the above in one dict, for building a table across conditions."""
    between, within = mean_pairwise_distance(X, y)
    return {
        "participation_ratio": participation_ratio(X),
        "class_separation": class_separation(X, y),
        "between_class_distance": between,
        "within_class_radius": within,
    }
