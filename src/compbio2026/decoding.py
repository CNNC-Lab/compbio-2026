"""Cross-validated linear readouts and the controls that make them mean something.

The rule this module exists to enforce: an accuracy number is meaningless
without a chance level, a held-out split, and a statement of how much data it
took to get there. Every function returns the control alongside the result.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_readout(kind: str = "logistic", C: float = 1.0, alpha: float = 1.0):
    """A standardiser plus a linear classifier.

    ``logistic``
        Multinomial logistic regression. Gives calibrated probabilities and is
        the default.
    ``ridge``
        Ridge classifier. Much faster on wide matrices, which matters when you
        sweep bin widths; use it for exploration and confirm with logistic.
    """
    if kind == "logistic":
        clf = LogisticRegression(C=C, max_iter=2000, multi_class="multinomial")
    elif kind == "ridge":
        clf = RidgeClassifier(alpha=alpha)
    else:
        raise ValueError(f"unknown readout {kind!r}")
    return make_pipeline(StandardScaler(with_mean=True), clf)


def chance_level(y: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class.

    For balanced SHD this is ~1/n_classes, but compute it rather than assume it:
    any subsetting you do can unbalance the classes.
    """
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def shuffle_control(X, y, estimator=None, cv: int = 5, n_repeats: int = 5, seed: int = 0) -> np.ndarray:
    """Accuracies obtained after shuffling the labels.

    A stronger null than :func:`chance_level`: it absorbs any leakage your
    pipeline happens to have. If your real accuracy is not clearly above this
    distribution, you have not shown anything.
    """
    rng = np.random.default_rng(seed)
    est = estimator or make_readout("ridge")
    out = []
    for _ in range(n_repeats):
        yp = rng.permutation(y)
        out.append(cross_val_score(clone(est), X, yp, cv=StratifiedKFold(cv, shuffle=True, random_state=0)).mean())
    return np.asarray(out)


def decode(X, y, estimator=None, cv: int = 5, groups: np.ndarray | None = None, seed: int = 0) -> dict:
    """Cross-validated accuracy with its controls.

    Pass ``groups`` (for instance the speaker id) to hold whole groups out. On
    SHD the speaker-held-out score is markedly lower than the random-split score,
    and it is the one that answers "does this generalise".
    """
    est = estimator or make_readout()
    if groups is None:
        splitter = StratifiedKFold(cv, shuffle=True, random_state=seed)
        scores = cross_val_score(clone(est), X, y, cv=splitter)
    else:
        splitter = GroupKFold(n_splits=min(cv, len(np.unique(groups))))
        scores = cross_val_score(clone(est), X, y, cv=splitter, groups=groups)
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "chance": chance_level(y),
        "grouped": groups is not None,
    }


def learning_curve_trials(X, y, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), estimator=None, cv: int = 5, seed: int = 0):
    """Accuracy against the number of training trials.

    Tells you whether you are data-limited or model-limited. A curve that has
    flattened means more trials will not help and the ceiling is in the
    representation.
    """
    rng = np.random.default_rng(seed)
    est = estimator or make_readout("ridge")
    ns, means, stds = [], [], []
    for f in fractions:
        n = max(int(f * X.shape[0]), 2 * len(np.unique(y)))
        idx = rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False)
        s = cross_val_score(clone(est), X[idx], y[idx], cv=StratifiedKFold(cv, shuffle=True, random_state=seed))
        ns.append(len(idx))
        means.append(s.mean())
        stds.append(s.std())
    return np.array(ns), np.array(means), np.array(stds)


def learning_curve_channels(X3, y, sizes=(10, 25, 50, 100, 200, 400, 700), estimator=None, cv: int = 5, n_repeats: int = 3, seed: int = 0):
    """Accuracy against the number of channels, averaged over random subsets.

    ``X3`` is the un-flattened ``(trials, channels, bins)`` array. Repeating over
    random channel subsets is what separates "the population is redundant" from
    "I happened to pick good channels".
    """
    rng = np.random.default_rng(seed)
    est = estimator or make_readout("ridge")
    n_ch = X3.shape[1]
    out_n, out_m, out_s = [], [], []
    for n in sizes:
        if n > n_ch:
            continue
        reps = []
        for _ in range(n_repeats):
            keep = np.sort(rng.choice(n_ch, size=n, replace=False))
            Xf = X3[:, keep, :].reshape(X3.shape[0], -1)
            reps.append(cross_val_score(clone(est), Xf, y, cv=StratifiedKFold(cv, shuffle=True, random_state=seed)).mean())
        out_n.append(n)
        out_m.append(np.mean(reps))
        out_s.append(np.std(reps))
    return np.array(out_n), np.array(out_m), np.array(out_s)


def accuracy_over_time(X3, y, t_ms, estimator=None, cv: int = 5, cumulative: bool = True, seed: int = 0):
    """When in the trial does the digit become decodable?

    With ``cumulative=True`` the readout at time *t* sees every bin up to *t*,
    which is what a downstream reader would have. With ``cumulative=False`` each
    bin is decoded on its own, which localises information in time but ignores
    that a reader can integrate.
    """
    est = estimator or make_readout("ridge")
    accs = []
    for b in range(X3.shape[2]):
        Xb = X3[:, :, : b + 1] if cumulative else X3[:, :, b : b + 1]
        Xb = Xb.reshape(X3.shape[0], -1)
        accs.append(cross_val_score(clone(est), Xb, y, cv=StratifiedKFold(cv, shuffle=True, random_state=seed)).mean())
    return np.asarray(t_ms), np.asarray(accs)
