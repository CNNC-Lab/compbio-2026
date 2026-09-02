"""Structured feature selection and the ablation test that keeps it honest.

Stage 3 asks which part of the population carries the digit. Two things make
that question easy to answer badly:

1. **The wrong unit.** Ranking individual features gives a weight map, not an
   answer. Here a *group* is one cochlear channel across all of its time bins,
   so the group penalty selects channels and the within-group penalty keeps each
   selected channel's temporal support tight.
2. **No necessity test.** A selected set that the decoder *can* use is not a set
   the decoder *needs*. :func:`ablate` removes the selection, refits from
   scratch, and reports the drop. In a redundant code that drop is small, and its
   size is the most informative number in the project.

Framing to state explicitly in the report: selection and ablation here establish
causal relevance **for the decoder, not for the circuit**. The data are fixed
recordings and nothing is intervened upon in the biological system. Confusing
the two is a live error in the interpretability literature.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .decoding import chance_level, make_readout


def channel_groups(n_channels: int, n_bins: int) -> np.ndarray:
    """Group index per feature for a channel-major flattened matrix.

    Matches :func:`compbio2026.data.flatten`: feature ``c * n_bins + b`` belongs
    to group ``c``.
    """
    return np.repeat(np.arange(n_channels), n_bins)


# --------------------------------------------------------------------------
# Heuristic baselines — the things a structured method has to beat
# --------------------------------------------------------------------------

def select_by_rate(X3: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` channels with the highest mean firing rate."""
    score = X3.mean(axis=(0, 2))
    return np.sort(np.argsort(score)[::-1][:k])


def select_by_variance(X3: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` channels whose trial-averaged activity varies most over time."""
    score = X3.mean(axis=0).var(axis=-1)
    return np.sort(np.argsort(score)[::-1][:k])


def select_by_discriminability(X3: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` channels with the largest univariate class discriminability.

    One ANOVA F per channel on its time-averaged activity. This is the baseline
    that most often wins, and if it matches the structured method that is a
    result, not a failure.
    """
    F, _ = f_classif(X3.mean(axis=-1), y)
    F = np.nan_to_num(F)
    return np.sort(np.argsort(F)[::-1][:k])


def select_random(n_channels: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` channels at random — the floor every method must clear."""
    return np.sort(rng.choice(n_channels, size=k, replace=False))


# --------------------------------------------------------------------------
# Sparse group lasso
# --------------------------------------------------------------------------

def sparse_group_lasso_channels(
    X3: np.ndarray,
    y: np.ndarray,
    group_reg: float = 0.05,
    l1_reg: float = 0.01,
    k: int | None = None,
    scale: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Select channels with a sparse group lasso.

    Parameters
    ----------
    X3
        ``(trials, channels, bins)``.
    group_reg, l1_reg
        Group-level and within-group regularisation. Increase ``group_reg`` to
        select fewer channels. Tune them by cross-validation rather than by
        eye — and report the path, not one point on it.
    k
        If given, return the ``k`` channels with the largest group norm instead
        of whatever the regularisation happens to zero out. Useful for a
        like-for-like comparison against the heuristics, which take a fixed
        ``k``.

    Returns
    -------
    channels, group_norms
        The selected channel indices and the L2 norm of each channel's
        coefficient block (its "importance", for the whole channel at once).

    Notes
    -----
    Requires the ``group-lasso`` package. If it is missing, this raises with the
    install command rather than silently substituting something else.
    """
    try:
        from group_lasso import LogisticGroupLasso
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pip install group-lasso") from exc

    n_trials, n_ch, n_bins = X3.shape
    X = X3.reshape(n_trials, -1)
    if scale:
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    groups = channel_groups(n_ch, n_bins)

    model = LogisticGroupLasso(
        groups=groups,
        group_reg=group_reg,
        l1_reg=l1_reg,
        scale_reg="inverse_group_size",
        supress_warning=True,
        **kwargs,
    )
    model.fit(X, y)

    coef = np.asarray(model.coef_)
    if coef.ndim == 1:
        coef = coef[:, None]
    norms = np.linalg.norm(coef.reshape(n_ch, n_bins, -1), axis=(1, 2))

    if k is not None:
        channels = np.sort(np.argsort(norms)[::-1][:k])
    else:
        channels = np.flatnonzero(norms > 1e-10)
    return channels, norms


# --------------------------------------------------------------------------
# The necessity test
# --------------------------------------------------------------------------

def ablate(X3, y, channels, estimator=None, cv: int = 5, seed: int = 0) -> dict:
    """Refit with the selected channels removed, and with only them kept.

    Returns
    -------
    dict with ``full``, ``kept`` (only the selection), ``ablated`` (everything
    but the selection), ``chance``, and ``compensation`` — the gap between the
    full accuracy and the ablated accuracy.

    A small ``compensation`` after removing a set the selector called essential
    means the code was redundant and the set was never necessary. Report that
    number; it is more informative than the selection itself.
    """
    est = estimator or make_readout("ridge")
    splitter = StratifiedKFold(cv, shuffle=True, random_state=seed)
    channels = np.asarray(channels)
    mask = np.zeros(X3.shape[1], dtype=bool)
    mask[channels] = True

    def _score(Xa):
        if Xa.shape[1] == 0:
            return chance_level(y)
        return float(cross_val_score(clone(est), Xa.reshape(Xa.shape[0], -1), y, cv=splitter).mean())

    full = _score(X3)
    kept = _score(X3[:, mask, :])
    ablated = _score(X3[:, ~mask, :])
    return {
        "n_selected": int(mask.sum()),
        "full": full,
        "kept": kept,
        "ablated": ablated,
        "chance": chance_level(y),
        "compensation": full - ablated,
    }


def compare_selectors(X3, y, k: int, seed: int = 0, include_sgl: bool = True, **sgl_kwargs) -> dict:
    """Run every selector at the same ``k`` and ablate each one.

    The output table is the deliverable of Stage 3: it says whether structured
    selection buys anything over a firing-rate threshold.
    """
    rng = np.random.default_rng(seed)
    selectors = {
        "random": select_random(X3.shape[1], k, rng),
        "firing_rate": select_by_rate(X3, k),
        "variance": select_by_variance(X3, k),
        "discriminability": select_by_discriminability(X3, y, k),
    }
    if include_sgl:
        try:
            selectors["sparse_group_lasso"], _ = sparse_group_lasso_channels(X3, y, k=k, **sgl_kwargs)
        except ImportError:
            pass
    return {name: {**ablate(X3, y, ch, seed=seed), "channels": ch} for name, ch in selectors.items()}
