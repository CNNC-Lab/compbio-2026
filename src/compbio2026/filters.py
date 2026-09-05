"""Turning spike trains into continuous signals.

A spike train is a sum of delta functions; almost nothing downstream wants that.
Convolving with a kernel gives a continuous trace — which is what a synapse does
to its input, and what a readout needs to work on.

The kernel is a modelling choice with consequences:

``exponential``
    :math:`k(t) = e^{-t/\\tau}` for :math:`t \\ge 0`. Causal, one parameter, and
    the same shape as a first-order synaptic conductance. The default.
``alpha``
    :math:`k(t) = (t/\\tau)\\,e^{1 - t/\\tau}`. Causal, with a rise time — closer
    to a real PSP, and it delays the response by roughly :math:`\\tau`.
``gaussian``
    Symmetric and **non-causal**: the filtered signal at time *t* depends on
    spikes that have not happened yet. Fine for offline analysis, wrong for
    anything claiming to be a mechanism.

``tau`` sets the memory of the filter and therefore how much temporal detail
survives. It is the same decision as bin width, made continuously.
"""

from __future__ import annotations

import numpy as np


def kernel(kind: str = "exponential", tau_ms: float = 20.0, dt_ms: float = 1.0,
           length_factor: float = 5.0) -> np.ndarray:
    """Build a normalised filter kernel sampled at ``dt_ms``."""
    n = max(int(round(length_factor * tau_ms / dt_ms)), 2)
    t = np.arange(n) * dt_ms
    if kind == "exponential":
        k = np.exp(-t / tau_ms)
    elif kind == "alpha":
        k = (t / tau_ms) * np.exp(1.0 - t / tau_ms)
    elif kind == "gaussian":
        t = np.arange(-n, n + 1) * dt_ms
        k = np.exp(-0.5 * (t / tau_ms) ** 2)
    else:
        raise ValueError(f"unknown kernel {kind!r}")
    return k / k.sum()


def filter_spikes(X: np.ndarray, kind: str = "exponential", tau_ms: float = 20.0,
                  dt_ms: float = 1.0) -> np.ndarray:
    """Convolve binned spike counts with a kernel, along the time axis.

    Parameters
    ----------
    X
        ``(n_trials, n_channels, n_bins)`` spike counts.
    dt_ms
        Bin width of ``X`` in milliseconds — needed to sample the kernel on the
        same grid.

    Returns
    -------
    Array of the same shape, continuous-valued. Causal kernels are trimmed so
    the output stays aligned with the input; the Gaussian is centred.
    """
    k = kernel(kind, tau_ms, dt_ms)
    n_pad = len(k) - 1
    out = np.empty_like(X, dtype=np.float32)
    flat = X.reshape(-1, X.shape[-1])
    res = np.empty_like(flat, dtype=np.float32)
    for i, row in enumerate(flat):
        c = np.convolve(row, k, mode="full")
        res[i] = c[: X.shape[-1]] if kind != "gaussian" else c[n_pad // 2 : n_pad // 2 + X.shape[-1]]
    return res.reshape(X.shape)


def pool_channels(X: np.ndarray, n_bands: int, reduce: str = "mean") -> np.ndarray:
    """Combine adjacent channels into ``n_bands`` tonotopic bands.

    Channel index in SHD maps monotonically onto frequency, so combining
    neighbours is combining over a frequency band — a meaningful operation,
    unlike combining a random subset.

    ``reduce``
        ``"mean"`` averages within the band. ``"sum"`` adds, which is what
        synaptic convergence onto a single postsynaptic cell actually does and
        which preserves total spike count; it makes wide bands louder than
        narrow ones, so only use it when the bands are equally wide.

    Pooling destroys single-channel identity: a band is a mixture, and a mixture
    is not a neuron. Anything you conclude afterwards is a claim about a
    frequency band. If you need the claim to be about channels, use
    :func:`select_channels` instead and pay for it in discarded data.
    """
    if reduce not in {"mean", "sum"}:
        raise ValueError(f"unknown reduce {reduce!r}")
    n_ch = X.shape[1]
    edges = np.linspace(0, n_ch, n_bands + 1).astype(int)
    op = np.mean if reduce == "mean" else np.sum
    return np.stack([op(X[:, a:b, :], axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)


def select_channels(
    X: np.ndarray,
    n: int,
    method: str = "uniform",
    rng: np.random.Generator | None = None,
    idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ``n`` whole channels instead of averaging channels together.

    The alternative to :func:`pool_channels`. Both reduce the width of the
    design matrix; they differ in what they destroy. Pooling keeps every spike
    but loses the identity of the neuron that fired it. Selection keeps identity
    — every retained column is still one bushy cell, and a weight on it is a
    statement about that cell — but throws away everything you did not pick.

    Which is right depends on the claim. Anything about receptive fields,
    ablation, or "which neurons matter" needs selection. Anything about
    frequency-band energy is fine, and cheaper, with pooling.

    Parameters
    ----------
    method
        ``"uniform"``
            Evenly spaced channel indices. Because the index is tonotopic this
            is a regular sample of the frequency axis — the closest thing to an
            unbiased subset, and the honest default.
        ``"random"``
            Uniform without replacement. The control for ``"uniform"``: if a
            result survives one and not the other, it depended on the spacing.
        ``"rate"``
            The ``n`` most active channels. Biased towards low frequencies in
            SHD, which is a property of speech, not of the method.
        ``"variance"``
            The ``n`` channels whose trial-averaged activity varies most over
            time — the channels that are modulated rather than merely loud.
    idx
        Explicit channel indices, bypassing ``method``. Use it to plug in a
        selector from :mod:`compbio2026.selection` (for example
        ``select_by_discriminability``), which needs labels and therefore must
        be fitted inside your cross-validation folds rather than here.

    Returns
    -------
    Xs : ``(n_trials, n, n_bins)``
    idx : ``(n,)`` the channel indices that were kept, sorted
    """
    n_ch = X.shape[1]
    if idx is not None:
        keep = np.sort(np.asarray(idx))
    elif method == "uniform":
        keep = np.unique(np.linspace(0, n_ch - 1, n).round().astype(int))
    elif method == "random":
        rng = rng or np.random.default_rng()
        keep = np.sort(rng.choice(n_ch, size=n, replace=False))
    elif method == "rate":
        keep = np.sort(np.argsort(X.mean(axis=(0, 2)))[::-1][:n])
    elif method == "variance":
        keep = np.sort(np.argsort(X.mean(axis=0).var(axis=-1))[::-1][:n])
    else:
        raise ValueError(f"unknown method {method!r}")
    return X[:, keep, :], keep


def resample_time(X: np.ndarray, n_steps: int) -> np.ndarray:
    """Linearly resample the time axis to ``n_steps`` samples.

    Used to compress a trial onto a simulation grid: an 800 ms recording can be
    presented to a simulated circuit over a much shorter window, which keeps the
    number of integration steps affordable.
    """
    src = np.linspace(0.0, 1.0, X.shape[-1])
    dst = np.linspace(0.0, 1.0, n_steps)
    out = np.empty(X.shape[:-1] + (n_steps,), dtype=np.float32)
    flat_in = X.reshape(-1, X.shape[-1])
    flat_out = out.reshape(-1, n_steps)
    for i, row in enumerate(flat_in):
        flat_out[i] = np.interp(dst, src, row)
    return out
