"""Recovering the hair-cell layer that lies underneath the SHD bushy-cell spikes.

Why this is possible
--------------------
SHD ships only the **last** stage of the Lauscher conversion chain: 700 channels
of bushy-cell (BC) spikes. The stages above it are

    audio -> basilar membrane -> hair cell -> bushy cell

and in Lauscher the hair-cell (HC) stage does **not** emit spikes. It emits a
continuous *firing probability* per channel per sample, from the Meddis
transmitter-pool model. The bushy cell is then a leaky integrate-and-fire neuron
driven by ``n_convergence = 40`` independent Bernoulli draws of that same
channel's probability.

The important structural fact: **HC and BC channels are one-to-one.** The
convergence number counts stochastic afferent draws from one channel, not a fan-
in across channels. So recovering the HC layer is a per-channel problem: given
BC spikes for channel *i*, estimate the scalar probability signal ``p_i(t)`` that
produced them. That is a rate estimation followed by the inversion of a static,
monotone, saturating transfer function — not a blind deconvolution.

How this module does it
-----------------------
1. :class:`BushyCell` re-implements Lauscher's BC stage in plain NumPy, with the
   published parameters.
2. :func:`transfer_curve` drives it with constant probabilities across a grid and
   measures the resulting BC firing rate. That is the forward map ``p -> rate``.
3. :func:`estimate_hair_cell_rate` smooths the observed BC spikes into an
   instantaneous rate and inverts the curve to get ``p̂(t)``.

What it cannot do
-----------------
Be honest about this in the report — it is the interesting part of the method:

* **Refractoriness caps the channel.** ``tau_refrac = 1 ms`` means a BC cannot
  exceed ~1000 spikes/s, so the transfer curve saturates. Above the knee, very
  different ``p`` map to nearly the same rate and the inversion is ill-posed.
  :func:`estimate_hair_cell_rate` returns a validity mask for this.
* **The estimate is a rate, not the transmitter-pool state.** Meddis has internal
  variables (``q`` available, ``c`` cleft, ``w`` reprocessing) that a rate does
  not identify. You recover the model's *output*, not its state.
* **Smoothing sets the time resolution.** The BC stage exists to sharpen phase
  locking; smoothing to estimate a rate throws that away. Any claim about fine
  temporal structure in ``p̂`` is a claim about your kernel.
* **It is stochastic in one direction only.** 40 Bernoulli draws per timestep
  average out a lot; a single trial does not carry enough spikes to invert
  finely. Average over trials of the same digit, or accept a coarse kernel.

Validating it
-------------
:func:`run_lauscher` runs the real package on source audio and returns every
intermediate stage, so you can compare ``p̂`` against the true ``p``. This is the
control that turns the inversion from an assertion into a measurement. It needs
the Heidelberg Digits audio, which is a separate download from the spikes.

References
----------
Cramer et al. (2020), *IEEE TNNLS* — the SHD datasets and the conversion chain.
Meddis, R. (1986, 1988), *JASA* 79(3):702 and 83(3):1056 — the hair-cell model.
Lauscher: https://github.com/electronicvisions/lauscher
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

#: Lauscher's published bushy-cell parameters (``lauscher/transformations/bushy_cell.py``).
BC_DEFAULTS = dict(n_convergence=40, tau_mem=1e-3, tau_syn=5e-4, tau_refrac=1e-3, weight=13e3)


@dataclass
class BushyCell:
    """Lauscher's bushy-cell stage, re-implemented in NumPy.

    A leaky integrate-and-fire unit with exponential synapses, driven by
    ``n_convergence`` Bernoulli spike trains sampled from the hair-cell firing
    probability of the same channel. ``weight`` is divided by the convergence,
    exactly as in the reference implementation, so the total drive is
    independent of how many afferents you assume.
    """

    n_convergence: int = 40
    tau_mem: float = 1e-3
    tau_syn: float = 5e-4
    tau_refrac: float = 1e-3
    weight: float = 13e3

    def _sample_afferents(self, p: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
        """Bernoulli draws with an absolute refractory period, per afferent."""
        spikes = rng.random((p.size, self.n_convergence)) < p[:, None]
        refrac = int(round(self.tau_refrac * fs))
        if refrac > 1:
            for a in range(self.n_convergence):
                last = -refrac
                col = spikes[:, a]
                for j in np.flatnonzero(col):
                    if j - last < refrac:
                        col[j] = False
                    else:
                        last = j
        return spikes

    def simulate(self, p: np.ndarray, fs: float, seed: int = 0) -> np.ndarray:
        """Run one channel. ``p`` is the HC firing probability per sample.

        Returns a binary spike vector of the same length.
        """
        rng = np.random.default_rng(seed)
        stim = self._sample_afferents(np.asarray(p, dtype=float), fs, rng).astype(float)
        dt = 1.0 / fs
        w = self.weight / float(self.n_convergence)
        n = stim.shape[0]

        vm = 0.0
        isyn = 0.0
        out = np.zeros(n)
        refrac_counter = 0
        n_refrac = self.tau_refrac * fs
        decay_mem = np.exp(-dt / self.tau_mem)
        decay_syn = np.exp(-dt / self.tau_syn)

        for step in range(n):
            isyn = isyn * decay_syn + stim[step].sum()
            if refrac_counter <= 0:
                vm = vm * decay_mem + isyn * w * dt
                if vm >= 1.0:
                    out[step] = 1.0
                    vm = 0.0
                    refrac_counter = n_refrac
            refrac_counter -= 1
        return out


def transfer_curve(
    probs: np.ndarray | None = None,
    fs: float = 20_000.0,
    duration: float = 0.25,
    n_repeats: int = 3,
    seed: int = 0,
    **bc_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the forward map: constant HC probability -> BC firing rate.

    Parameters
    ----------
    probs
        Grid of hair-cell firing probabilities per sample. Defaults to a
        log-spaced grid, because the interesting behaviour is at the low end.
    fs
        Sample rate of the simulation. Lauscher runs at the audio rate.
    duration, n_repeats
        Longer and more repeats give a smoother curve; the default is a few
        seconds of simulated time per grid point.

    Returns
    -------
    probs, rates
        ``rates`` in spikes/second. Monotone increasing and saturating near
        ``1 / tau_refrac``.
    """
    if probs is None:
        probs = np.concatenate([[0.0], np.logspace(-4, np.log10(0.5), 24)])
    bc = BushyCell(**{**BC_DEFAULTS, **bc_kwargs})
    n = int(round(duration * fs))
    rates = np.zeros(len(probs))
    for i, p in enumerate(probs):
        acc = []
        for r in range(n_repeats):
            spikes = bc.simulate(np.full(n, p), fs, seed=seed + 1000 * i + r)
            acc.append(spikes.sum() / duration)
        rates[i] = float(np.mean(acc))
    return np.asarray(probs), rates


def invert_transfer(rates_obs: np.ndarray, probs: np.ndarray, rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert a measured transfer curve by monotone interpolation.

    Returns the estimated probabilities and a boolean mask marking samples that
    fall in the **saturated** part of the curve, where the inversion is not
    trustworthy. Treat masked samples as censored, not as high-confidence highs.
    """
    order = np.argsort(rates)
    r_sorted, p_sorted = rates[order], probs[order]
    # Strictly increasing region only; ties mean saturation.
    keep = np.concatenate([[True], np.diff(r_sorted) > 1e-9])
    r_mono, p_mono = r_sorted[keep], p_sorted[keep]

    p_hat = np.interp(rates_obs, r_mono, p_mono)
    knee = r_mono[-1] * 0.95
    valid = rates_obs < knee
    return p_hat, valid


def estimate_hair_cell_rate(
    X: np.ndarray,
    bin_ms: float,
    curve: tuple[np.ndarray, np.ndarray] | None = None,
    smooth_ms: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the hair-cell firing probability behind binned bushy-cell spikes.

    Parameters
    ----------
    X
        ``(n_trials, n_channels, n_bins)`` spike counts, as produced by
        :func:`compbio2026.data.build_design_matrix`.
    bin_ms
        Bin width used to build ``X``, in milliseconds.
    curve
        A ``(probs, rates)`` pair from :func:`transfer_curve`. Computed on demand
        if omitted, which takes a minute or two — cache it.
    smooth_ms
        Gaussian kernel for the instantaneous-rate estimate. Larger is more
        stable and less temporally precise; this parameter *is* the time
        resolution of the result.

    Returns
    -------
    p_hat : ``(n_trials, n_channels, n_bins)`` estimated firing probability
    valid : boolean array, ``False`` where the BC rate is in saturation

    Notes
    -----
    Single trials are sparse. If the estimate looks like noise, average ``X``
    over trials of the same digit first and estimate from the average — the
    result is then a statement about the class, not about one utterance.
    """
    if curve is None:
        curve = transfer_curve()
    probs, rates = curve

    rate_obs = gaussian_filter1d(X, sigma=max(smooth_ms / bin_ms, 1e-6), axis=-1, mode="constant")
    rate_obs = rate_obs * (1e3 / bin_ms)  # counts per bin -> spikes/second

    flat = rate_obs.ravel()
    p_flat, valid_flat = invert_transfer(flat, probs, rates)
    return p_flat.reshape(X.shape), valid_flat.reshape(X.shape)


def run_lauscher(wav_path: str, num_channels: int = 700):
    """Run the real Lauscher chain on an audio file and return every stage.

    This is the ground truth for validating :func:`estimate_hair_cell_rate`. It
    needs the ``lauscher`` package (``pip install
    git+https://github.com/electronicvisions/lauscher``) and the Heidelberg
    Digits **audio**, which is a separate download from the spike dataset.

    Returns
    -------
    dict with keys ``membrane_velocity``, ``firing_probability`` (the hair-cell
    output) and ``spikes`` (the bushy-cell output).
    """
    try:
        from lauscher.audiowaves import FileMonoAudioWave
        from lauscher.transformations.basilar_membrane import BasilarMembrane
        from lauscher.transformations.bushy_cell import BushyCell as LauscherBushyCell
        from lauscher.transformations.hair_cell import HairCell
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The `lauscher` package is not installed. Install it with:\n"
            "    pip install git+https://github.com/electronicvisions/lauscher"
        ) from exc

    wave = FileMonoAudioWave(wav_path)
    membrane = wave.transform(BasilarMembrane(num_channels=num_channels))
    firing_probability = membrane.transform(HairCell())
    spikes = firing_probability.transform(LauscherBushyCell())
    return {
        "membrane_velocity": membrane,
        "firing_probability": firing_probability,
        "spikes": spikes,
    }
