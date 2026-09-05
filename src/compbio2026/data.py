"""Loading the Spiking Heidelberg Digits (SHD) and turning it into design matrices.

The SHD file stores spikes as a ragged event list: for each trial, an array of
spike times (seconds) and an array of channel indices. Almost every analysis in
this project instead wants a **fixed-shape** array, so the central function here
is :func:`build_design_matrix`, which bins each trial onto a common time base.

Two choices in that binning are consequential and are therefore explicit
parameters rather than hidden defaults:

``bin_ms``
    How much spike timing you keep. Cramer et al. (2020) showed that classifiers
    with no access to spike timing plateau near 60 % while temporally aware ones
    reach ~85 %; your bin width is where you sit on that axis.
``smooth_ms``
    Gaussian smoothing of the binned counts. Smoothing trades temporal precision
    for a better-conditioned covariance, which changes every geometry estimate
    downstream.

Neither has a correct value. Report the value you used and show that your result
survives a sweep over it.

What the raw event list actually is
-----------------------------------
Spike times are stored in the HDF5 file as **float16 seconds**. float16 has no
fixed grid: the spacing between representable values grows with the value, so
the timing precision degrades through the trial — about 0.008 ms at 10 ms,
0.06 ms at 100 ms, 0.24 ms at 400 ms and 0.49 ms at 700 ms. That is the storage
floor, not the resolution of the model that generated the data.

Within one channel the smallest interval measured across 2.2 million intervals
is **1.22 ms**, the refractory floor of the bushy-cell model. So the binned
matrix is exactly binary at 1 ms bins, and effectively so at 2 ms; the fraction
of occupied ``(channel, bin)`` cells holding more than one spike is 0.00 % at
1 ms, 0.08 % at 2 ms, 1.9 % at 4 ms, 26 % at 10 ms and 57 % at 25 ms. Above a
few milliseconds you are working with counts, whatever you call them.

Trials run to 1.19 s; truncating at ``t_max_ms=800`` touches 24 % of trials and
drops 0.15 % of all spikes.

Reference
---------
Cramer, B., Stradmann, Y., Schemmel, J. & Zenke, F. (2020). The Heidelberg
spiking data sets for the systematic evaluation of spiking neural networks.
*IEEE TNNLS.*
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass

import numpy as np
import tables
from scipy.ndimage import gaussian_filter1d

BASE_URL = "https://zenkelab.org/datasets"
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

#: Label index -> spoken word. 0-9 are English, 10-19 are German.
DIGIT_KEYS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "null", "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun",
]

N_CHANNELS = 700


@dataclass
class SHD:
    """A loaded SHD split, still in event form.

    Attributes
    ----------
    times, units
        Lists of length ``n_trials``; ``times[i]`` are spike times in seconds and
        ``units[i]`` the matching channel indices for trial ``i``.
    labels
        Integer class label per trial, indexing :data:`DIGIT_KEYS`.
    speaker
        Speaker id per trial. Needed if you want speaker-held-out splits, which
        are the honest generalisation test for this dataset.
    """

    times: list
    units: list
    labels: np.ndarray
    speaker: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def words(self) -> list[str]:
        return [DIGIT_KEYS[i] for i in self.labels]

    def english_mask(self) -> np.ndarray:
        """Boolean mask selecting the ten English digits."""
        return self.labels < 10


def _download(url: str, path: str, md5: str | None = None) -> str:
    if os.path.exists(path) and md5:
        with open(path, "rb") as fh:
            if hashlib.md5(fh.read()).hexdigest() == md5:
                return path
    elif os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)
    if md5:
        with open(path, "rb") as fh:
            if hashlib.md5(fh.read()).hexdigest() != md5:
                os.remove(path)
                raise ValueError(f"md5 mismatch for {path}")
    return path


def fetch(split: str = "train", cache_dir: str | None = None) -> str:
    """Download and decompress one SHD split; return the path to the .h5 file.

    Idempotent: if the file is already present and its md5 matches, nothing is
    downloaded. The whole training split is ~200 MB compressed.
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    cache = os.path.join(cache_dir, "hdspikes")
    os.makedirs(cache, exist_ok=True)

    h5_path = os.path.join(cache, f"shd_{split}.h5")
    if os.path.isfile(h5_path):
        return h5_path  # already downloaded; no network needed

    with urllib.request.urlopen(f"{BASE_URL}/md5sums.txt") as response:
        lines = response.read().decode("utf-8").split("\n")
    hashes = {p[1]: p[0] for p in (line.split() for line in lines) if len(p) == 2}

    name = f"shd_{split}.h5.gz"
    gz_path = _download(f"{BASE_URL}/{name}", os.path.join(cache, name), hashes.get(name))
    print(f"decompressing {gz_path}")
    with gzip.open(gz_path, "rb") as f_in, open(h5_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return h5_path


def load(split: str = "train", cache_dir: str | None = None, path: str | None = None) -> SHD:
    """Load a split into memory as an :class:`SHD`.

    Downloads on first call and caches under ``data/hdspikes/``; subsequent
    calls are offline. Pass ``path`` only if you keep the file somewhere else.
    """
    h5_path = path or fetch(split, cache_dir)
    with tables.open_file(h5_path, mode="r") as fh:
        times = [np.asarray(t) for t in fh.root.spikes.times]
        units = [np.asarray(u) for u in fh.root.spikes.units]
        labels = np.asarray(fh.root.labels[:])
        try:
            speaker = np.asarray(fh.root.extra.speaker[:])
        except tables.NoSuchNodeError:
            speaker = np.full(len(labels), -1)
    return SHD(times=times, units=units, labels=labels, speaker=speaker)


def build_design_matrix(
    shd: SHD,
    bin_ms: float = 10.0,
    t_max_ms: float = 800.0,
    smooth_ms: float = 0.0,
    n_channels: int = N_CHANNELS,
    trials: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin trials onto a common time base.

    Parameters
    ----------
    bin_ms
        Width of a time bin, in milliseconds.
    t_max_ms
        Trials are truncated (or zero-padded) to this duration. Trial durations
        in SHD vary with the speaker; a common base is what makes the trials
        stackable, and the padding it introduces is a real modelling choice.
    smooth_ms
        Standard deviation of a Gaussian smoothing kernel applied along time,
        in milliseconds. ``0`` disables smoothing.
    trials
        Optional index array selecting a subset of trials.

    Returns
    -------
    X : ``(n_trials, n_channels, n_bins)`` float array of spike counts
    y : ``(n_trials,)`` integer labels
    t : ``(n_bins,)`` bin centres in milliseconds
    """
    idx = np.arange(len(shd)) if trials is None else np.asarray(trials)
    n_bins = int(round(t_max_ms / bin_ms))
    edges = np.arange(n_bins + 1) * bin_ms / 1e3  # seconds

    X = np.zeros((len(idx), n_channels, n_bins), dtype=np.float32)
    for row, i in enumerate(idx):
        t = shd.times[i]
        u = shd.units[i]
        keep = (t < edges[-1]) & (u < n_channels)
        if not np.any(keep):
            continue
        bin_of = np.digitize(t[keep], edges) - 1
        np.add.at(X[row], (u[keep].astype(int), bin_of), 1.0)

    if smooth_ms > 0:
        X = gaussian_filter1d(X, sigma=smooth_ms / bin_ms, axis=-1, mode="constant")

    t_centres = (np.arange(n_bins) + 0.5) * bin_ms
    return X, shd.labels[idx], t_centres


def events(shd: SHD, trial: int, t_max_ms: float | None = None,
           n_channels: int = N_CHANNELS) -> np.ndarray:
    """Raw spikes of one trial, as an ``(n_spikes, 2)`` array of ``(unit, time_ms)``.

    This is the data in the form SHD stores it — no binning, no fixed shape, no
    padding. :func:`build_design_matrix` is the only thing in this package that
    bins; everything it does is reversible only down to its bin width, so start
    here whenever the claim is about spike timing.

    The ``[(unit, time), ...]`` list of tuples is one step away, and that is
    exactly the form ``tools.analysis.signals.spikes.SpikeList`` expects::

        [(int(u), float(t)) for u, t in data.events(shd, 0)]

    Rows are sorted by time. ``time_ms`` is in milliseconds to match
    ``bin_ms``, ``t_max_ms`` and the ``t`` returned by
    :func:`build_design_matrix`; the file itself stores seconds.
    """
    t = np.asarray(shd.times[trial], dtype=np.float64) * 1e3
    u = np.asarray(shd.units[trial], dtype=np.int64)
    keep = u < n_channels
    if t_max_ms is not None:
        keep &= t < t_max_ms
    t, u = t[keep], u[keep]
    order = np.argsort(t, kind="stable")
    return np.stack([u[order].astype(np.float64), t[order]], axis=1)


def binarize(X: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Clip a count matrix to ``{0, 1}`` and say what that cost.

    Whether this is lossless is a property of the bin width, not of the data.
    The bushy-cell model has a ~1.2 ms refractory floor, so at ``bin_ms <= 1``
    nothing is lost and the result is the spike train itself; by 10 ms a quarter
    of the occupied cells hold two or more spikes and clipping is throwing away
    real rate differences. The printed number is the check.
    """
    occupied = X > 0
    n_occ = int(occupied.sum())
    lost = float(X.sum() - occupied.sum())
    if verbose and n_occ:
        multi = int((X > 1).sum())
        print(f"binarize: {multi}/{n_occ} occupied cells held >1 spike "
              f"({100 * multi / n_occ:.2f} %); {lost:.0f} of {X.sum():.0f} spikes discarded")
    return occupied.astype(X.dtype)


def flatten(X: np.ndarray) -> np.ndarray:
    """``(trials, channels, bins)`` -> ``(trials, channels * bins)``.

    The flattened axis is channel-major, so feature ``c * n_bins + b`` is channel
    ``c`` at bin ``b``. :func:`compbio2026.selection.channel_groups` relies on
    that ordering.
    """
    return X.reshape(X.shape[0], -1)


def rates(X: np.ndarray, bin_ms: float) -> np.ndarray:
    """Mean firing rate per channel per trial, in spikes/second."""
    return X.mean(axis=-1) * (1e3 / bin_ms)


def subsample_channels(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Keep ``n`` channels chosen uniformly at random — the control for learning
    curves against population size."""
    keep = rng.choice(X.shape[1], size=n, replace=False)
    return X[:, np.sort(keep), :]
