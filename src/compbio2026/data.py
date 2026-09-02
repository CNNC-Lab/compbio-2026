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

    with urllib.request.urlopen(f"{BASE_URL}/md5sums.txt") as response:
        lines = response.read().decode("utf-8").split("\n")
    hashes = {p[1]: p[0] for p in (line.split() for line in lines) if len(p) == 2}

    name = f"shd_{split}.h5.gz"
    gz_path = _download(f"{BASE_URL}/{name}", os.path.join(cache, name), hashes.get(name))
    h5_path = gz_path[:-3]
    if not os.path.isfile(h5_path):
        print(f"decompressing {gz_path}")
        with gzip.open(gz_path, "rb") as f_in, open(h5_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return h5_path


def load(split: str = "train", cache_dir: str | None = None, path: str | None = None) -> SHD:
    """Load a split into memory as an :class:`SHD`.

    Pass ``path`` to use a file you already have (for instance the copy in the
    2025 course repository) instead of downloading.
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
