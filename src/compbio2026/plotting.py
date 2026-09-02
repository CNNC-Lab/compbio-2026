"""House style and the standard figures.

Import :func:`apply_style` once at the top of a notebook. Every plotting helper
returns the ``Axes`` it drew on so you can keep customising.

Two conventions worth keeping:

* **Plot the control on the same axes as the result.** Chance level and the
  shuffled-label distribution belong on the accuracy figure, not in the text.
* **Save vector.** ``savefig(name)`` writes an SVG next to a PNG, both with a
  transparent background, so a figure can drop into a slide or a report without
  a white box around it.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

INK = "#0A0A0A"
INK_SOFT = "#3B3B3D"
INK_MUTED = "#6B6B6B"
HAIRLINE = "#E5E2DC"
ACCENT = "#7A2A3D"

#: Qualitative palette, colour-blind safe, ordered by how distinguishable the
#: first few are from each other.
PALETTE = ["#7A2A3D", "#2B6E8F", "#C08A2E", "#4B7F52", "#6B5B95", "#B5651D",
           "#3F7F7F", "#8C6A4E", "#A03C55", "#57606C"]


def apply_style() -> None:
    """Set the rcParams used throughout the project."""
    mpl.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "axes.edgecolor": HAIRLINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK_SOFT,
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "figure.dpi": 110,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def savefig(fig, name: str, outdir: str = "plots") -> None:
    """Write ``name.svg`` and ``name.png``, both transparent."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", transparent=True, dpi=300)


def raster(X3, trial: int, ax=None, bin_ms: float = 10.0, **kw):
    """Raster of one trial from a binned ``(trials, channels, bins)`` array."""
    ax = ax or plt.gca()
    ch, b = np.nonzero(X3[trial] > 0)
    ax.scatter(b * bin_ms, ch, s=kw.pop("s", 0.7), c=kw.pop("c", INK), marker=".",
               linewidths=0, alpha=kw.pop("alpha", 0.85), **kw)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cochlear channel")
    return ax


def embedding(Z, y, labels=None, ax=None, **kw):
    """Scatter of a 2-D embedding coloured by class."""
    ax = ax or plt.gca()
    for i, c in enumerate(np.unique(y)):
        m = y == c
        name = labels[c] if labels is not None else str(c)
        ax.scatter(Z[m, 0], Z[m, 1], s=kw.get("s", 12), alpha=kw.get("alpha", 0.75),
                   color=PALETTE[i % len(PALETTE)], label=name, linewidths=0)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    return ax


def learning_curve(x, mean, std, ax=None, chance: float | None = None, label: str | None = None, xlabel: str = ""):
    """Mean ± sd against ``x``, with chance drawn in if given."""
    ax = ax or plt.gca()
    mean, std = np.asarray(mean), np.asarray(std)
    ax.plot(x, mean, marker="o", ms=4, label=label, color=ACCENT)
    ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=ACCENT, linewidth=0)
    if chance is not None:
        ax.axhline(chance, ls="--", lw=1, color=INK_MUTED)
        ax.text(x[0], chance, " chance", va="bottom", ha="left", fontsize=8, color=INK_MUTED)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    return ax


def accuracy_over_time(t_ms, acc, ax=None, chance: float | None = None):
    """Decoding accuracy as the trial unfolds."""
    ax = ax or plt.gca()
    ax.plot(t_ms, acc, color=ACCENT)
    if chance is not None:
        ax.axhline(chance, ls="--", lw=1, color=INK_MUTED)
    ax.set_xlabel("Time from trial onset (ms)")
    ax.set_ylabel("Accuracy")
    return ax


def ablation_bars(results: dict, ax=None):
    """Full / kept-only / ablated accuracy per selector, chance marked."""
    ax = ax or plt.gca()
    names = list(results)
    width = 0.26
    x = np.arange(len(names))
    for i, key in enumerate(("full", "kept", "ablated")):
        ax.bar(x + (i - 1) * width, [results[n][key] for n in names], width,
               label=key, color=PALETTE[i], linewidth=0)
    chance = results[names[0]]["chance"]
    ax.axhline(chance, ls="--", lw=1, color=INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.legend()
    return ax


def transfer_curve(probs, rates, ax=None, tau_refrac: float = 1e-3):
    """The bushy-cell forward map, with the refractory ceiling marked."""
    ax = ax or plt.gca()
    ax.plot(probs, rates, marker="o", ms=3, color=ACCENT)
    ax.axhline(1.0 / tau_refrac, ls="--", lw=1, color=INK_MUTED)
    ax.text(probs[1] if len(probs) > 1 else 0, 1.0 / tau_refrac, " refractory ceiling",
            va="bottom", fontsize=8, color=INK_MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Hair-cell firing probability per sample")
    ax.set_ylabel("Bushy-cell rate (spikes/s)")
    return ax
