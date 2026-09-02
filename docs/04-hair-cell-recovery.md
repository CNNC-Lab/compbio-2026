# Recovering the hair-cell layer from SHD

SHD ships only the last stage of the conversion chain. This document explains how to get the stage above it back, how well that can work, and where it breaks.

## The structure that makes it possible

```
audio ──► basilar membrane ──► hair cell ──► bushy cell ──► SHD
                               (Meddis)      (LIF, 40 afferents)
                               continuous     spikes
                               probability
```

Two facts from the [LAUSCHER source](https://github.com/electronicvisions/lauscher) do the work:

1. **The hair-cell stage emits a firing *probability*, not spikes.** `HairCell` runs the Meddis transmitter-pool model and returns a `FiringProbability` — one continuous value per channel per audio sample.
2. **Hair-cell and bushy-cell channels are one-to-one.** `BushyCell(n_convergence=40)` draws 40 *independent Bernoulli samples of the same channel's probability* and feeds them to one LIF unit. The convergence number counts stochastic afferents within a channel; it is not a fan-in across channels.

So this is not a blind deconvolution across a mixing matrix. It is a per-channel problem: given the bushy-cell spikes of channel *i*, estimate the scalar probability signal `p_i(t)` that produced them.

Published bushy-cell parameters: `n_convergence=40`, `tau_mem=1 ms`, `tau_syn=0.5 ms`, `tau_refrac=1 ms`, `weight=13000/40`.

## The method

Since the forward map is a fixed, monotone, saturating function of `p`, measure it and invert it.

```python
from compbio2026 import data, lauscher, plotting

X, y, t = data.build_design_matrix(shd, bin_ms=2.0, smooth_ms=0.0)

curve = lauscher.transfer_curve()                    # simulate p -> BC rate; cache this
p_hat, valid = lauscher.estimate_hair_cell_rate(X, bin_ms=2.0, curve=curve, smooth_ms=5.0)
```

1. `transfer_curve()` drives the re-implemented `BushyCell` with constant probabilities across a grid and records the resulting firing rate. That is the forward map.
2. `estimate_hair_cell_rate()` smooths the observed spikes into an instantaneous rate and inverts the curve by monotone interpolation.
3. `valid` marks samples where the bushy cell is **saturated** and the inversion is not trustworthy.

Plot the curve before you trust anything: `plotting.transfer_curve(*curve)`.

## What this cannot do

Say all of this in the report — the limits are the interesting part.

**The refractory period caps the channel.** `tau_refrac = 1 ms` means a bushy cell cannot exceed ~1000 spikes/s. The transfer curve saturates, and above the knee very different `p` map to nearly the same rate. Those samples are **censored**, not confidently high. `valid` tells you which they are; report the fraction.

**You recover an output, not a state.** The Meddis model has internal variables — transmitter available (`q`), in the cleft (`c`), being reprocessed (`w`) — that a firing rate does not identify. Two different pool states can produce the same instantaneous probability.

**Smoothing sets the time resolution.** The bushy-cell stage exists partly to sharpen phase locking. Smoothing to estimate a rate throws that away. Any claim you make about fine temporal structure in `p̂` is really a claim about your kernel.

**Single trials are sparse.** Forty Bernoulli draws per timestep average out a lot, but one trial does not carry enough spikes to invert finely. Average across trials of the same digit first, and then your statement is about the class, not the utterance.

**The inverse is not unique in general.** It is well-posed here *only because* the channel mapping is 1:1 and the nonlinearity is monotone. If you change `n_convergence`, or model convergence across channels, that guarantee goes away.

## Validating it — do this before believing it

The estimate is an assertion until you check it against the truth. LAUSCHER can produce the true hair-cell layer from the source audio:

```python
stages = lauscher.run_lauscher("path/to/digit.wav")
p_true = stages["firing_probability"]
```

Compare `p_true` against `p̂` recovered from the bushy-cell spikes that LAUSCHER generated in the same run. That is a closed loop with no unknowns, and it gives you an honest error bar for the method before you apply it to SHD, where no ground truth exists.

The Heidelberg Digits **audio** is a separate download from the spike dataset. Without it, you can still validate synthetically: pick an arbitrary smooth `p(t)`, run `BushyCell.simulate` on it, and recover it.

## Why bother

Three reasons, in increasing order of interest:

1. **It makes the pipeline concrete.** Students see that "the dataset" is the output of a model with stages, not a given.
2. **It is a real inverse problem** with a real identifiability limit, at a scale that fits on a laptop.
3. **It opens route C.** With the hair-cell layer recovered you can ask how the representation changes *between* stages — does the bushy-cell layer separate the digits better than the hair-cell layer it came from, or does coincidence detection discard information that a decoder could have used? Run the whole Stage 1–3 analysis on both layers and compare. That is a genuinely open question and a good project.

## References

- Cramer, B. et al. (2020). *IEEE TNNLS.* [arXiv:1910.07407](https://arxiv.org/abs/1910.07407) — the chain and its parameters.
- Meddis, R. (1986). *JASA* 79(3):702; Meddis, R. (1988). *JASA* 83(3):1056 — the hair-cell model.
- LAUSCHER source: <https://github.com/electronicvisions/lauscher>
- Fettiplace, R. & Hackney, C. M. (2006). *Nat. Rev. Neurosci.* 7:19–29 — what hair cells actually do.
