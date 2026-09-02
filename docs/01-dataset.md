# The dataset: Spiking Heidelberg Digits

## What it is

Audio recordings of spoken digits, pushed through a biophysical model of the inner ear and emitted as spike trains.

- **~700 channels**, one per position along a modelled basilar membrane
- **~10 000 trials** (8 156 in the training split)
- **20 classes**: digits 0–9 in English (labels 0–9) and German (labels 10–19)
- **12 speakers**, with speaker identity recorded per trial
- Trial durations vary from roughly 0.5 s to 1.4 s

Ground truth — which digit was said — is known for every trial. That is the property that makes this dataset useful for a two-week project: every structural claim you make is checkable against something.

## Where the channels come from

The conversion chain is [LAUSCHER](https://github.com/electronicvisions/lauscher):

```
audio ──► basilar membrane ──► hair cell ──► bushy cell ──► SHD
          (hydrodynamic)       (Meddis)      (LIF, 40:1)
```

1. **Basilar membrane.** A hydrodynamic model of a fluid-filled chamber with a stiffness gradient along its length. A travelling wave peaks at a position determined by frequency, so *frequency becomes place* and place becomes channel index. This is why channel number is not arbitrary: channels are **tonotopically ordered**, low index to high index maps monotonically onto frequency. Any analysis that permutes channels is throwing that away — sometimes deliberately, as a control.
2. **Hair cell.** The Meddis (1986, 1988) transmitter-pool model converts membrane velocity into a *firing probability*, with adaptation and saturation built in. Note it emits a continuous probability, not spikes.
3. **Bushy cell.** A leaky integrate-and-fire neuron per channel, driven by 40 independent Bernoulli draws of that channel's hair-cell probability. Coincidence detection across those 40 afferents sharpens phase locking. This is the layer SHD ships.

The biological counterpart of the bushy cell sits in the **cochlear nucleus** in the brainstem — the first synapse of the ascending auditory pathway. So SHD is not "the cochlea": it is one synapse past it.

## Loading it

```python
from compbio2026 import data

shd = data.load("train")
print(len(shd), "trials")
print(shd.words[:5])

X, y, t = data.build_design_matrix(shd, bin_ms=10.0, t_max_ms=800.0, smooth_ms=0.0)
# X: (trials, channels, bins)
```

`SHD` keeps the data in event form — a ragged list of spike times and channel indices per trial. `build_design_matrix` bins it onto a common time base, which is what every downstream analysis wants.

## The two choices that are not defaults

**`bin_ms` — how much spike timing you keep.** Cramer et al. showed classifiers with no access to spike timing plateau near **60 %** accuracy, while temporally aware ones reach **~85 %**. Your bin width places you somewhere on that axis. It is a scientific decision and it belongs in your report.

**`t_max_ms` — the common time base.** Trials have different durations, mostly because speakers do. Truncating loses the tail of long trials; zero-padding invents silence in short ones. Both distort the geometry. Look at the duration distribution before you pick a value, and check your result survives a different one.

A third choice, `smooth_ms`, trades temporal precision for a better-conditioned covariance. Everything in [03-dimensionality-reduction.md](03-dimensionality-reduction.md) depends on it.

## Splits that mean different things

```python
# random split — optimistic
data_dict = decoding.decode(X_flat, y)

# speaker held out — the honest generalisation test
data_dict = decoding.decode(X_flat, y, groups=shd.speaker)
```

Random cross-validation lets the classifier see other utterances by the same speaker. Holding whole speakers out asks whether you learned the digit or the voice. The gap between the two numbers is worth reporting on its own.

## Subsets worth considering

- **English only** (`shd.english_mask()`), 10 classes instead of 20. Cleaner story, faster to fit, and avoids conflating "which digit" with "which language".
- **A pair of confusable digits**, for a focused geometry analysis where you can actually see what is happening.

## References

- Cramer, B., Stradmann, Y., Schemmel, J. & Zenke, F. (2020). The Heidelberg spiking data sets for the systematic evaluation of spiking neural networks. *IEEE TNNLS.* [arXiv:1910.07407](https://arxiv.org/abs/1910.07407)
- Dataset: <https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/>
- Meddis, R. (1986). Simulation of mechanical to neural transduction in the auditory receptor. *JASA* 79(3):702.
- Fettiplace, R. & Hackney, C. M. (2006). The sensory and motor roles of auditory hair cells. *Nat. Rev. Neurosci.* 7:19–29.
