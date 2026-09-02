# CompBio 2026 — ML approaches for decoding and interpreting neural population activity

Project repository for the [Summer School in Computational Biology 2026](https://www.uc.pt/en/events/compbio2026/), University of Coimbra.

**Tutor:** Renato Duarte — CNC, CiBB, University of Coimbra · renato.duarte@cnc.uc.pt · (+351) 913 090 735
**Team size:** up to 4, 5 at a push
**Prerequisites:** basic Python (numpy, matplotlib, scikit-learn). No neuroscience background required.
**Compute:** laptop only — no cluster, no GPU, no restricted data.

---

## The problem

Neural population recordings are high-dimensional, redundant, and rarely interpretable at face value. The working hypothesis of modern systems neuroscience is that computation lives in a small number of latent modes shared across many neurons, rather than in individual cells. That hypothesis licenses almost all population-level analysis, and it is usually assumed rather than tested.

Three things follow, and they are the substance of this project:

1. **Structure is method-dependent.** Different dimensionality-reduction methods impose different assumptions — linearity, noise model, metric preservation, whether task parameters are used. Whether the class structure survives all of them is an empirical question.
2. **Visible structure is not usable structure.** A separation visible in a 3-D embedding may be an artefact of the projection. Decoding accuracy is a common, defensible currency.
3. **Decodability does not localise.** Knowing information is present says nothing about what carries it, and in a redundant code the obvious answer — the most strongly tuned units — is frequently wrong.

The intended payoff is the relationship between the three: **does any geometric property of the representation predict what a downstream linear reader can extract, and can that capacity be traced to an identifiable subspace of the population?**

## The data

**Spiking Heidelberg Digits (SHD)** — spoken digits converted to spike trains by a biophysically grounded model of the inner ear: basilar membrane → hair cells → bushy cells.

- ~700 channels, ~10 000 trials, 20 classes (digits 0–9 in English and German), 12 speakers
- Ground truth is known for every trial, so every structural claim is checkable
- Public, small, and it loads locally

See [docs/01-dataset.md](docs/01-dataset.md).

## Quick start

```shell
git clone https://github.com/CNNC-Lab/compbio-2026.git
cd compbio-2026
conda env create -f environment.yml
conda activate compbio-2026
jupyter lab notebooks/01_data_and_statistics.ipynb
```

```python
from compbio2026 import data, geometry, decoding, selection, plotting

plotting.apply_style()
shd = data.load("train")                                  # downloads once, then cached
X, y, t = data.build_design_matrix(shd, bin_ms=10.0)      # (trials, channels, bins)

geometry.summarize(data.flatten(X), y)                    # participation ratio, class separation, ...
decoding.decode(data.flatten(X), y)                       # accuracy with chance level
selection.compare_selectors(X, y, k=50)                   # sparse group lasso vs. heuristics, ablated
```

Full setup, including what to do when something fails: [docs/00-setup.md](docs/00-setup.md).

## The three stages

| Stage | Question | Notebook |
|---|---|---|
| **1 — Structure** | What does this population look like, and does its geometry survive every method? | [`01`](notebooks/01_data_and_statistics.ipynb), [`02`](notebooks/02_dimensionality_reduction.ipynb) |
| **2 — Decodability** | How much of that structure can a linear reader use? | [`03`](notebooks/03_decoding_baseline.ipynb) |
| **3 — Interpretability** | Which subspace carries the digit, and is it necessary? | [`04`](notebooks/04_interpretability.ipynb) |

Details and what counts as doing each stage well: [docs/02-stages.md](docs/02-stages.md).

Stages are a spine, not a recipe. **You design the plan and the experiments.**

## Three routes

- **A — analyse this population directly.** Clean, labelled, checkable. Reliably finishes.
- **B — use it as input.** Drive a biophysical circuit and analyse the *second* layer. What does recurrence add to a representation that already carries the signal?
- **C — decompose across layers.** Recover the hair-cell layer underneath SHD (see below), or implement one stage above the bushy cell and ask what it adds.

Trade-offs and risks: [docs/05-routes.md](docs/05-routes.md).

## Recovering the hair-cell layer

SHD ships only the last stage of the conversion chain. The stage above it — the Meddis transmitter-pool hair-cell model — emits a continuous firing probability, and LAUSCHER's bushy cells map **one-to-one** onto hair-cell channels (40 stochastic afferents *within* a channel, not a fan-in across channels).

That makes recovery a well-posed per-channel inverse problem: measure the forward map from firing probability to bushy-cell rate, then invert it.

```python
from compbio2026 import lauscher
curve = lauscher.transfer_curve()
p_hat, valid = lauscher.estimate_hair_cell_rate(X, bin_ms=2.0, curve=curve)
```

It works, with real limits — a refractory ceiling that censors high rates, a smoothing kernel that sets the time resolution, and a rate that identifies the model's output but not its internal state. All of that, plus the validation loop against genuine LAUSCHER output, is in [docs/04-hair-cell-recovery.md](docs/04-hair-cell-recovery.md).

## Layout

```
src/compbio2026/     data · geometry · decoding · selection · lauscher · plotting
notebooks/           the four stage notebooks
docs/                setup, dataset, stages, methods, hair-cell recovery, routes, reading
tools/               SpikeList / StateMatrix and analysis utilities, carried over from 2025
data/                downloaded datasets (gitignored)
plots/               figure output (gitignored)
```

`tools/` comes from the [2025 course repository](https://github.com/CNNC-Lab/computational-biology-2025). It is not required by anything in `src/`, but `SpikeList` and `StateMatrix` implement a lot of useful spike-train machinery and are worth reading.

## Deliverables

- A short written report built around **one defensible result**, with methods, controls, and stated limitations
- Final presentation, Saturday 12 September
- Reproducible analysis code in this repository

**A negative or null result, properly established and controlled, counts fully.**

## One framing that is not optional

Selection and ablation in Stage 3 establish causal relevance **for the decoder, not for the circuit**. The data are fixed recordings and nothing is intervened upon in the biological system. Distinguishing decoder-causality from circuit-causality is a live confusion in the interpretability literature, and getting it right is part of the assessment.

## Schedule

| Date | |
|---|---|
| Wed 2 Sep, 17:00 | Project pitch; Pizza & Games 19:30, Casa das Artes |
| Thu 3 Sep, 14:00 | Student distribution, room F42 — work begins |
| Fri 4 – Tue 8 Sep, 17:00–18:00 | Daily project blocks |
| Wed 9 Sep, 10:00–13:00 & 15:00–17:00 | Computational Neuroscience / NeuroAI teaching block |
| Wed 9 Sep, 17:00 | Beer & poster session, Physics Department atrium |
| Thu 10 Sep, 16:00–18:00 | Progress reports |
| Sat 12 Sep, 15:00 | Final presentations |

The final three days are hosted at **CNC-UC, Polo I** (Rua Larga, Faculty of Medicine building). Laptop only.

## Reading

[docs/06-reading.md](docs/06-reading.md) — three starred items to start with, the rest on demand.

## Acknowledgements

Built on the [2025 course repository](https://github.com/CNNC-Lab/computational-biology-2025). Dataset by Cramer, Stradmann, Schemmel & Zenke (2020); conversion chain by [LAUSCHER](https://github.com/electronicvisions/lauscher).
