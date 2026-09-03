# Algorithms for finding structure in neural data

A catalogue, not a recommendation list. Every entry has working code and a paper; every link below was checked against Crossref or the arXiv API on 2026-09-03.

**Laptop** says whether it is realistic inside a ten-day project on the SHD dataset: ✅ minutes · ⚠️ hours, needs care · ❌ read about it, do not build on it.

---

## 1 · Start here

The methods you should have tried before reaching for anything else. If these already answer your question, stop.

| Method | What it gives you | Code | Paper | Laptop |
|---|---|---|---|---|
| **PCA** | Directions of maximum variance; effective dimensionality | [`sklearn.decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | — | ✅ |
| **Factor analysis** | Shared variance, separated from per-neuron private noise | [`sklearn.decomposition.FactorAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html) | — | ✅ |
| **Isomap / LLE / spectral** | Nonlinear embedding preserving geodesics or neighbourhoods | [`sklearn.manifold`](https://scikit-learn.org/stable/modules/manifold.html) | — | ✅ |
| **t-SNE** | Local neighbourhood structure, for *looking* only | [`sklearn.manifold.TSNE`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) | van der Maaten & Hinton 2008 | ✅ |
| **UMAP** | Like t-SNE, retains a little more global structure | [lmcinnes/umap](https://github.com/lmcinnes/umap) | [arXiv:1802.03426](https://arxiv.org/abs/1802.03426) | ✅ |
| **PHATE** | Embedding built for *continuous transitions* rather than clusters | [KrishnaswamyLab/PHATE](https://github.com/KrishnaswamyLab/PHATE) | [Moon et al. 2019, *Nat. Biotechnol.* 37:1482](https://doi.org/10.1038/s41587-019-0336-3) | ✅ |

> ⚠️ **t-SNE and UMAP:** distances *between* clusters are not meaningful. Use them to look; quantify somewhere else.

## 2 · Built for neural population data

Classics of the field. Each assumes something specific about your experiment — check the assumption holds before using it.

| Method | What it gives you | Code | Paper | Laptop |
|---|---|---|---|---|
| **dPCA** | Components *demixed* by task variable (stimulus, time, decision) | [machenslab/dPCA](https://github.com/machenslab/dPCA) | [Kobak et al. 2016, *eLife* 5:e10989](https://doi.org/10.7554/eLife.10989) | ✅ |
| **GPFA** | Smooth single-trial latent trajectories | [`elephant.gpfa`](https://elephant.readthedocs.io/en/latest/reference/gpfa.html) | [Yu et al. 2009, *J. Neurophysiol.* 102:614](https://doi.org/10.1152/jn.90941.2008) | ⚠️ |
| **jPCA** | Rotational structure in population dynamics | [bantin/jPCA](https://github.com/bantin/jPCA) | [Churchland et al. 2012, *Nature* 483:78](https://doi.org/10.1038/nature11129) | ✅ |
| **TCA** (tensor decomposition) | Neuron × time × trial factors, jointly | [neurostatslab/tensortools](https://github.com/neurostatslab/tensortools) | [Williams et al. 2018, *Neuron* 98:1099](https://doi.org/10.1016/j.neuron.2018.05.015) | ✅ |

**Applicability to SHD.** dPCA wants a factorial design; your only factors are digit and speaker, so it is usable but thin. GPFA and jPCA were built for trial-aligned reaching data — SHD's variable durations need care. TCA fits SHD well and is under-used on it.

## 3 · Latent dynamical systems

These infer a *dynamical system*, not just a projection. More powerful, more ways to fool yourself.

| Method | What it gives you | Code | Paper | Laptop |
|---|---|---|---|---|
| **rSLDS / SLDS** | Discrete states with linear dynamics in each — interpretable regimes | [lindermanlab/ssm](https://github.com/lindermanlab/ssm) · [probml/dynamax](https://github.com/probml/dynamax) | Linderman et al. 2017 | ⚠️ |
| **LFADS / AutoLFADS** | Single-trial denoised latent trajectories from a sequential autoencoder | [snel-repo/autolfads](https://github.com/snel-repo/autolfads) | [Pandarinath et al. 2018, *Nat. Methods* 15:805](https://doi.org/10.1038/s41592-018-0109-9) | ⚠️ |
| **PSID** | Latents split into *behaviourally relevant* and irrelevant subspaces | [ShanechiLab/PSID](https://github.com/ShanechiLab/PSID) | [Sani et al. 2021, *Nat. Neurosci.* 24:140](https://doi.org/10.1038/s41593-020-00733-0) | ✅ |
| **CEBRA** | Contrastive embeddings jointly constrained by behaviour or by time | [AdaptiveMotorControlLab/CEBRA](https://github.com/AdaptiveMotorControlLab/CEBRA) | [Schneider, Lee & Mathis 2023, *Nature* 617:360](https://doi.org/10.1038/s41586-023-06031-6) | ✅ |

**CEBRA is the one to try** if you want something modern on this dataset: it takes labels (the digit) as a constraint, the API is close to scikit-learn's, and it runs on CPU at this scale. It is also the method most likely to produce a beautiful embedding that means less than it appears to — so hold it to the same shuffled-label control as everything else.

## 4 · Comparing representations

Once you have two representations — two methods, two layers, model versus data — these say whether they are the same.

| Method | What it gives you | Code | Paper | Laptop |
|---|---|---|---|---|
| **RSA** | Compare representational dissimilarity matrices | [rsagroup/rsatoolbox](https://github.com/rsagroup/rsatoolbox) | Kriegeskorte et al. 2008 | ✅ |
| **Shape metrics** | A proper *metric* between representations, so distances obey the triangle inequality | [neurostatslab/netrep](https://github.com/neurostatslab/netrep) | [Williams et al. 2021, arXiv:2110.14739](https://arxiv.org/abs/2110.14739) | ✅ |
| **DSA** | Compares the *dynamics*, not just the geometry, of two systems | [mitchellostrow/DSA](https://github.com/mitchellostrow/DSA) | [Ostrow et al. 2023, arXiv:2306.10168](https://arxiv.org/abs/2306.10168) | ✅ |

**This is where the project's route C lives.** Recover the hair-cell layer, then use one of these to say precisely how the bushy-cell representation differs from it — rather than eyeballing two heatmaps.

## 5 · Topology and intrinsic dimension

Asks what *shape* the manifold is, rather than what it looks like projected.

| Method | What it gives you | Code | Paper | Laptop |
|---|---|---|---|---|
| **Persistent homology** | Holes, loops, connected components — is it a ring? a sphere? | [scikit-tda/ripser.py](https://github.com/scikit-tda/ripser.py) · [giotto-ai/giotto-tda](https://github.com/giotto-ai/giotto-tda) | [Chaudhuri et al. 2019, *Nat. Neurosci.* 22:1512](https://doi.org/10.1038/s41593-019-0460-x) | ✅ |
| **Intrinsic dimension** (TwoNN, MLE) | One number for manifold dimension, no embedding needed | [`scikit-dimension`](https://scikit-dimension.readthedocs.io/) | Facco et al. 2017 | ✅ |
| **Participation ratio** | Effective dimensionality from the covariance spectrum, threshold-free | `compbio2026.geometry.participation_ratio` | Gao et al. 2017 | ✅ |

Chaudhuri et al. is the model result for this section: they recovered a *ring* in head-direction cells without assuming one, which is exactly the kind of claim topology can support and a 2-D embedding cannot.

## 6 · Foundation models — read about, do not build on

Where the field is going, and out of scope for ten days on a laptop. Knowing they exist is worth two sentences in your report.

| Model | Idea | Code | Paper |
|---|---|---|---|
| **POYO** | Tokenise individual spikes; one transformer across sessions, animals and tasks | [nerdslab/poyo](https://github.com/nerdslab/poyo) · [neuro-galaxy/torch_brain](https://github.com/neuro-galaxy/torch_brain) | [Azabou et al. 2023, arXiv:2310.16046](https://arxiv.org/abs/2310.16046) |
| **Neuroformer** | Multimodal generative pretraining on neural data | ICLR 2024 | — |

❌ These need many sessions and GPU time. On a single dataset with 700 channels they will not beat a ridge readout.

## 7 · Benchmarks and reviews

- **Neural Latents Benchmark '21** — standardised comparison of latent variable models. [neurallatents/nlb_tools](https://github.com/neurallatents/nlb_tools) · [arXiv:2109.04463](https://arxiv.org/abs/2109.04463)
- **Machine Learning Methods for Studying Latent Neural Activity Dynamics** (2026) — the current survey of this whole table. [arXiv:2606.10530](https://arxiv.org/abs/2606.10530)
- Cunningham & Yu 2014, *Nat. Neurosci.* 17:1500 — [doi](https://doi.org/10.1038/nn.3776)
- Gallego et al. 2017, *Neuron* 94:978 — neural manifolds for movement control — [doi](https://doi.org/10.1016/j.neuron.2017.05.025)
- Vyas et al. 2020, *Annu. Rev. Neurosci.* 43:249 — computation through population dynamics — [doi](https://doi.org/10.1146/annurev-neuro-092619-094115)
- Jazayeri & Ostojic 2021, *Curr. Opin. Neurobiol.* 70:113 — intrinsic versus embedding dimensionality — [doi](https://doi.org/10.1016/j.conb.2021.08.002)

---

## How to choose

1. **PCA first.** If it separates the classes, most of the rest is decoration.
2. **Add one method at a time**, and justify each addition against a specific question.
3. **Every method gets the same control:** run it on shuffled labels. If the picture still looks structured, the structure is not about the digits.
4. **Quantify, do not eyeball.** Trustworthiness, class separation and decoding accuracy put every method on a common axis.
5. **Two methods that disagree are a result.** Do not quietly keep the one whose plot you preferred.

A project that uses three methods well beats one that uses ten badly.
