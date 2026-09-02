# The three stages

The stages are a spine, not a recipe. You design the plan and the experiments; this document says what each stage is *for* and what would count as having done it well.

---

## Stage 1 — Population statistics and state-space structure

**Question.** What does this population actually look like before any model is fitted to it?

Characterise the data first: firing-rate distributions, sparsity, temporal profiles, trial-to-trial variability, class balance, duration distribution. Then build the neurons × time matrix and confront the fact that binning and smoothing are consequential.

Then apply dimensionality reduction, linear first (PCA, factor analysis) and nonlinear after (UMAP, Isomap, spectral embedding, t-SNE). Visualise trajectories by digit class.

**What separates a good Stage 1 from a gallery of embeddings:** you compare methods on a *quantitative* criterion rather than by eye. `compbio2026.geometry` gives you participation ratio, class separation, between-class distance versus within-class radius, and trustworthiness. Build a table across methods and preprocessing conditions; that table is the deliverable.

```python
from compbio2026 import data, geometry
X, y, t = data.build_design_matrix(shd, bin_ms=10)
Xf = data.flatten(X)
geometry.summarize(Xf, y)
```

**A good result looks like:** "participation ratio drops from 84 to 31 as smoothing goes from 0 to 20 ms, while class separation is flat — so the dimensions being removed carry no class information."

---

## Stage 2 — Decodability

**Question.** How much of the visible structure is usable by a downstream reader?

Train linear readouts (ridge, multinomial logistic) to classify the spoken digit. Establish the baseline properly: cross-validation, chance level, a shuffled-label control, learning curves against number of trials and number of channels, and accuracy as the trial unfolds.

```python
from compbio2026 import decoding
res = decoding.decode(Xf, y)
null = decoding.shuffle_control(Xf, y)
n, m, s = decoding.learning_curve_channels(X, y)
t_ms, acc = decoding.accuracy_over_time(X, y, t)
```

**The published baseline to hand yourself:** Cramer et al. 2020, Figure 3 — ~60 % without spike timing, ~85 % with. Reproducing it is a real check on your pipeline; beating it is a result.

**Then the actual question of this project:** does any geometric property from Stage 1 predict decoding accuracy? Compute both across a sweep of preprocessing conditions and regress one on the other. This is assumed constantly in the population-coding literature and tested rarely.

**A good result looks like:** "across 24 preprocessing conditions, class separation predicts accuracy (r = 0.8) but participation ratio does not (r = 0.1)" — or the opposite, which would be more interesting.

---

## Stage 3 — Which subspace carries the digit

**Question.** Knowing the information is there, what carries it?

The naive version is "which neurons?", and that is the wrong unit. A population code is distributed and redundant: stimulus identity is carried by a **shared subspace**, a coordinated pattern across many channels, not by a list of important cells. So ask instead: is there a low-dimensional subspace in which the digit classes separate, and how much of the population do you need to span it?

**Sparse group lasso** gives a handle. Each *group* is one cochlear channel across all of its time bins, so the group penalty selects channels while within-group sparsity keeps the temporal support tight. The output is a short list of channels — a candidate basis, not a ranking.

**Baselines it has to beat:** selection by firing rate, by variance, by univariate class discriminability, and by random choice. If a simple heuristic matches the structured method, *that is a result*, not a failure.

**Ablation is the necessity test.** Remove the selected channels, refit from scratch, measure the drop. In a redundant code the remaining population compensates, and the size of that gap is the most informative number in the project.

```python
from compbio2026 import selection
table = selection.compare_selectors(X, y, k=50)
```

### Required framing

State this explicitly in the report:

> Selection and ablation here establish causal relevance **for the decoder, not for the circuit**. The data are fixed recordings and nothing is intervened upon in the biological system.

Distinguishing decoder-causality from circuit-causality is a live confusion in the interpretability literature. Getting it right is part of the assessment.

---

## Deliverables

- A short written report built around **one defensible result**, with methods, controls, and stated limitations
- The final presentation (Saturday 12 September)
- Reproducible analysis code — notebooks or scripts, in this repository

A negative or null result, properly established and controlled, counts fully.
