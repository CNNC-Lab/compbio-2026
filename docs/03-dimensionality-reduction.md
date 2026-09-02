# Dimensionality reduction: what each method assumes

Every method here maps an `(n_samples, n_features)` matrix to a lower-dimensional one. They differ in what they preserve, and therefore in what they can show you. Choosing among them is a scientific decision.

| Method | Preserves | Linear | Noise model | Uses labels | Watch out for |
|---|---|---|---|---|---|
| **PCA** | Directions of maximum variance | yes | none (implicitly isotropic) | no | Variance ≠ information. A high-variance direction can be a speaker artefact. |
| **Factor analysis** | Shared variance, separating private noise | yes | explicit, per-feature | no | Needs enough trials relative to features; on wide matrices it will struggle. |
| **MDS** | Pairwise distances | no | none | no | Global structure at the expense of local. |
| **Isomap** | Geodesic distances on a neighbour graph | no | none | no | Neighbourhood size `k` changes the answer; a disconnected graph fails silently. |
| **LLE / spectral embedding** | Local neighbourhoods | no | none | no | Very sensitive to `k`; unstable on sparse spike data. |
| **t-SNE** | Local neighbourhoods, probabilistically | no | none | no | **Distances between clusters are not meaningful.** Perplexity changes cluster count. Never quantify separation on a t-SNE plot. |
| **UMAP** | Local neighbourhoods, with more global structure than t-SNE | no | none | no | Same caveat as t-SNE, slightly less severe. Stochastic; set a seed. |
| **dPCA** | Variance, demixed by task variable | yes | none | **yes** | Needs a designed factorial structure; on SHD your only factor is digit (and speaker). |
| **GPFA** | Smooth latent trajectories | yes latent, smooth in time | explicit | no | Built for trial-aligned data; SHD's variable durations need care. |
| **CEBRA** | Latents consistent with a label or with time | no | contrastive | optional | Heavier machinery; a reasonable stretch goal, not a starting point. |

## The trap

A 3-D embedding in which the digits look separated is **not** evidence that the digits are separable. Points that look separated in three dimensions may be separable only because you chose those three dimensions, and nonlinear methods will happily manufacture apparent clusters from noise.

Two defences, and you should use both:

1. **Quantify rather than eyeball.** `geometry.trustworthiness` puts every method on one axis. `geometry.class_separation` says whether classes are actually apart.
2. **Run the embedding on shuffled labels.** If the picture still looks structured, the structure is not about the digits.

## A sane order of work

1. PCA first, always. It is fast, deterministic, and tells you the effective dimensionality. If PCA already separates the classes, most of the nonlinear machinery is decoration.
2. Add one nonlinear method and compare it to PCA on trustworthiness *and* on decoding accuracy from the embedding.
3. Only then add more.

## Reading

- Cunningham, J. P. & Yu, B. M. (2014). Dimensionality reduction for large-scale neural recordings. *Nat. Neurosci.* 17:1500–1509.
- Jazayeri, M. & Ostojic, S. (2021). Interpreting neural computations by examining intrinsic and embedding dimensionality. *Curr. Opin. Neurobiol.* 70:113–120.
- Hurwitz, C. et al. (2021). Building population models for large-scale neural recordings: opportunities and pitfalls. *Curr. Opin. Neurobiol.* 70:64–73.
- Chari, T. & Pachter, L. (2023). The specious art of single-cell genomics. *PLoS Comput. Biol.* — the sharpest available warning about reading t-SNE/UMAP plots as if distances meant something.
