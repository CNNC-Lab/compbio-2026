# Reading

Start with the three starred items. The rest is there when you need it.

## The dataset

- ★ **Cramer, B., Stradmann, Y., Schemmel, J. & Zenke, F. (2020).** The Heidelberg spiking data sets for the systematic evaluation of spiking neural networks. *IEEE TNNLS.* [arXiv:1910.07407](https://arxiv.org/abs/1910.07407) — where the data comes from and the decoding baseline you will reproduce (Figure 3).
- Meddis, R. (1986). Simulation of mechanical to neural transduction in the auditory receptor. *JASA* 79(3):702. — the hair-cell model.
- Fettiplace, R. & Hackney, C. M. (2006). The sensory and motor roles of auditory hair cells. *Nat. Rev. Neurosci.* 7:19–29. — what the biology actually does; Figure 1 is the one-page version of the auditory periphery.

## Population coding — the framing

- ★ **Saxena, S. & Cunningham, J. P. (2019).** Towards the neural population doctrine. *Curr. Opin. Neurobiol.* 55:103–111. — the argument for analysing populations rather than cells. Figure 2a is the clearest picture of a distributed code you will find.
- Cunningham, J. P. & Yu, B. M. (2014). Dimensionality reduction for large-scale neural recordings. *Nat. Neurosci.* 17:1500–1509. — still the standard reference for the method landscape.
- Hurwitz, C., Kudryashova, N., Onken, A. & Hennig, M. H. (2021). Building population models for large-scale neural recordings: opportunities and pitfalls. *Curr. Opin. Neurobiol.* 70:64–73.
- Jazayeri, M. & Ostojic, S. (2021). Interpreting neural computations by examining intrinsic and embedding dimensionality of neural activity. *Curr. Opin. Neurobiol.* 70:113–120. — the distinction that keeps "dimensionality" from being a meaningless word.

## Dynamics

- Rabinovich, M. I., Huerta, R. & Laurent, G. (2008). Transient dynamics for neural processing. *Science* 321(5885):48–50. — three pages, and the reason this project is about trajectories rather than states.
- Durstewitz, D. & Deco, G. (2008). Computational significance of transient dynamics in cortical networks. *Eur. J. Neurosci.* 27(1):217–227.
- Rabinovich, M. I. & Varona, P. (2011). Robust transient dynamics and brain functions. *Front. Comput. Neurosci.* 5:24.

## Methods you will use

- ★ **Simon, N., Friedman, J., Hastie, T. & Tibshirani, R. (2013).** A sparse-group lasso. *J. Comput. Graph. Stat.* 22(2):231–245. — read at least the first three pages before using it.
- McInnes, L., Healy, J. & Melville, J. (2018). UMAP. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
- Kobak, D. & Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. *Nat. Commun.* 10:5416. — practical, and the warnings transfer directly.
- Chari, T. & Pachter, L. (2023). The specious art of single-cell genomics. *PLoS Comput. Biol.* 19(8):e1011288. — the case against reading distances off a 2-D embedding. Read it before you make a claim from a UMAP plot.

## Interpretability, and the trap in it

- Jonas, E. & Kording, K. P. (2017). Could a neuroscientist understand a microprocessor? *PLoS Comput. Biol.* 13(1):e1005268. — what happens when you apply these methods to a system whose ground truth you actually know.
- Vilas, M. G., Schaeffer, R. & Kriegeskorte, N., or any recent treatment of decoder-versus-circuit causality. The distinction matters more than any particular method.

## If you want the wider course

- Dayan, P. & Abbott, L. F. (2001). *Theoretical Neuroscience.* MIT Press.
- Gerstner, W., Kistler, W., Naud, R. & Paninski, L. (2014). *Neuronal Dynamics.* Cambridge. Free at <https://neuronaldynamics.epfl.ch/>
- [Neuromatch Academy](https://compneuro.neuromatch.io/) — computational neuroscience and NeuroAI curricula, free, with runnable notebooks.
