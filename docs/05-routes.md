# Three routes through the project

Pick one and justify the choice. They are different projects, not difficulty levels, and all three can produce a defensible result in a week and a half.

## Route A — analyse this population directly

Take SHD as given and run Stages 1–3 on it.

**For you if:** you want a clean, checkable result and would rather spend your time on the analysis than on the model. The structure is clean, the labelling is complete, and every claim is verifiable against ground truth.

**Risk:** low. This is the route that reliably finishes.

**The interesting version:** don't stop at "the digits are decodable". Go after the relationship between geometry and decodability, or the ablation compensation gap. Both are open questions.

## Route B — use it as input to a circuit

Drive a biophysical circuit model with the SHD spike trains and analyse the *second* layer instead.

The cochlea gives you a realistic input; the circuit gives you recurrent dynamics the cochlea does not have. The question is what recurrence **adds** to a representation that already carries the signal: does it separate the classes further, hold them longer, or throw information away?

**Tooling.** The encoder / circuit / decoder pattern is the template: encode the SHD spikes as input currents, drive a recurrent pool, read out from its state. A balanced random network of leaky integrate-and-fire neurons is the standard cortical caricature. Any of [Brian2](https://brian2.readthedocs.io/), [NEST](https://nest-simulator.org/) or a hand-rolled NumPy/JAX LIF network will do; Brian2 is the easiest to install and fast enough at this scale.

**Risk:** medium-high. Simulation parameters are their own research problem, and it is easy to burn a week on a network that produces nothing but silence or saturation. Fix that risk by starting from a network that already works, and changing one thing.

**Minimum viable result:** decoding accuracy from the circuit layer versus from the input layer, with matched readouts and matched dimensionality. If the circuit does not help, say so — that is a result.

## Route C — decompose and compare across layers

Recover the layers of the auditory pathway and ask how the representation changes as you ascend it.

Two directions:

**Downwards, into the model.** Recover the hair-cell layer that lies underneath the SHD bushy-cell spikes — see [04-hair-cell-recovery.md](04-hair-cell-recovery.md). Then run the same Stage 1–3 analysis on the hair-cell layer and on the bushy-cell layer and compare. Does coincidence detection at the bushy cell improve class separation, or discard information a decoder could have used?

**Upwards, into the brain.** SHD's bushy cells sit in the **cochlear nucleus**, the first synapse of the ascending auditory pathway. Above them:

| Stage | What it is thought to do |
|---|---|
| Hair cells | Transduction; adaptation; saturating nonlinearity |
| Bushy cells (cochlear nucleus) | Coincidence detection; sharpened phase locking |
| Superior olivary complex | Binaural comparison — interaural time and level differences |
| Inferior colliculus | Convergence hub; modulation-rate tuning emerges |
| Thalamus (MGN) | Gating and modulation; strong descending control |
| Cortex A1 | Tonotopic map; spectrotemporal receptive fields; task-dependent plasticity |

Modelling any of these is a project on its own. A tractable version: implement **one** stage above the bushy cell — coincidence detection or modulation-rate tuning are both a few lines — and ask what it does to decodability.

**Risk:** medium. The recovery direction is well-posed and has a validation path; the ascending direction needs you to pick one stage and resist the temptation to build the whole pathway.

**Note on binaural stages.** SHD is monaural. Anything at or above the superior olive is about comparing two ears, and you only have one. You would have to simulate the second, and then you are modelling rather than analysing. Be explicit about that if you go there.

---

## Choosing

| | A | B | C |
|---|---|---|---|
| Finishes in 10 days | reliably | if you are disciplined | yes for the recovery direction |
| Needs simulation | no | yes | some |
| Ground truth available | yes | no | yes, via LAUSCHER |
| Most open scientific question | geometry ↔ decodability | what recurrence adds | what each stage adds |

Whatever you choose: **one defensible result, with its controls, beats three suggestive ones.**
