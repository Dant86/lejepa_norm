# E2 part 2: is the radius redundancy LeJEPA-specific? Trained baselines (SimCLR, VICReg, supervised)

- **Branch / version:** `radius-baselines-train` (pre-registration committed before any baseline result existed; see the git history of this file)
- **Pre-registration written (UTC):** 2026-10-26T00:50Z (the trainings had been submitted, SLURM jobs 2247675 / 2247676 / 2247677, and none had started)
- **Scope:** h is not analyzed; LeJEPA's product is z. All analyses are in z-space.
- **Datasets:** STL-10 unlabeled + train (100k images, labels unused) for SimCLR and VICReg; STL-10 `train` split with labels (5000 images) for the supervised baseline; evaluation on STL-10 `train` (bank) and `test` ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10); the canonical archive, via the prepared arrays, was used; the HF page is linked for reference only).
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), pretrained weights not used), the same 2048-2048-256 projector as LeJEPA.
- **Per-run notes** (training facts, GPU, commands, generated results): `train_simclr.md`, `train_vicreg.md`, `train_supervised.md`.

## Question

The LeJEPA embedding z has a radius that is predictable from its direction (held-out R^2 0.945 at lambda = 0.05, seed 0; 0.947 +- 0.001 over three seeds; disjoint-halves 0.86 against a split-half reliability of 0.93; permutation null about 0). Is that redundancy specific to LeJEPA / SIGReg, or does any learned embedding of this backbone show it? And does randomizing the radius of those embeddings hurt downstream readers the way (not) it does for LeJEPA?

## Setup (what is fixed here, before results)

All three runs: ResNet-18, projector 512 -> 2048 -> 2048 -> 256 (BatchNorm + GELU, plain linear output), z = projector output, AdamW, peak lr 5e-4 at batch 256 with 10 warmup epochs and cosine decay, weight decay 0.05, bf16, 200 epochs, seed 0, the same GPU augmentation pipeline and online k-NN monitor as the LeJEPA runs. Only the loss (and the views it uses) differs:

| run | loss | views | notes |
|---|---|---|---|
| `baseline_simclr_s0` | NT-Xent, temperature **0.2** | 2 global (96 px) | z is the projector output *before* the L2 normalization that the loss applies internally; the loss never sees `||z||`, so the radius is a built-in nuisance reference. Temperature 0.2 is our choice (0.1-0.5 are common; not tuned). 510 negatives per anchor at batch 256. |
| `baseline_vicreg_s0` | VICReg: 25 * MSE(z1, z2) + 25 * variance hinge + 1 * covariance, on z | 2 global | Standard coefficients 25 / 25 / 1 (Bardes et al., arXiv:2105.04906). No expander (z itself, d = 256, is regularized), batch 256. No radial constraint. |
| `baseline_supervised_s0` | cross-entropy of a linear head on z, the projector trained jointly with the head | 2 global + 4 local (same views as LeJEPA, every view classified) | **Data difference:** STL-10 `train` only (5000 labeled images, 19 steps per epoch, 3800 steps in total against 78,000 for the 100k-image runs). This is a weaker and odd baseline (little data, few steps, supervised labels shape z directly); it is included because the spec lists it, not because it is an equal-footing comparison. |

Analyses (z only, center views, same `configs/base.yaml` evaluation settings as the LeJEPA study): the redundancy controls (same-view, disjoint halves, single view, permutation null with 500 shuffles, F-test; `analyze --redundancy-controls`, STL-10 test dump) and the E1 radius randomization (permute / residual-only shuffle / constant radius; Euclidean k-NN, linear probe, MLP; retrained and frozen protocols; paired bootstrap with TOST, **equivalence margin 0.5 accuracy points, fixed before any baseline result**; `analyze --randomize-radius`). Comparison reference: the LeJEPA lambda = 0.05 runs (`radius_conditional/redundancy_controls_summary.md`, `radius_randomization/lam0.05.md`).

Note on the generated tables: the shared analysis code groups runs by `loss.lambda`; a baseline has no SIGReg weight, so the "lambda" column of its randomization tables shows the unused default 0.05 and means nothing.

## Pre-registered reading (fixed before looking at any baseline result)

**Validity gate.** Every training reports its k-NN accuracy (k = 20, cosine, STL-10 test, bank = train) for h and z from the online monitor and the dumps. A baseline whose z k-NN accuracy is below **0.55** (LeJEPA: about 0.78-0.79; chance 0.10) is declared a *failed or undertrained* run and none of its redundancy or randomization numbers is read as evidence about nuisance behaviour.

**Redundancy (same-view held-out R^2 of the mean log-radius from the direction, nested-CV kernel ridge, STL-10 test, compared with LeJEPA's 0.94-0.95).**

- SimCLR and/or VICReg R^2 >= 0.85 and a permutation null near 0: the redundancy is *generic to learned embeddings* of this backbone, not specific to LeJEPA / SIGReg. The disjoint-halves value is read against that run's own split-half reliability, as for LeJEPA.
- SimCLR R^2 <= 0.50: SimCLR's radius is *not* predictable from the direction, so LeJEPA's SIGReg (or its invariance-to-global-mean loss) is what creates the redundancy. (For SimCLR the loss gives the radius no gradient at all, so a low value means the radius is free noise of the network; a high value means the network ties the radius to the direction without being asked to.)
- 0.50 < R^2 < 0.85: partial / mixed; reported as such and not assigned to either story.
- **Decision boundary for more seeds:** if a baseline's R^2 falls in [0.80, 0.90] or [0.45, 0.55], seeds 1 and 2 are trained for that baseline (and this is said in the note); otherwise one seed is reported as one seed.
- The supervised run is read separately and only as a descriptive reference (different data, labels used); it does not decide between the two stories above.

**Usage (radius randomization).** Margin 0.5 points, verdicts as in E1 (`equivalent` if the 90% CI is inside +-0.5, `different` if not equivalent and the 95% CI excludes 0, otherwise `inconclusive`). Per baseline: equivalence for the retrained and frozen linear probe and MLP means the radius is unused by those readers, as for LeJEPA; a `different` verdict means the radius carries label information (or a scale the readers rely on) for that baseline. Expected, not a surprise: the Euclidean k-NN's frozen constant-radius condition collapses by construction (the metric depends on the radius); a supervised-head z may lose accuracy under the frozen linear head because the logits scale with the radius. A high redundancy R^2 *together with* equivalence under randomization is what "radius is a nuisance, and that is generic" would look like; a high R^2 with a significant randomization effect would say the radius is predictable from the direction yet still read by the downstream readers, a mixed result reported as such.

**What this does not show.** One seed per baseline (unless the boundary rule triggers), one backbone, one temperature / coefficient set; R^2 calibration (random-init network, random smooth functions of the direction, learning curve in N) is E2 part 1 on `radius-baselines-controls` and is not repeated here.

## Results

(Generated sections are added below after the runs finish.)
