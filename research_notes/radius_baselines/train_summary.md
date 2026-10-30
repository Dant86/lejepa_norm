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

## Addendum (2026-10-26, written before any analysis of h; all z results above are unchanged)

**Explicit exception to the z-only rule, scoped to the E2 baselines.** For SimCLR and VICReg the projector output z is discarded after training and the used representation is the backbone feature h; for the supervised run h is the penultimate feature. The quantity of interest for the baselines is therefore h. LeJEPA's product is still z; its h (`radius_marginal_lam0.05`) is included as a clearly labelled *secondary reference only*.

**Hypothesis under test.** SimCLR's low z same-view R^2 (0.684) is low because z is not the representation used. If that is right, SimCLR's h same-view R^2 should be clearly higher (>= 0.85).

**Readings, identical to those for z and fixed now:** same-view held-out R^2 (nested-CV kernel ridge, STL-10 test) >= 0.85 with a null near 0 = generic redundancy; <= 0.50 = not predictable; 0.50-0.85 = mixed. Randomization: margin 0.5 accuracy points, same verdicts, same conditions and readers. The more-seeds boundary rule is not re-applied (one seed each, stated). If h does not resolve the gap (SimCLR h still < 0.85), that is reported as such. Already seen before this addendum: the redundancy-controls tables generated for the z runs contained an h block (SimCLR h same-view R^2 0.869 in the generated table); it was not read at the time. This is disclosed so the h analysis is not mistaken for blind.

## Results

Numbers are copied from the per-run notes (generated there by `analyze --redundancy-controls` and `--randomize-radius`); one seed each; z only.

| run | z k-NN acc (gate 0.55) | same-view R^2 (null ~ 0) | disjoint halves R^2 | split-half reliability | pre-registered reading |
|---|---|---|---|---|---|
| LeJEPA lambda 0.05 (reference) | 0.78-0.79 | 0.945 | 0.86 | 0.93 | reference |
| `baseline_simclr_s0` | 0.7815 | 0.684 | 0.60 | 0.819 | mixed (0.50 < R^2 < 0.85); one seed |
| `baseline_vicreg_s0` | 0.7494 | 0.990 | 0.95 | 0.971 | generic (>= 0.85) |
| `baseline_supervised_s0` | 0.7955 | 0.929 | 0.65 | 0.684 | descriptive only |

Usage (equivalence margin 0.5): SimCLR and supervised: linear/MLP readers equivalent under permute/residual shuffle (one small positive `different`, +0.29, for SimCLR retrained linear residual shuffle); VICReg: residual shuffle equivalent for linear/MLP, but permute gives small `different` effects (linear -0.46 / -0.31) and frozen k-NN -5.25. Euclidean k-NN gains from removing the radius after retraining for SimCLR (+1.0 to +2.3) and VICReg constant (+0.70), as for LeJEPA. Frozen constant radius collapses every reader in every run (scale mismatch, not radius use).

Reading: the radius is substantially predictable from the direction in every trained baseline, strongest for VICReg (0.99) and weaker for SimCLR (0.68), so the redundancy is not specific to LeJEPA/SIGReg; it is not a unique SIGReg signature. VICReg is the only baseline where destroying the direction-radius link measurably (if slightly) hurts linear readers. One seed per baseline, one temperature/coefficient set, one backbone.

Per-run notes: `train_simclr.md`, `train_vicreg.md`, `train_supervised.md`.

## Results: z vs h (explicit exception to the z-only rule, per the addendum)

**Caveats:** deviation from the z-only rule, scoped to the E2 baselines; 1 seed each; the LeJEPA row is a *secondary reference only* (LeJEPA's product is z, its h is not its product; redundancy seed 0, randomization 3 seeds). Redundancy numbers come from the same `--redundancy-controls` runs as the z results (the h block of `train_<run>.md`, jobs 2247712-14; LeJEPA h: `h_lejepa_ref.md`, job 2252608); that block had been generated before the addendum, as disclosed there. Randomization on h: `h_<run>.md` (jobs 2252605-07, 2252609; code: `analyze --space h`).

**Hypothesis under test:** SimCLR's low z same-view R^2 (0.684) is because z is not the representation used.

| run | z same-view R^2 | h same-view R^2 | z disjoint halves | h disjoint halves | h split-half reliability | h reading (bands 0.50 / 0.85) |
|---|---|---|---|---|---|---|
| SimCLR | 0.684 | **0.869** | 0.60 | 0.75 | 0.856 | >= 0.85: generic |
| VICReg | 0.990 | 0.956 | 0.95 | 0.90 | 0.937 | generic |
| supervised | 0.929 | 0.950 | 0.65 | 0.57 | 0.576 | descriptive only |
| LeJEPA lam 0.05 s0 (reference, h not its product) | 0.946 | 0.925 | 0.86 | 0.79 | 0.858 | generic |

**Hypothesis verdict:** supported for the pre-registered statistic. SimCLR's h same-view R^2 is 0.869 (z 0.684): h clears the 0.85 band, so the gap is largely a property of z, not of the radius being unpredictable. Two qualifications: 0.869 is only just above 0.85 for one seed, and the disjoint-halves value (0.75) sits below that run's own reliability (0.856), so the cross-view redundancy of SimCLR's h (0.75; single-view-to-other-view 0.52) is weaker than LeJEPA's z or VICReg's h. Same-view R^2 can include view-specific structure shared between input and target. h does not turn SimCLR into a VICReg-like case on the cross-view measures.

**Usage on h (margin 0.5; retrained / frozen protocols; generated in `h_<run>.md`):**
- SimCLR: every linear and MLP cell and Euclidean k-NN permute / residual-shuffle is `equivalent`; k-NN retrained constant radius +0.45 `inconclusive`. Unlike on z (where retrained k-NN gained 1-2 points from removing the radius and residual shuffle on the linear probe was +0.29 `different`), nothing is detectable on h.
- VICReg: linear/MLP retrained all equivalent; frozen permute -0.39 (linear) and -0.56 (MLP) `different`; k-NN gains +1.28 (retrained constant radius) and is `different` in frozen permute (+0.62) and residual (+0.33). Small effects, mixed.
- Supervised: all linear/MLP equivalent; k-NN retrained inconclusive (-0.28 to -0.39), frozen permute -0.42 `different`.
- LeJEPA h (reference): all linear/MLP equivalent; k-NN retrained permute -0.42 `different`, otherwise equivalent.
- Frozen constant radius collapses every reader in every run (-35 to -70): norm mismatch (h norms are far from 1), not radius use.

**Bottom line:** for all three baselines, and for LeJEPA's h, the radius of h is as predictable from the direction as that of z (SimCLR clearly more so), and the linear/MLP readers do not use it beyond 0.5 points except VICReg's frozen-permute cells. The z-only conclusion that the redundancy is not specific to LeJEPA stands on h, and SimCLR's lower z value is explained mostly by z being a discarded projector output.
