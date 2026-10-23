# Spec: is the LeJEPA embedding radius a nuisance parameter?

- **Branch:** `radius-nuisance` (spec only; each experiment below gets its own feature/research branch, see "Branches")
- **Status:** draft, agreed in discussion, not yet reviewed line by line
- **Scope rule:** **z only** (the projector output, LeJEPA's product). `h` is not analyzed in any new experiment; every new note carries the line "h is not analyzed; LeJEPA's product is z". Existing `h` results in `research_notes/radius_conditional/` stay as they are.

## Claim and the three legs

Claim: in LeJEPA embeddings the radius `||z||` is a nuisance parameter, i.e. it carries no label-relevant information beyond the direction `u = z/||z||`.

1. **Predictability (done).** The mean log-radius is predicted from the direction: held-out R^2 0.945 (lambda = 0.05, seed 0; kernel ridge), 0.947 ± 0.001 / 0.936 ± 0.001 over three seeds at lambda 0.05 / 0.1; cross-view (disjoint halves) 0.86 / 0.84 against split-half reliabilities 0.93 / 0.92; permutation null about 0. Notes: `radius_conditional/redundancy_controls_summary.md`.
2. **Usage (in progress, `radius-randomization`).** Randomizing the radius does not hurt downstream readers.
3. **Necessity and mechanism (this spec).** The model does not need the radius to train, the predictability is not a generic property of networks, and SIGReg has little power to constrain the radius.

Known fact that motivates 3 and is *not* explained yet: `kappa_z` = 26.3 / 20.2 / 14.6 at lambda 0.02 / 0.05 / 0.1 (Gaussian: 1).

## Experiments

### E1. Usage (running; extended brief)
Branch `radius-randomization`. z only. Conditions: baseline, full radius permutation, residual-only shuffle (direction-predicted part of the log-radius kept, out-of-fold), constant radius. Readers: Euclidean k-NN, cosine k-NN (reference), linear probe, 1-hidden-layer MLP (fixed budget, weight decay chosen on a validation split). Probes in two protocols: *retrained* on shuffled train features, and *frozen* (trained clean, applied to shuffled test features). Runs: lambda {0.02, 0.05, 0.1} x seeds {0, 1, 2}.
Statistics: paired differences to baseline with a paired bootstrap over test images (resample images within seed, average over seeds), and an equivalence verdict (TOST / CI inside the margin). **Margin fixed before looking: 0.5 accuracy points.** Shuffle repeats only measure shuffle noise.
Reading: equivalence for retrained probes and for the frozen probe = radius unused. A drop for Euclidean k-NN alone is partly mechanical (the metric depends on the radius by construction).

### E2. Is the predictability LeJEPA-specific? (baselines)
Branch `radius-baselines`. Same redundancy controls as the existing study (same-view, disjoint-halves, permutation null, F), z only, on:
- random-init network (same backbone and projector, untrained),
- supervised (cross-entropy on STL-10 labels; **open:** how to define z, default = the same projector head trained jointly with a linear classifier on its output),
- SimCLR (z = projector output *before* its L2 normalization; the loss never sees the radius: a built-in nuisance reference),
- VICReg (per-dimension variance constraint, no radial constraint).
Controls on the same z: (a) predict a *random smooth function of u* with matched smoothness (kernel-ridge-drawn from the same kernel family) to calibrate what R^2 means; (b) a learning curve of the redundancy R^2 in N (is 4000 saturating?).
Reading: if random-init gives R^2 ~ 0.9, the predictability says little about SIGReg. If random-init is low and trained models are high, training creates it. SimCLR high would say the predictability is generic to learned embeddings.
Backbones: ResNet-18 first; ViT-S added once E5 provides it. Training budget: one seed per baseline at first (plus seeds if the numbers sit near a decision boundary).

### E3. Training without the radius (intervention)
Branch `sigreg-sphere`. A flag in the LeJEPA loss path that replaces z by `z~ = sqrt(d) * z/||z||` before **both** SIGReg and the invariance loss; the radius is then exactly constant. Caveat to document: SIGReg targets N(0, I); 1-D projections of `z~` are only approximately Gaussian (mismatch O(1/d), small at d = 256). The backbone feature `h` keeps its radius (out of scope).
Runs: ResNet-18, lambda = 0.05, seeds {0, 1, 2}, otherwise `configs/base.yaml`. Compare with the matching lambda = 0.05 radius runs: k-NN and linear-probe accuracy (z), direction-only quality, invariance loss. Equivalence margin as in E1. This is the cheap version of the planned Variants B/C question; it does not use a radial loss.

### E4. Mechanism: why is the radius loosely constrained? (synthetic, no training)
Branch `sigreg-power`. Hypothesis (back-of-envelope, **not yet measured**): for `z = r u` with u uniform on the sphere, a 1-D projection is a scale mixture of Gaussians with kurtosis 3 E[r^4]/E[r^2]^2 = 3 (1 + 2 kappa/d). At kappa = 20, d = 256 that is 3.47 against 3, a small departure in any single marginal, so the sliced Epps-Pulley statistic has little power and a weak gradient on the radius. Consistent with the measured relative spread of `||z||` (0.17-0.23 vs 0.044 for a Gaussian).
Design: synthetic samples with Gaussian-uniform directions and radial spread matched to kappa in {1, 5, 15, 20, 26}, the training batch size and slice count; (a) SIGReg value against its sampling distribution under exact Gaussians (power curve in kappa), (b) norm of the gradient of SIGReg with respect to the radius versus with respect to the direction, (c) a direct radial statistic (chi_d KS) as the high-power contrast. Also report the same quantities on the real embeddings of the nine sweep checkpoints.
Reading: flat SIGReg and gradient in (a)-(b) at the observed kappa = mechanism for the unconstrained radius.

### E5. Architecture generality (four models)
Branch `backbones-vit`. Models: **ResNet-18, ResNet-50, ViT-S, ViT-B** (ViT-Ti / ViT-S if compute is tight; decide after measuring ViT-S epoch time). ResNet-50 is already registered; the ViTs go through timm with an explicit `img_size` / patch size for 96 px STL-10 (a builder with tests; the ViT final LayerNorm affects `h` only, which we do not analyze). Runs: lambda = 0.05; 3 seeds for ResNet-18 (exists) and the smaller models, 1-2 seeds for the large ones. On each: the redundancy controls and E1 (and E3 if E3 comes out clean). Known risk: ViTs from scratch on 100k unlabeled images are fragile; report accuracy so a failed training is not read as a nuisance result. Compute is the main cost (ResNet-18 is ~2.5 h per run; ViT at 96 px will be several times that).

### E6. Clean-to-corrupted transfer (cheap, no training)
Branch `radius-transfer`. Fit the map u -> rho on clean STL-10 test embeddings and predict rho on the paired STL-10-C embeddings (z only; the corruption radius shift is systematic and significant, so this separates "the direction moved" from "the radius carries shift information beyond the direction"). Report the R^2 of the transferred map and the residual shift per corruption and severity with bootstrap CIs. Optionally the same for DTD and SVHN as OOD sets.
Reading: transfer works = the shift is a movement of the direction; fails = the radius carries information not in the direction (a counterexample to be reported).

### E7. When does the redundancy appear? (cheap, no training)
Branch `radius-dynamics`. Redundancy R^2 and `kappa_z` at the epoch checkpoints (`epoch_XXXX.pt`, every 10 epochs) of the lambda = 0.05 runs. The `kappa_z` minimum at epochs 3-4 and the later rise are the thing to relate to the onset of the redundancy.

## Branches and ordering
Each question gets its own branch and `research_notes/<name>/` directory (CLAUDE.md). Library code that is reusable goes into `src/lejepa_norm/` on a feature branch with tests, merged to `master`, then used.

1. E1 (running).
2. E4, E6, E7: no training, reuse existing embeddings and checkpoints; can run in parallel with E1.
3. E3: small training change (flag + tests), 3 runs of ~2.5 h.
4. E2: needs baseline losses (SimCLR, VICReg, supervised head) and a random-init mode; trainings.
5. E5: new backbones, then the largest compute; start the ResNet-50 and ViT-S pilots early to size the budget.

## Open decisions (defaults used unless changed)
- Supervised baseline's z: projector trained jointly with a linear head (E2).
- Equivalence margin: 0.5 accuracy points (E1, E3).
- Seeds for the large models: 1-2 (E5).
- Whether E6 includes DTD/SVHN: optional, default no.

## Reporting rules (all experiments)
Notes follow the template in CLAUDE.md: branch / commit, UTC datetime, exact commands, datasets and backbones linked to their HF pages (and the canonical source where it differs), then Question, Setup, Results (generated tables, PNGs), Interpretation, Next steps. Result tables are generated by the analysis app; figures via `lejepa_norm.plotting`.
