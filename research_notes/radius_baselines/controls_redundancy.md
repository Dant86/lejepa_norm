# E2 (part 1): is the radius-direction redundancy specific to trained LeJEPA? Random-init baseline
- **Branch / version:** `radius-baselines-controls` @ `<commit>` (filled in with the results commit; the reading below was committed before any job was submitted)
- **Run datetime (UTC):** <filled in with the results>
- **Command(s):** <filled in with the results>
- **Datasets:** STL-10 `test` (first 4000 images after the seeded subsample used by every redundancy table) clean images ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10)); the canonical archive was used (via the existing `stl10_test` embedding dumps); the HF page is linked for reference only. BatchNorm calibration of the random-init networks uses STL-10 `unlabeled` + `train` images (same canonical archive, no labels).
- **Backbone:** ResNet-18 + 2048-2048-256 projector from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used) in three flavors: trained (`last.pt` of `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}`), random-init with calibrated BatchNorm statistics (`random_init_bncal_s{0,1,2}`) and random-init with identity BatchNorm (`random_init_bnid_s{0,1,2}`).
- **Scope:** h is not analyzed; LeJEPA's product is z.

## Question

The redundancy of the radius with the direction (held-out kernel-ridge R^2 of the mean log-radius from the mean direction, about 0.95 for the trained LeJEPA runs) is the premise of the radius-nuisance story. Is it a consequence of SIGReg training, or does it already hold for an *untrained* network of the same architecture (and so say little about SIGReg)? The spec is E2 of `research_notes/radius_nuisance/spec.md` (branch `radius-nuisance`).

## Setup (deviations from the base config)

No deviation from `configs/base.yaml`. The random-init network is built by the same builders as the LeJEPA runs (`torch.manual_seed(seed)`, then backbone, then projector; seeds 0, 1, 2) and written through the checkpoint contract with `epoch = 0` (`lejepa_norm.models.random_init`, `apps/embed.py --random-init SEED --bn-mode MODE`), then embedded exactly like a trained checkpoint (center view plus K = 8 augmented global views, same augmentation seeds as the trained runs, `stl10_train` and `stl10_test` only). BatchNorm needs an explicit statement because an untrained network has no meaningful running statistics:

- **calibrated (primary).** Running statistics are set by a gradient-free forward pass in train mode over 16384 augmented global views of random STL-10 `unlabeled` + `train` images (`momentum=None`, i.e. a cumulative average, so the batch order does not matter); no weight changes. The eval-mode network is then the large-batch equivalent of train-mode (batch-statistics) BatchNorm, and it is the same statistics mode (running) in which the trained checkpoints are embedded.
- **identity (secondary).** No calibration: the initial statistics (mean 0, variance 1) make eval-mode BatchNorm the identity map.
- Train-mode batch statistics at embedding time are not used (the embedding of an image would depend on its batch).

The redundancy controls are those of `research_notes/radius_conditional/redundancy_controls_summary.md` (same-view, disjoint halves, single view, permutation null with 500 permutations, OLS F-test, split-half reliability), run on z only through `lejepa_norm.eval.controls`; the new part is that the held-out predictions are kept so that each R^2 gets a 95% bootstrap interval over the 4000 images (resampled within each seed, then averaged over the seeds of a group; `randomization.bootstrap_draws` / `bootstrap_seed_mean` / `intervals` as in E1). With three seeds per group, the seed spread is reported next to it (`seed range`), the interval is conditional on the seeds. Contrasts between groups use the same resampled images for both groups (paired by image).

## Pre-registered reading (written before any result was computed)

Decision variable: same-view R^2 of z (the 0.95 headline); the disjoint-halves R^2 is the stricter check and has to agree. Trained group = lambda 0.05 (the design default), the other lambdas are read the same way. "Contrast" = trained minus random-init (calibrated), same images, 95% bootstrap interval.

1. **Predictability says little about SIGReg:** random-init same-view R^2 is at least 0.85 or the contrast's 95% interval lies above -0.05 and below +0.05 (random-init is within 0.05 of the trained R^2). The redundancy is then a property of the architecture and of how directions and norms co-vary in a ResNet-plus-MLP projector, not something SIGReg produces.
2. **Training creates it:** random-init same-view R^2 is at most 0.5 **and** the contrast's 95% interval is entirely above +0.3.
3. **Mixed:** anything between (random-init R^2 in (0.5, 0.85) or a contrast in [+0.05, +0.3]): both the architecture and the training contribute; reported as such, no sharper claim.
4. **BatchNorm mode:** if the calibrated and identity random-init groups differ by more than 0.1 in same-view R^2, the random-init verdict is declared BatchNorm-dependent and both rows are quoted wherever it is used; otherwise the calibrated group is the headline.
5. The permutation null and the F-test are expected to be "significant" for essentially any network (the effect size, not the p-value, carries the information): a significant p for random-init is not evidence of anything by itself. A split-half reliability of rho_bar much lower for random-init than for trained says the radius itself is noisier across views there (which lowers the R^2 ceiling); this is read together with the R^2.

## Results

## Interpretation

## Next steps
