# E2 (part 1): is the radius-direction redundancy specific to trained LeJEPA? Random-init baseline
- **Branch / version:** `radius-baselines-controls` @ `567c498` (code and pre-registered readings; this results commit follows it)
- **Run datetime (UTC):** 2026-10-26 (jobs started 2026-10-26T01:11Z by sacct; aggregation 2026-10-26T02:11Z)
- **Command(s):** `sbatch --export=ALL --array=0-5 slurm/embed_random_init.sbatch configs/base.yaml` (job 2247669, random-init embeddings, 1 GPU each, node r003: NVIDIA H200); `sbatch --export=ALL --array=0-14%8 slurm/baseline_controls.sbatch configs/base.yaml radius_marginal random_init` (job 2247779, CPU only, nodes m002/g002/h001); aggregation: `python apps/analyze.py --config configs/base.yaml --baseline-controls radius_marginal random_init --note radius_baselines/controls.md` (writes `controls_redundancy.md`, `controls_smooth.md`, `controls_curve.md`)
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

<!-- BEGIN GENERATED: baseline.controls -->
### Redundancy controls per group (z)

Held-out nested-CV kernel-ridge R^2 of the radius from the direction on the first 4000 `stl10_test` images, mean over the seeds of a group; the interval is a 95% percentile bootstrap over images (resampled within each seed, then averaged over seeds). `null max`, `worst permutation p` (500 permutations) and `min F` are the least favorable value over seeds.

| group | setup | seeds | r2 (mean over seeds) | 95% CI (images) | seed range | null max | worst permutation p | min F |
|---|---|---|---|---|---|---|---|---|
| radius_marginal_lam0.02 | same views (all 8): rho_bar ~ u_hat | 3 | 0.9484 | [0.9449, 0.9516] | [0.9471, 0.9506] | 0.003696 | 0.001996 | 247 |
| radius_marginal_lam0.02 | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.886 | [0.8809, 0.8909] | [0.8785, 0.8917] | 0.002259 | 0.001996 | 119.7 |
| radius_marginal_lam0.02 | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.8824 | [0.8771, 0.8874] | [0.8737, 0.8872] | 0.003458 | 0.001996 | 119.7 |
| radius_marginal_lam0.02 | single view, same view: rho_0 ~ u_0 | 3 | 0.9313 | [0.9275, 0.9347] | [0.9303, 0.9329] | 0.002497 | 0.001996 | 164.1 |
| radius_marginal_lam0.02 | single view, other view: rho_1 ~ u_0 | 3 | 0.7425 | [0.7330, 0.7516] | [0.7296, 0.7567] | 0.004018 | 0.001996 | 47.65 |
| radius_marginal_lam0.02 | reference: split-half reliability r^2 of rho_bar | 3 | 0.9415 |  | [0.9385, 0.9447] |  |  |  |
| radius_marginal_lam0.05 | same views (all 8): rho_bar ~ u_hat | 3 | 0.9467 | [0.9441, 0.9491] | [0.9459, 0.9484] | 0.002633 | 0.001996 | 72.19 |
| radius_marginal_lam0.05 | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.8612 | [0.8560, 0.8665] | [0.8572, 0.8654] | 0.002686 | 0.001996 | 54.69 |
| radius_marginal_lam0.05 | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.8602 | [0.8548, 0.8658] | [0.8546, 0.8664] | 0.003703 | 0.001996 | 55.33 |
| radius_marginal_lam0.05 | single view, same view: rho_0 ~ u_0 | 3 | 0.9182 | [0.9147, 0.9218] | [0.9176, 0.9189] | 0.004124 | 0.001996 | 51.65 |
| radius_marginal_lam0.05 | single view, other view: rho_1 ~ u_0 | 3 | 0.6979 | [0.6873, 0.7078] | [0.6913, 0.7083] | 0.004897 | 0.001996 | 29.08 |
| radius_marginal_lam0.05 | reference: split-half reliability r^2 of rho_bar | 3 | 0.9286 |  | [0.9276, 0.9304] |  |  |  |
| radius_marginal_lam0.1 | same views (all 8): rho_bar ~ u_hat | 3 | 0.9355 | [0.9331, 0.9380] | [0.9346, 0.9363] | 0.003358 | 0.001996 | 49.82 |
| radius_marginal_lam0.1 | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.8381 | [0.8322, 0.8439] | [0.8347, 0.8402] | 0.002058 | 0.001996 | 40.9 |
| radius_marginal_lam0.1 | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.8363 | [0.8301, 0.8421] | [0.8350, 0.8377] | 0.0044 | 0.001996 | 39.68 |
| radius_marginal_lam0.1 | single view, same view: rho_0 ~ u_0 | 3 | 0.8992 | [0.8953, 0.9028] | [0.8957, 0.9010] | 0.006754 | 0.001996 | 36.55 |
| radius_marginal_lam0.1 | single view, other view: rho_1 ~ u_0 | 3 | 0.6516 | [0.6390, 0.6631] | [0.6425, 0.6577] | 0.004319 | 0.001996 | 23 |
| radius_marginal_lam0.1 | reference: split-half reliability r^2 of rho_bar | 3 | 0.9185 |  | [0.9177, 0.9194] |  |  |  |
| random_init_bncal | same views (all 8): rho_bar ~ u_hat | 3 | 0.9726 | [0.9714, 0.9736] | [0.9697, 0.9767] | 0.002333 | 0.001996 | 294.9 |
| random_init_bncal | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.3657 | [0.3508, 0.3798] | [0.3421, 0.3797] | 0.002191 | 0.001996 | 9.553 |
| random_init_bncal | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.3779 | [0.3646, 0.3917] | [0.3531, 0.3910] | 0.003022 | 0.001996 | 10.06 |
| random_init_bncal | single view, same view: rho_0 ~ u_0 | 3 | 0.9651 | [0.9636, 0.9665] | [0.9580, 0.9713] | 0.003526 | 0.001996 | 193.6 |
| random_init_bncal | single view, other view: rho_1 ~ u_0 | 3 | 0.1013 | [0.0897, 0.1119] | [0.0867, 0.1192] | 0.005228 | 0.001996 | 2.946 |
| random_init_bncal | reference: split-half reliability r^2 of rho_bar | 3 | 0.3496 |  | [0.3256, 0.3634] |  |  |  |
| random_init_bnid | same views (all 8): rho_bar ~ u_hat | 3 | 0.9993 | [0.9992, 0.9993] | [0.9989, 0.9995] | 0.003912 | 0.001996 | 1.575e+04 |
| random_init_bnid | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.4023 | [0.3828, 0.4193] | [0.3651, 0.4272] | 0.003023 | 0.001996 | 10.67 |
| random_init_bnid | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.4025 | [0.3846, 0.4197] | [0.3678, 0.4303] | 0.002618 | 0.001996 | 11.11 |
| random_init_bnid | single view, same view: rho_0 ~ u_0 | 3 | 0.9998 | [0.9997, 0.9998] | [0.9996, 0.9999] | 0.002361 | 0.001996 | 5.681e+04 |
| random_init_bnid | single view, other view: rho_1 ~ u_0 | 3 | 0.1157 | [0.1009, 0.1295] | [0.0977, 0.1276] | 0.002584 | 0.001996 | 2.793 |
| random_init_bnid | reference: split-half reliability r^2 of rho_bar | 3 | 0.386 |  | [0.3536, 0.4136] |  |  |  |
<!-- END GENERATED: baseline.controls -->

<!-- BEGIN GENERATED: baseline.contrast -->
### R^2 minus the `random_init_bncal` group

Same held-out images, so the resamples are shared by image; 95% percentile bootstrap, seeds averaged within each group.

| group | minus | setup | difference in r2 | 95% CI (images) |
|---|---|---|---|---|
| radius_marginal_lam0.02 | random_init_bncal | same views (all 8): rho_bar ~ u_hat | -0.02423 | [-0.0304, -0.0190] |
| radius_marginal_lam0.02 | random_init_bncal | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 0.5203 | [0.4957, 0.5450] |
| radius_marginal_lam0.02 | random_init_bncal | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 0.5044 | [0.4806, 0.5280] |
| radius_marginal_lam0.02 | random_init_bncal | single view, same view: rho_0 ~ u_0 | -0.03378 | [-0.0395, -0.0285] |
| radius_marginal_lam0.02 | random_init_bncal | single view, other view: rho_1 ~ u_0 | 0.6411 | [0.6178, 0.6628] |
| radius_marginal_lam0.05 | random_init_bncal | same views (all 8): rho_bar ~ u_hat | -0.02586 | [-0.0304, -0.0220] |
| radius_marginal_lam0.05 | random_init_bncal | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 0.4955 | [0.4711, 0.5213] |
| radius_marginal_lam0.05 | random_init_bncal | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 0.4823 | [0.4568, 0.5061] |
| radius_marginal_lam0.05 | random_init_bncal | single view, same view: rho_0 ~ u_0 | -0.04684 | [-0.0526, -0.0413] |
| radius_marginal_lam0.05 | random_init_bncal | single view, other view: rho_1 ~ u_0 | 0.5966 | [0.5750, 0.6177] |
| radius_marginal_lam0.1 | random_init_bncal | same views (all 8): rho_bar ~ u_hat | -0.03708 | [-0.0414, -0.0331] |
| radius_marginal_lam0.1 | random_init_bncal | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 0.4724 | [0.4497, 0.4989] |
| radius_marginal_lam0.1 | random_init_bncal | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 0.4583 | [0.4349, 0.4832] |
| radius_marginal_lam0.1 | random_init_bncal | single view, same view: rho_0 ~ u_0 | -0.06585 | [-0.0718, -0.0601] |
| radius_marginal_lam0.1 | random_init_bncal | single view, other view: rho_1 ~ u_0 | 0.5502 | [0.5262, 0.5731] |
| random_init_bnid | random_init_bncal | same views (all 8): rho_bar ~ u_hat | 0.02665 | [0.0253, 0.0280] |
| random_init_bnid | random_init_bncal | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 0.03657 | [0.0065, 0.0631] |
| random_init_bnid | random_init_bncal | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 0.0246 | [-0.0038, 0.0515] |
| random_init_bnid | random_init_bncal | single view, same view: rho_0 ~ u_0 | 0.03468 | [0.0330, 0.0366] |
| random_init_bnid | random_init_bncal | single view, other view: rho_1 ~ u_0 | 0.01439 | [-0.0068, 0.0346] |
<!-- END GENERATED: baseline.contrast -->

## Interpretation

Against the pre-registered reading (z only; trained group lambda 0.05, calibrated random-init as reference):

- **Same-view R^2 says little about SIGReg (reading 1 met on its numeric criterion).** An untrained ResNet-18 + projector reaches same-view R^2 = 0.973 (calibrated BN, CI [0.971, 0.974]) and 0.999 (identity BN), against 0.947 for trained lambda 0.05. The trained-minus-random contrast is -0.026 [-0.030, -0.022] (lambda 0.02: -0.024, lambda 0.1: -0.037), inside +-0.05 and in fact slightly negative. The 0.95 headline is therefore a property of how norm and direction co-vary in this architecture (a within-view, deterministic map), not something SIGReg produces. BatchNorm mode changes the same-view value by 0.027 (< 0.1), so the calibrated group is the headline.
- **But the pre-registered stricter check does not agree, so reading 1 is not clean.** Across disjoint view halves the random-init R^2 is 0.37 (trained 0.86; contrast +0.50 [+0.47, +0.52]); single view to another view 0.10 (trained 0.70); split-half reliability of the mean log-radius 0.35 (trained 0.93). In the random-init network the radius is predictable from the direction *of the same view* but is mostly view-specific noise across augmentations; in the trained network it is an image-level quantity (ICC 0.87 in the sweep) that is also predictable across views. What training creates is cross-view stability of the radius, not predictability from the direction.
- Caveat: with three random-init seeds, only the image bootstrap enters the intervals (seed ranges are listed). Calibrated BN statistics come from augmented global views; the identity-BN group is a secondary bound. All on STL-10 / ResNet-18.

## Next steps
