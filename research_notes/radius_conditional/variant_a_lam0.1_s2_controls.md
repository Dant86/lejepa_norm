# variant_a_lam0.1_s2_controls
- **Branch / version:** `radius-conditional` @ `8f92ad3` (the cluster checkout the job ran from)
- **Run datetime (UTC):** 2026-10-22T20:48Z to 22:59Z (job 2232899); written up 2026-10-23
- **Command(s):**
  `sbatch --mem=32G slurm/analyze.sbatch configs/base.yaml radius_marginal_lam0.1_s2 radius_conditional/variant_a_lam0.1_s2_controls.md --redundancy-controls`
- **Datasets:** STL-10 `test` split, clean images only ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF](https://huggingface.co/datasets/tanganke/stl10)); canonical archive used; K = 8 augmented views plus the center view from the existing `stl10_test` dump
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); checkpoint `last.pt` of `radius_marginal_lam0.1_s2` (Variant A, epoch 200)

## Question

How much of the held-out R^2 of the radius from the direction (redundancy) survives a permutation null and cross-view regressions that share no view between input and target? This is one run; the seed aggregate with the interpretation is in `redundancy_controls_summary.md`.

## Setup (deviations from the base config)

Base config unchanged; `eval.num_permutations` and `eval.max_kernel_images` at their `configs/base.yaml` defaults (500 shuffles, 4000 images). Only the `stl10_test` dump is used.

## Results

<!-- BEGIN GENERATED: analyze.redundancy_controls -->
### Redundancy controls (run `radius_marginal_lam0.1_s2`)

Held-out R^2 of the radius from the direction (nested-CV kernel ridge, same images as the redundancy table) against a permutation null (500 shuffles of the radius through the identical pipeline; p = (1 + #{null >= observed}) / (1 + permutations)), with cross-view setups that share no view between input and target, and the classical OLS F-test (linear, in-sample). The reference row is the squared correlation of the two half-view means of rho_bar.

| space | setup | N | r2 (nested CV, kernel) | null mean | null sd | null max | permutations | permutation p | F | df | F p | r2 (in-sample, linear) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9357 | -0.001089 | 0.0009057 | 0.003358 | 500 | 0.001996 | 54.22 | 256, 3743 | 0 | 0.7876 |
| z | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8393 | -0.001102 | 0.0008192 | 0.001674 | 500 | 0.001996 | 42.46 | 256, 3743 | 0 | 0.7438 |
| z | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8361 | -0.001012 | 0.0008336 | 0.004092 | 500 | 0.001996 | 42.64 | 256, 3743 | 0 | 0.7447 |
| z | single view, same view: rho_0 ~ u_0 | 4000 | 0.901 | -0.00107 | 0.0008046 | 0.004273 | 500 | 0.001996 | 39.83 | 256, 3743 | 0 | 0.7315 |
| z | single view, other view: rho_1 ~ u_0 | 4000 | 0.6577 | -0.001009 | 0.0008519 | 0.004319 | 500 | 0.001996 | 24.31 | 256, 3743 | 0 | 0.6244 |
| z | reference: split-half reliability r^2 of rho_bar |  | 0.9185 |  |  |  |  |  |  |  |  |  |
| h | same views (all 8): rho_bar ~ u_hat | 4000 | 0.936 | -0.001014 | 0.0008187 | 0.005508 | 500 | 0.001996 | 89.59 | 512, 3487 | 0 | 0.9293 |
| h | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8311 | -0.001034 | 0.0008353 | 0.003194 | 500 | 0.001996 | 39.36 | 512, 3487 | 0 | 0.8525 |
| h | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8291 | -0.001046 | 0.0007789 | 0.002154 | 500 | 0.001996 | 39.16 | 512, 3487 | 0 | 0.8519 |
| h | single view, same view: rho_0 ~ u_0 | 4000 | 0.8822 | -0.00104 | 0.0008462 | 0.002942 | 500 | 0.001996 | 50.33 | 512, 3487 | 0 | 0.8808 |
| h | single view, other view: rho_1 ~ u_0 | 4000 | 0.6397 | -0.0009426 | 0.0007774 | 0.002989 | 500 | 0.001996 | 15.2 | 512, 3487 | 0 | 0.6906 |
| h | reference: split-half reliability r^2 of rho_bar |  | 0.8926 |  |  |  |  |  |  |  |  |  |
<!-- END GENERATED: analyze.redundancy_controls -->

## Interpretation

See `redundancy_controls_summary.md` (aggregate over the three seeds of each lambda).

## Next steps
