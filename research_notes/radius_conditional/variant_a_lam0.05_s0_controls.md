# variant_a_lam0.05_s0_controls
- **Branch / version:** `radius-conditional` @ `8f92ad3` (the cluster checkout the job ran from)
- **Run datetime (UTC):** 2026-10-22T20:48Z to 21:58Z (job 2232894); written up 2026-10-23
- **Command(s):**
  `sbatch --mem=32G slurm/analyze.sbatch configs/base.yaml radius_marginal_lam0.05_s0 radius_conditional/variant_a_lam0.05_s0_controls.md --redundancy-controls`
- **Datasets:** STL-10 `test` split, clean images only ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF](https://huggingface.co/datasets/tanganke/stl10)); canonical archive used; K = 8 augmented views plus the center view from the existing `stl10_test` dump
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); checkpoint `last.pt` of `radius_marginal_lam0.05_s0` (Variant A, epoch 200)

## Question

How much of the held-out R^2 of the radius from the direction (redundancy) survives a permutation null and cross-view regressions that share no view between input and target? This is one run; the seed aggregate with the interpretation is in `redundancy_controls_summary.md`.

## Setup (deviations from the base config)

Base config unchanged; `eval.num_permutations` and `eval.max_kernel_images` at their `configs/base.yaml` defaults (500 shuffles, 4000 images). Only the `stl10_test` dump is used.

## Results

<!-- BEGIN GENERATED: analyze.redundancy_controls -->
### Redundancy controls (run `radius_marginal_lam0.05_s0`)

Held-out R^2 of the radius from the direction (nested-CV kernel ridge, same images as the redundancy table) against a permutation null (500 shuffles of the radius through the identical pipeline; p = (1 + #{null >= observed}) / (1 + permutations)), with cross-view setups that share no view between input and target, and the classical OLS F-test (linear, in-sample). The reference row is the squared correlation of the two half-view means of rho_bar.

| space | setup | N | r2 (nested CV, kernel) | null mean | null sd | null max | permutations | permutation p | F | df | F p | r2 (in-sample, linear) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9459 | -0.001127 | 0.0008805 | 0.002633 | 500 | 0.001996 | 78.19 | 256, 3743 | 0 | 0.8425 |
| z | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8572 | -0.001141 | 0.0008401 | 0.001586 | 500 | 0.001996 | 58.57 | 256, 3743 | 0 | 0.8002 |
| z | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8597 | -0.001073 | 0.0008897 | 0.003703 | 500 | 0.001996 | 55.57 | 256, 3743 | 0 | 0.7917 |
| z | single view, same view: rho_0 ~ u_0 | 4000 | 0.9182 | -0.001131 | 0.0008578 | 0.004124 | 500 | 0.001996 | 55.94 | 256, 3743 | 0 | 0.7928 |
| z | single view, other view: rho_1 ~ u_0 | 4000 | 0.6913 | -0.001079 | 0.000822 | 0.001767 | 500 | 0.001996 | 30.29 | 256, 3743 | 0 | 0.6744 |
| z | reference: split-half reliability r^2 of rho_bar |  | 0.9276 |  |  |  |  |  |  |  |  |  |
| h | same views (all 8): rho_bar ~ u_hat | 4000 | 0.925 | -0.001108 | 0.0008214 | 0.001475 | 500 | 0.001996 | 72.78 | 512, 3487 | 0 | 0.9144 |
| h | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.7927 | -0.00103 | 0.0008216 | 0.002833 | 500 | 0.001996 | 30.9 | 512, 3487 | 0 | 0.8194 |
| h | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.7948 | -0.001087 | 0.0008284 | 0.004022 | 500 | 0.001996 | 31.24 | 512, 3487 | 0 | 0.821 |
| h | single view, same view: rho_0 ~ u_0 | 4000 | 0.8549 | -0.00106 | 0.000849 | 0.002193 | 500 | 0.001996 | 40.55 | 512, 3487 | 0 | 0.8562 |
| h | single view, other view: rho_1 ~ u_0 | 4000 | 0.5719 | -0.000922 | 0.000881 | 0.005429 | 500 | 0.001996 | 11.84 | 512, 3487 | 0 | 0.6348 |
| h | reference: split-half reliability r^2 of rho_bar |  | 0.8581 |  |  |  |  |  |  |  |  |  |
<!-- END GENERATED: analyze.redundancy_controls -->

## Interpretation

See `redundancy_controls_summary.md` (aggregate over the three seeds of each lambda).

## Next steps
