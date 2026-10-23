# variant_a_lam0.05_s1_controls
- **Branch / version:** `radius-conditional` @ `8f92ad3` (the cluster checkout the job ran from)
- **Run datetime (UTC):** 2026-10-22T20:48Z to 22:10Z (job 2232895); written up 2026-10-23
- **Command(s):**
  `sbatch --mem=32G slurm/analyze.sbatch configs/base.yaml radius_marginal_lam0.05_s1 radius_conditional/variant_a_lam0.05_s1_controls.md --redundancy-controls`
- **Datasets:** STL-10 `test` split, clean images only ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF](https://huggingface.co/datasets/tanganke/stl10)); canonical archive used; K = 8 augmented views plus the center view from the existing `stl10_test` dump
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); checkpoint `last.pt` of `radius_marginal_lam0.05_s1` (Variant A, epoch 200)

## Question

How much of the held-out R^2 of the radius from the direction (redundancy) survives a permutation null and cross-view regressions that share no view between input and target? This is one run; the seed aggregate with the interpretation is in `redundancy_controls_summary.md`.

## Setup (deviations from the base config)

Base config unchanged; `eval.num_permutations` and `eval.max_kernel_images` at their `configs/base.yaml` defaults (500 shuffles, 4000 images). Only the `stl10_test` dump is used.

## Results

<!-- BEGIN GENERATED: analyze.redundancy_controls -->
### Redundancy controls (run `radius_marginal_lam0.05_s1`)

Held-out R^2 of the radius from the direction (nested-CV kernel ridge, same images as the redundancy table) against a permutation null (500 shuffles of the radius through the identical pipeline; p = (1 + #{null >= observed}) / (1 + permutations)), with cross-view setups that share no view between input and target, and the classical OLS F-test (linear, in-sample). The reference row is the squared correlation of the two half-view means of rho_bar.

| space | setup | N | r2 (nested CV, kernel) | null mean | null sd | null max | permutations | permutation p | F | df | F p | r2 (in-sample, linear) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9484 | -0.00113 | 0.0008832 | 0.00182 | 500 | 0.001996 | 72.19 | 256, 3743 | 0 | 0.8316 |
| z | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8654 | -0.001132 | 0.0008478 | 0.001802 | 500 | 0.001996 | 54.69 | 256, 3743 | 0 | 0.7891 |
| z | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8664 | -0.001037 | 0.0008833 | 0.003667 | 500 | 0.001996 | 55.33 | 256, 3743 | 0 | 0.791 |
| z | single view, same view: rho_0 ~ u_0 | 4000 | 0.9176 | -0.00111 | 0.0008632 | 0.003445 | 500 | 0.001996 | 51.65 | 256, 3743 | 0 | 0.7794 |
| z | single view, other view: rho_1 ~ u_0 | 4000 | 0.7083 | -0.00107 | 0.0008468 | 0.004897 | 500 | 0.001996 | 29.84 | 256, 3743 | 0 | 0.6712 |
| z | reference: split-half reliability r^2 of rho_bar |  | 0.9304 |  |  |  |  |  |  |  |  |  |
| h | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9308 | -0.001077 | 0.0008755 | 0.002946 | 500 | 0.001996 | 79.78 | 512, 3487 | 0 | 0.9214 |
| h | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8207 | -0.001024 | 0.0008163 | 0.002988 | 500 | 0.001996 | 35.57 | 512, 3487 | 0 | 0.8393 |
| h | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.814 | -0.001097 | 0.0008676 | 0.005661 | 500 | 0.001996 | 34.73 | 512, 3487 | 0 | 0.8361 |
| h | single view, same view: rho_0 ~ u_0 | 4000 | 0.8684 | -0.001029 | 0.0008101 | 0.0039 | 500 | 0.001996 | 43.06 | 512, 3487 | 0 | 0.8634 |
| h | single view, other view: rho_1 ~ u_0 | 4000 | 0.59 | -0.000947 | 0.0008386 | 0.004354 | 500 | 0.001996 | 12.95 | 512, 3487 | 0 | 0.6554 |
| h | reference: split-half reliability r^2 of rho_bar |  | 0.8762 |  |  |  |  |  |  |  |  |  |
<!-- END GENERATED: analyze.redundancy_controls -->

## Interpretation

See `redundancy_controls_summary.md` (aggregate over the three seeds of each lambda).

## Next steps
