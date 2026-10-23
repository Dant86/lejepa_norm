# variant_a_lam0.1_s0_controls
- **Branch / version:** `radius-conditional` @ `8f92ad3` (the cluster checkout the job ran from)
- **Run datetime (UTC):** 2026-10-22T20:48Z to 22:08Z (job 2232897); written up 2026-10-23
- **Command(s):**
  `sbatch --mem=32G slurm/analyze.sbatch configs/base.yaml radius_marginal_lam0.1_s0 radius_conditional/variant_a_lam0.1_s0_controls.md --redundancy-controls`
- **Datasets:** STL-10 `test` split, clean images only ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF](https://huggingface.co/datasets/tanganke/stl10)); canonical archive used; K = 8 augmented views plus the center view from the existing `stl10_test` dump
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); checkpoint `last.pt` of `radius_marginal_lam0.1_s0` (Variant A, epoch 200)

## Question

How much of the held-out R^2 of the radius from the direction (redundancy) survives a permutation null and cross-view regressions that share no view between input and target? This is one run; the seed aggregate with the interpretation is in `redundancy_controls_summary.md`.

## Setup (deviations from the base config)

Base config unchanged; `eval.num_permutations` and `eval.max_kernel_images` at their `configs/base.yaml` defaults (500 shuffles, 4000 images). Only the `stl10_test` dump is used.

## Results

<!-- BEGIN GENERATED: analyze.redundancy_controls -->
### Redundancy controls (run `radius_marginal_lam0.1_s0`)

Held-out R^2 of the radius from the direction (nested-CV kernel ridge, same images as the redundancy table) against a permutation null (500 shuffles of the radius through the identical pipeline; p = (1 + #{null >= observed}) / (1 + permutations)), with cross-view setups that share no view between input and target, and the classical OLS F-test (linear, in-sample). The reference row is the squared correlation of the two half-view means of rho_bar.

| space | setup | N | r2 (nested CV, kernel) | null mean | null sd | null max | permutations | permutation p | F | df | F p | r2 (in-sample, linear) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9346 | -0.00108 | 0.000837 | 0.003267 | 500 | 0.001996 | 49.82 | 256, 3743 | 0 | 0.7731 |
| z | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8402 | -0.001102 | 0.0008388 | 0.002058 | 500 | 0.001996 | 40.9 | 256, 3743 | 0 | 0.7367 |
| z | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8377 | -0.001005 | 0.0008815 | 0.0044 | 500 | 0.001996 | 39.68 | 256, 3743 | 0 | 0.7307 |
| z | single view, same view: rho_0 ~ u_0 | 4000 | 0.901 | -0.001053 | 0.000864 | 0.003629 | 500 | 0.001996 | 36.55 | 256, 3743 | 0 | 0.7143 |
| z | single view, other view: rho_1 ~ u_0 | 4000 | 0.6425 | -0.001011 | 0.0007882 | 0.002362 | 500 | 0.001996 | 23 | 256, 3743 | 0 | 0.6114 |
| z | reference: split-half reliability r^2 of rho_bar |  | 0.9194 |  |  |  |  |  |  |  |  |  |
| h | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9387 | -0.001076 | 0.0008505 | 0.001903 | 500 | 0.001996 | 85.6 | 512, 3487 | 0 | 0.9263 |
| h | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8353 | -0.001035 | 0.0007868 | 0.002793 | 500 | 0.001996 | 39.03 | 512, 3487 | 0 | 0.8514 |
| h | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8327 | -0.00107 | 0.0008536 | 0.005663 | 500 | 0.001996 | 37.5 | 512, 3487 | 0 | 0.8463 |
| h | single view, same view: rho_0 ~ u_0 | 4000 | 0.8847 | -0.00102 | 0.0008505 | 0.002807 | 500 | 0.001996 | 50.18 | 512, 3487 | 0 | 0.8805 |
| h | single view, other view: rho_1 ~ u_0 | 4000 | 0.6248 | -0.0009752 | 0.0008815 | 0.005612 | 500 | 0.001996 | 14.32 | 512, 3487 | 0 | 0.6776 |
| h | reference: split-half reliability r^2 of rho_bar |  | 0.8921 |  |  |  |  |  |  |  |  |  |
<!-- END GENERATED: analyze.redundancy_controls -->

## Interpretation

See `redundancy_controls_summary.md` (aggregate over the three seeds of each lambda).

## Next steps
