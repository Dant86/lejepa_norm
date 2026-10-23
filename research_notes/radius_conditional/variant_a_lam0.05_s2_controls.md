# variant_a_lam0.05_s2_controls
- **Branch / version:** `radius-conditional` @ `8f92ad3` (the cluster checkout the job ran from)
- **Run datetime (UTC):** 2026-10-22T20:48Z to 22:07Z (job 2232896); written up 2026-10-23
- **Command(s):**
  `sbatch --mem=32G slurm/analyze.sbatch configs/base.yaml radius_marginal_lam0.05_s2 radius_conditional/variant_a_lam0.05_s2_controls.md --redundancy-controls`
- **Datasets:** STL-10 `test` split, clean images only ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF](https://huggingface.co/datasets/tanganke/stl10)); canonical archive used; K = 8 augmented views plus the center view from the existing `stl10_test` dump
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); checkpoint `last.pt` of `radius_marginal_lam0.05_s2` (Variant A, epoch 200)

## Question

How much of the held-out R^2 of the radius from the direction (redundancy) survives a permutation null and cross-view regressions that share no view between input and target? This is one run; the seed aggregate with the interpretation is in `redundancy_controls_summary.md`.

## Setup (deviations from the base config)

Base config unchanged; `eval.num_permutations` and `eval.max_kernel_images` at their `configs/base.yaml` defaults (500 shuffles, 4000 images). Only the `stl10_test` dump is used.

## Results

<!-- BEGIN GENERATED: analyze.redundancy_controls -->
### Redundancy controls (run `radius_marginal_lam0.05_s2`)

Held-out R^2 of the radius from the direction (nested-CV kernel ridge, same images as the redundancy table) against a permutation null (500 shuffles of the radius through the identical pipeline; p = (1 + #{null >= observed}) / (1 + permutations)), with cross-view setups that share no view between input and target, and the classical OLS F-test (linear, in-sample). The reference row is the squared correlation of the two half-view means of rho_bar.

| space | setup | N | r2 (nested CV, kernel) | null mean | null sd | null max | permutations | permutation p | F | df | F p | r2 (in-sample, linear) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9459 | -0.001116 | 0.0009149 | 0.00222 | 500 | 0.001996 | 84.41 | 256, 3743 | 0 | 0.8524 |
| z | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8609 | -0.001133 | 0.0008876 | 0.002686 | 500 | 0.001996 | 62.26 | 256, 3743 | 0 | 0.8098 |
| z | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8546 | -0.001082 | 0.0008869 | 0.002919 | 500 | 0.001996 | 59.7 | 256, 3743 | 0 | 0.8033 |
| z | single view, same view: rho_0 ~ u_0 | 4000 | 0.9189 | -0.00112 | 0.000862 | 0.002541 | 500 | 0.001996 | 58.22 | 256, 3743 | 0 | 0.7993 |
| z | single view, other view: rho_1 ~ u_0 | 4000 | 0.6942 | -0.001059 | 0.0008626 | 0.003401 | 500 | 0.001996 | 29.08 | 256, 3743 | 0 | 0.6654 |
| z | reference: split-half reliability r^2 of rho_bar |  | 0.9277 |  |  |  |  |  |  |  |  |  |
| h | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9208 | -0.001081 | 0.0008481 | 0.003606 | 500 | 0.001996 | 74.76 | 512, 3487 | 0 | 0.9165 |
| h | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8103 | -0.001018 | 0.0007956 | 0.002639 | 500 | 0.001996 | 34.17 | 512, 3487 | 0 | 0.8338 |
| h | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8044 | -0.001127 | 0.0008266 | 0.002842 | 500 | 0.001996 | 33.37 | 512, 3487 | 0 | 0.8305 |
| h | single view, same view: rho_0 ~ u_0 | 4000 | 0.8458 | -0.001021 | 0.0007877 | 0.002941 | 500 | 0.001996 | 38.68 | 512, 3487 | 0 | 0.8503 |
| h | single view, other view: rho_1 ~ u_0 | 4000 | 0.5786 | -0.0009017 | 0.0008818 | 0.004389 | 500 | 0.001996 | 12.25 | 512, 3487 | 0 | 0.6427 |
| h | reference: split-half reliability r^2 of rho_bar |  | 0.8737 |  |  |  |  |  |  |  |  |  |
<!-- END GENERATED: analyze.redundancy_controls -->

## Interpretation

See `redundancy_controls_summary.md` (aggregate over the three seeds of each lambda).

## Next steps
