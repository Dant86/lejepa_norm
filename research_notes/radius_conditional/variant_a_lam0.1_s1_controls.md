# variant_a_lam0.1_s1_controls
- **Branch / version:** `radius-conditional` @ `8f92ad3` (the cluster checkout the job ran from)
- **Run datetime (UTC):** 2026-10-22T20:48Z to 22:07Z (job 2232898); written up 2026-10-23
- **Command(s):**
  `sbatch --mem=32G slurm/analyze.sbatch configs/base.yaml radius_marginal_lam0.1_s1 radius_conditional/variant_a_lam0.1_s1_controls.md --redundancy-controls`
- **Datasets:** STL-10 `test` split, clean images only ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF](https://huggingface.co/datasets/tanganke/stl10)); canonical archive used; K = 8 augmented views plus the center view from the existing `stl10_test` dump
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); checkpoint `last.pt` of `radius_marginal_lam0.1_s1` (Variant A, epoch 200)

## Question

How much of the held-out R^2 of the radius from the direction (redundancy) survives a permutation null and cross-view regressions that share no view between input and target? This is one run; the seed aggregate with the interpretation is in `redundancy_controls_summary.md`.

## Setup (deviations from the base config)

Base config unchanged; `eval.num_permutations` and `eval.max_kernel_images` at their `configs/base.yaml` defaults (500 shuffles, 4000 images). Only the `stl10_test` dump is used.

## Results

<!-- BEGIN GENERATED: analyze.redundancy_controls -->
### Redundancy controls (run `radius_marginal_lam0.1_s1`)

Held-out R^2 of the radius from the direction (nested-CV kernel ridge, same images as the redundancy table) against a permutation null (500 shuffles of the radius through the identical pipeline; p = (1 + #{null >= observed}) / (1 + permutations)), with cross-view setups that share no view between input and target, and the classical OLS F-test (linear, in-sample). The reference row is the squared correlation of the two half-view means of rho_bar.

| space | setup | N | r2 (nested CV, kernel) | null mean | null sd | null max | permutations | permutation p | F | df | F p | r2 (in-sample, linear) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9363 | -0.001077 | 0.0008522 | 0.002097 | 500 | 0.001996 | 54.44 | 256, 3743 | 0 | 0.7883 |
| z | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.8347 | -0.001103 | 0.0008019 | 0.001091 | 500 | 0.001996 | 42.63 | 256, 3743 | 0 | 0.7446 |
| z | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.835 | -0.001021 | 0.0008381 | 0.003558 | 500 | 0.001996 | 43.57 | 256, 3743 | 0 | 0.7487 |
| z | single view, same view: rho_0 ~ u_0 | 4000 | 0.8957 | -0.001078 | 0.0008752 | 0.006754 | 500 | 0.001996 | 36.84 | 256, 3743 | 0 | 0.7159 |
| z | single view, other view: rho_1 ~ u_0 | 4000 | 0.6546 | -0.001005 | 0.0007885 | 0.001781 | 500 | 0.001996 | 24.69 | 256, 3743 | 0 | 0.6281 |
| z | reference: split-half reliability r^2 of rho_bar |  | 0.9177 |  |  |  |  |  |  |  |  |  |
| h | same views (all 8): rho_bar ~ u_hat | 4000 | 0.9365 | -0.001041 | 0.0008455 | 0.001539 | 500 | 0.001996 | 80.59 | 512, 3487 | 0 | 0.9221 |
| h | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 4000 | 0.827 | -0.001017 | 0.0007706 | 0.001797 | 500 | 0.001996 | 35.85 | 512, 3487 | 0 | 0.8404 |
| h | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 4000 | 0.8161 | -0.001051 | 0.0008409 | 0.005526 | 500 | 0.001996 | 34.71 | 512, 3487 | 0 | 0.836 |
| h | single view, same view: rho_0 ~ u_0 | 4000 | 0.885 | -0.00107 | 0.0008294 | 0.001834 | 500 | 0.001996 | 48.25 | 512, 3487 | 0 | 0.8763 |
| h | single view, other view: rho_1 ~ u_0 | 4000 | 0.6198 | -0.0009557 | 0.0008019 | 0.002563 | 500 | 0.001996 | 14.2 | 512, 3487 | 0 | 0.6759 |
| h | reference: split-half reliability r^2 of rho_bar |  | 0.8857 |  |  |  |  |  |  |  |  |  |
<!-- END GENERATED: analyze.redundancy_controls -->

## Interpretation

See `redundancy_controls_summary.md` (aggregate over the three seeds of each lambda).

## Next steps
