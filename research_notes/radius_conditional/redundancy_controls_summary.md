# redundancy_controls_summary
- **Branch / version:** `radius-conditional` @ `8f92ad3` for the six per-run jobs; aggregation with `feat/aggregate-controls` @ `54a9119` (not yet on `master`)
- **Run datetime (UTC):** 2026-10-23T13:40Z (aggregation, seconds; the per-run jobs 2232894-2232899 ran 2026-10-22T20:48Z-22:59Z)
- **Command(s):**
  `python apps/analyze.py --aggregate-controls radius_conditional/variant_a_lam0.05_s{0,1,2}_controls.md radius_conditional/variant_a_lam0.1_s{0,1,2}_controls.md --note radius_conditional/redundancy_controls_summary.md` (brace expansion by the shell; per-run commands are in the individual `*_controls.md` notes)
- **Datasets:** STL-10 `test` split, clean images ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF](https://huggingface.co/datasets/tanganke/stl10)); canonical archive used
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` (epoch 200) of the six Variant A runs, lambda in {0.05, 0.1}, seeds {0, 1, 2}

## Question

Is the redundancy of the radius with the direction (held-out R^2 of `rho_bar` from `u_hat`) real, reproducible across seeds, and not an artifact of shared views? Primary space: `z` (the projector output); `h` rows are reported for completeness.

## Setup (deviations from the base config)

Base config; 4000 `stl10_test` images, K = 8 augmented views, nested-CV kernel ridge, 500 permutations per setup. Table cells aggregate the generated sections of the six per-run notes (nothing was recomputed); the six runs are the Variant A lambda = 0.05 and 0.1 checkpoints. lambda = 0.02 was not part of this study.

## Results

<!-- BEGIN GENERATED: analyze.redundancy_controls_aggregate -->
### Redundancy controls: mean ± std over seeds, per lambda

Runs: `radius_marginal_lam0.05_s0`, `radius_marginal_lam0.05_s1`, `radius_marginal_lam0.05_s2`, `radius_marginal_lam0.1_s0`, `radius_marginal_lam0.1_s1`, `radius_marginal_lam0.1_s2`. Per-run tables are in the individual controls notes; this table only aggregates them. `R^2` is the held-out nested-CV kernel-ridge value; `null max`, `worst permutation p` and `min F` are the least favorable value over seeds (the permutation p-value cannot go below 1 / (1 + permutations)).

| lambda | space | setup | seeds | r2 (nested CV, kernel) | null max (max over seeds) | worst permutation p | min F |
|---|---|---|---|---|---|---|---|
| 0.05 | z | same views (all 8): rho_bar ~ u_hat | 3 | 0.9467 ± 0.0014 | 0.002633 | 0.001996 | 72.19 |
| 0.05 | z | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.8612 ± 0.0041 | 0.002686 | 0.001996 | 54.69 |
| 0.05 | z | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.8602 ± 0.0059 | 0.003703 | 0.001996 | 55.33 |
| 0.05 | z | single view, same view: rho_0 ~ u_0 | 3 | 0.9182 ± 0.00065 | 0.004124 | 0.001996 | 51.65 |
| 0.05 | z | single view, other view: rho_1 ~ u_0 | 3 | 0.6979 ± 0.0091 | 0.004897 | 0.001996 | 29.08 |
| 0.05 | z | reference: split-half reliability r^2 of rho_bar | 3 | 0.9286 ± 0.0016 |  |  |  |
| 0.05 | h | same views (all 8): rho_bar ~ u_hat | 3 | 0.9255 ± 0.005 | 0.003606 | 0.001996 | 72.78 |
| 0.05 | h | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.8079 ± 0.014 | 0.002988 | 0.001996 | 30.9 |
| 0.05 | h | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.8044 ± 0.0096 | 0.005661 | 0.001996 | 31.24 |
| 0.05 | h | single view, same view: rho_0 ~ u_0 | 3 | 0.8564 ± 0.011 | 0.0039 | 0.001996 | 38.68 |
| 0.05 | h | single view, other view: rho_1 ~ u_0 | 3 | 0.5802 ± 0.0092 | 0.005429 | 0.001996 | 11.84 |
| 0.05 | h | reference: split-half reliability r^2 of rho_bar | 3 | 0.8693 ± 0.0098 |  |  |  |
| 0.1 | z | same views (all 8): rho_bar ~ u_hat | 3 | 0.9355 ± 0.00086 | 0.003358 | 0.001996 | 49.82 |
| 0.1 | z | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.8381 ± 0.003 | 0.002058 | 0.001996 | 40.9 |
| 0.1 | z | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.8363 ± 0.0014 | 0.0044 | 0.001996 | 39.68 |
| 0.1 | z | single view, same view: rho_0 ~ u_0 | 3 | 0.8992 ± 0.0031 | 0.006754 | 0.001996 | 36.55 |
| 0.1 | z | single view, other view: rho_1 ~ u_0 | 3 | 0.6516 ± 0.008 | 0.004319 | 0.001996 | 23 |
| 0.1 | z | reference: split-half reliability r^2 of rho_bar | 3 | 0.9185 ± 0.00085 |  |  |  |
| 0.1 | h | same views (all 8): rho_bar ~ u_hat | 3 | 0.9371 ± 0.0014 | 0.005508 | 0.001996 | 80.59 |
| 0.1 | h | disjoint halves: rho_bar(views 4-7) ~ u_hat(views 0-3) | 3 | 0.8311 ± 0.0042 | 0.003194 | 0.001996 | 35.85 |
| 0.1 | h | disjoint halves: rho_bar(views 0-3) ~ u_hat(views 4-7) | 3 | 0.826 ± 0.0087 | 0.005663 | 0.001996 | 34.71 |
| 0.1 | h | single view, same view: rho_0 ~ u_0 | 3 | 0.884 ± 0.0015 | 0.002942 | 0.001996 | 48.25 |
| 0.1 | h | single view, other view: rho_1 ~ u_0 | 3 | 0.6281 ± 0.01 | 0.005612 | 0.001996 | 14.2 |
| 0.1 | h | reference: split-half reliability r^2 of rho_bar | 3 | 0.8901 ± 0.0038 |  |  |  |
<!-- END GENERATED: analyze.redundancy_controls_aggregate -->

## Interpretation

- **The dependence is real.** In every run and setup the held-out R^2 (0.58 or more) is far above the permutation null (mean about -0.001, largest null value 0.0057 in any setup); all permutation p-values sit at the floor of 1/501, and the OLS F statistics are 11.8 or more on 256 or 512 numerator df.
- **Not a shared-view artifact (z).** With disjoint view halves the R^2 is 0.861 +- 0.004 / 0.860 +- 0.006 (lambda = 0.05) and 0.838 +- 0.003 / 0.836 +- 0.001 (lambda = 0.1), against split-half reliabilities of the radius of 0.929 and 0.919: about 93% and 91% of the radius variance that is stable across views is predicted by the direction from other views. A single other view still gives 0.70 and 0.65.
- **Effect of lambda (z).** Redundancy is higher at lambda = 0.05 than at 0.1 in every setup, by more than the seed spread (differences 0.01-0.05 against a std of at most 0.009). In `h` the ordering is the reverse: lambda = 0.1 is higher in all four setups (same-view 0.926 vs 0.937, disjoint 0.806 vs 0.829, single-view 0.58 vs 0.63; seed std up to 0.014), as is its split-half reliability (0.869 vs 0.890). `h` is secondary here and the reversal is not investigated.
- **What this does not show.** Redundancy means the direction predicts the radius; it does not show the radius is unused, nor that the unexplained variance (5-6% same-view, 14-16% with disjoint views, in z) is uninformative. Two lambdas and three seeds each; 4000 images; a permutation p-value cannot go below 1/501. The go/no-go thresholds of the Technical Design are not evaluated here.

## Next steps

- Test whether the radius is *used* (z-space Euclidean k-NN with radii randomized, including a residual-only shuffle that keeps the direction-predicted part); to be specified as its own research question.
