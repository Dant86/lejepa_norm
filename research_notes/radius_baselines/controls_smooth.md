# E2 (part 1): what does R^2 = 0.95 mean? Random smooth functions of the direction
- **Branch / version:** `radius-baselines-controls` @ `<commit>` (filled in with the results commit; the reading below was committed before any job was submitted)
- **Run datetime (UTC):** <filled in with the results>
- **Command(s):** <filled in with the results>
- **Datasets:** STL-10 `test` clean images ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10)); the canonical archive was used (via the existing `stl10_test` embedding dumps).
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` of `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}`, and the random-init networks `random_init_bncal_s{0,1,2}`, `random_init_bnid_s{0,1,2}` (see `controls_redundancy.md`) for reference.
- **Scope:** h is not analyzed; LeJEPA's product is z.

## Question

The held-out kernel-ridge R^2 of the log-radius from the direction is about 0.95. What R^2 does the same regressor reach on a *random* function of the same directions with comparable smoothness (and noise)? This calibrates the number: is 0.95 a surprising amount of structure, or what any function of that smoothness gives?

## Setup (deviations from the base config)

No deviation from `configs/base.yaml`. On the same 4000 `stl10_test` images and the same inputs `u_hat` (normalized mean direction of the 8 augmented views) as the redundancy table, targets are drawn from a Gaussian-process prior on the sphere with the kernel family of the fit, `k(u, v) = exp((u . v - 1) / tau)` (`eval.baseline_controls.gp_draws`: eigendecomposition of the 4000 x 4000 kernel matrix, draws `V sqrt(lambda) xi`), 50 draws per family per run, standardized to the standard deviation of the real log-radius (R^2 is scale free; the scale only fixes the noise), and predicted by the identical nested-CV kernel ridge (`NestedKernelRidge.r2_many`: outer 5-fold, bandwidth and penalty re-selected per target by exact LOO). `tau_hat` is the median bandwidth the real fit selected over its outer folds. Families:

- `GP tau = m x tau_hat`, `m` in {0.25, 0.5, 1, 2, 4}, noiseless: the ceiling for functions of that smoothness, and how fast R^2 falls for rougher ones (`m < 1`);
- the same with iid Gaussian noise of fraction `1 - reliability` of the target variance, where the reliability of rho_bar is the Spearman-Brown step-up of the split-half correlation of the 4- vs 4-view means: functions **matched to the learned map** in smoothness (`m = 1`) and noise level (`GP tau=1x, matched noise` is the reference family);
- random linear functions `u . w` (Gaussian `w`), noiseless and with matched noise.

Intervals: 95% percentile bootstrap over the draws of a seed, averaged over the seeds of a group (seeds are not resampled). Caveat built into the design: the GP draws are samples from the very kernel family the regressor uses, so they are the *favorable* case for it; `tau_hat` is estimated from the real data. A real map that is as smooth as a GP draw is the null being calibrated, not a claim about how the network works.

## Pre-registered reading (written before any result was computed)

Compare `GP tau=1x, matched noise` (mean R^2 of random functions, trained runs) with the observed same-view R^2 of the same run:

1. **Observed within 0.05 of the matched random functions** (`|difference| <= 0.05`): R^2 ~ 0.95 is what a generic function of that smoothness and noise level yields. The number then measures the *smoothness* (`tau_hat`) of the radius-direction relation, not a special coupling; it does not by itself show that the radius is encoded in the direction in any structured way.
2. **Matched random functions reach more than observed + 0.05:** the real radius map is rougher or noisier than a generic function of its own estimated bandwidth, so 0.95 is not "too good to be generic".
3. **Matched random functions reach less than observed - 0.05:** the real map is easier to predict than a generic function of that bandwidth and noise level.
4. **Rough control:** if `GP tau=0.25x` (noiseless) reaches an R^2 more than 0.2 below the observed R^2, high R^2 is not automatic for any function of the directions (the regressor cannot fit anything), and the 4000 images constrain the fit. If it reaches the observed value too, the R^2 is uninformative about structure.
5. The noiseless m = 1 family is the ceiling attainable at that smoothness; the matched-noise family should fall below it by roughly the noise fraction. A real R^2 above the matched-noise family means the real radius is less noisy than its split-half reliability suggests.

## Results

## Interpretation

## Next steps
