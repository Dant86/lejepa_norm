# E2 (part 1): what does R^2 = 0.95 mean? Random smooth functions of the direction
- **Branch / version:** `radius-baselines-controls` @ `567c498` (code and pre-registered readings; this results commit follows it)
- **Run datetime (UTC):** 2026-10-26 (jobs started 2026-10-26T01:11Z by sacct; aggregation 2026-10-26T02:11Z)
- **Command(s):** `sbatch --export=ALL --array=0-5 slurm/embed_random_init.sbatch configs/base.yaml` (job 2247669, random-init embeddings, 1 GPU each, node r003: NVIDIA H200); `sbatch --export=ALL --array=0-14%8 slurm/baseline_controls.sbatch configs/base.yaml radius_marginal random_init` (job 2247779, CPU only, nodes m002/g002/h001); aggregation: `python apps/analyze.py --config configs/base.yaml --baseline-controls radius_marginal random_init --note radius_baselines/controls.md` (writes `controls_redundancy.md`, `controls_smooth.md`, `controls_curve.md`)
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

<!-- BEGIN GENERATED: baseline.smooth -->
### Held-out R^2 of random smooth functions of the direction

Targets are drawn on the real `u_hat` of the same images and predicted by the identical nested-CV kernel ridge. `tau_hat` is the median bandwidth the fit selected for the real log-radius; `noise fraction` is 1 - (Spearman-Brown reliability of rho_bar). The interval is a 95% percentile bootstrap over draws (seeds averaged).

| group | family | seeds | r2 of random functions | 95% CI (draws) | draw sd (mean over seeds) | observed r2 (same views) | share of draws >= observed | tau_hat | noise fraction |
|---|---|---|---|---|---|---|---|---|---|
| radius_marginal_lam0.02 | GP tau=0.25x, noiseless | 3 | 0.9702 | [0.9690, 0.9713] | 0.006538 | 0.9484 | 0.92 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=0.25x, matched noise | 3 | 0.9442 | [0.9430, 0.9454] | 0.007294 | 0.9484 | 0.3467 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=0.5x, noiseless | 3 | 0.9929 | [0.9925, 0.9932] | 0.002006 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=0.5x, matched noise | 3 | 0.9703 | [0.9699, 0.9708] | 0.002747 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=1x, noiseless | 3 | 0.9981 | [0.9980, 0.9982] | 0.000546 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=1x, matched noise | 3 | 0.9781 | [0.9779, 0.9783] | 0.001237 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=2x, noiseless | 3 | 0.9994 | [0.9994, 0.9995] | 0.0002033 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=2x, matched noise | 3 | 0.9812 | [0.9811, 0.9813] | 0.0006955 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=4x, noiseless | 3 | 0.9998 | [0.9998, 0.9998] | 5.519e-05 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | GP tau=4x, matched noise | 3 | 0.9825 | [0.9824, 0.9826] | 0.0003663 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | random linear, noiseless | 3 | 1 | [1.0000, 1.0000] | 1.047e-05 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.02 | random linear, matched noise | 3 | 0.9842 | [0.9842, 0.9842] | 0.0001656 | 0.9484 | 1 | 2.824 | 0.01506 |
| radius_marginal_lam0.05 | GP tau=0.25x, noiseless | 3 | 0.959 | [0.9578, 0.9602] | 0.007459 | 0.9467 | 0.9467 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=0.25x, matched noise | 3 | 0.9282 | [0.9269, 0.9294] | 0.008031 | 0.9467 | 0 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=0.5x, noiseless | 3 | 0.9867 | [0.9862, 0.9871] | 0.00308 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=0.5x, matched noise | 3 | 0.9579 | [0.9573, 0.9585] | 0.004005 | 0.9467 | 0.9933 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=1x, noiseless | 3 | 0.9951 | [0.9949, 0.9953] | 0.001105 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=1x, matched noise | 3 | 0.9685 | [0.9683, 0.9688] | 0.001822 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=2x, noiseless | 3 | 0.9981 | [0.9980, 0.9982] | 0.0004823 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=2x, matched noise | 3 | 0.9734 | [0.9732, 0.9736] | 0.001269 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=4x, noiseless | 3 | 0.9991 | [0.9991, 0.9992] | 0.000161 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | GP tau=4x, matched noise | 3 | 0.976 | [0.9759, 0.9761] | 0.0006415 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | random linear, noiseless | 3 | 0.9999 | [0.9999, 0.9999] | 1.874e-05 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.05 | random linear, matched noise | 3 | 0.9798 | [0.9798, 0.9798] | 0.000262 | 0.9467 | 1 | 4.143 | 0.01853 |
| radius_marginal_lam0.1 | GP tau=0.25x, noiseless | 3 | 0.9171 | [0.9151, 0.9191] | 0.01177 | 0.9355 | 0.06 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=0.25x, matched noise | 3 | 0.8829 | [0.8810, 0.8849] | 0.01173 | 0.9355 | 0 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=0.5x, noiseless | 3 | 0.9677 | [0.9667, 0.9686] | 0.005886 | 0.9355 | 1 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=0.5x, matched noise | 3 | 0.9337 | [0.9326, 0.9348] | 0.006833 | 0.9355 | 0.3867 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=1x, noiseless | 3 | 0.9858 | [0.9854, 0.9862] | 0.002571 | 0.9355 | 1 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=1x, matched noise | 3 | 0.9535 | [0.9530, 0.9540] | 0.003268 | 0.9355 | 1 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=2x, noiseless | 3 | 0.9936 | [0.9934, 0.9938] | 0.001255 | 0.9355 | 1 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=2x, matched noise | 3 | 0.9639 | [0.9636, 0.9643] | 0.002244 | 0.9355 | 1 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=4x, noiseless | 3 | 0.9969 | [0.9968, 0.9969] | 0.0004774 | 0.9355 | 1 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | GP tau=4x, matched noise | 3 | 0.9692 | [0.9690, 0.9693] | 0.0009905 | 0.9355 | 1 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | random linear, noiseless | 3 | 0.9999 | [0.9998, 0.9999] | 3.081e-05 | 0.9355 | 1 | 4.096 | 0.02124 |
| radius_marginal_lam0.1 | random linear, matched noise | 3 | 0.9758 | [0.9757, 0.9758] | 0.00036 | 0.9355 | 1 | 4.096 | 0.02124 |
| random_init_bncal | GP tau=0.25x, noiseless | 3 | 0.7016 | [0.6934, 0.7092] | 0.04816 | 0.9726 | 0 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=0.25x, matched noise | 3 | 0.4733 | [0.4662, 0.4797] | 0.04107 | 0.9726 | 0 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=0.5x, noiseless | 3 | 0.8609 | [0.8562, 0.8661] | 0.02948 | 0.9726 | 0 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=0.5x, matched noise | 3 | 0.5877 | [0.5833, 0.5927] | 0.02824 | 0.9726 | 0 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=1x, noiseless | 3 | 0.936 | [0.9329, 0.9390] | 0.01861 | 0.9726 | 0.09333 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=1x, matched noise | 3 | 0.6475 | [0.6440, 0.6509] | 0.02124 | 0.9726 | 0 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=2x, noiseless | 3 | 0.9704 | [0.9690, 0.9718] | 0.008743 | 0.9726 | 0.4467 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=2x, matched noise | 3 | 0.6801 | [0.6781, 0.6821] | 0.01274 | 0.9726 | 0 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=4x, noiseless | 3 | 0.9864 | [0.9857, 0.9871] | 0.004389 | 0.9726 | 0.98 | 1.467 | 0.2571 |
| random_init_bncal | GP tau=4x, matched noise | 3 | 0.6988 | [0.6975, 0.7002] | 0.008828 | 0.9726 | 0 | 1.467 | 0.2571 |
| random_init_bncal | random linear, noiseless | 3 | 0.9998 | [0.9998, 0.9998] | 6.853e-05 | 0.9726 | 1 | 1.467 | 0.2571 |
| random_init_bncal | random linear, matched noise | 3 | 0.7156 | [0.7149, 0.7165] | 0.005004 | 0.9726 | 0 | 1.467 | 0.2571 |
| random_init_bnid | GP tau=0.25x, noiseless | 3 | 0.9933 | [0.9927, 0.9939] | 0.003609 | 0.9993 | 0 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=0.25x, matched noise | 3 | 0.738 | [0.7362, 0.7396] | 0.01052 | 0.9993 | 0 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=0.5x, noiseless | 3 | 0.9974 | [0.9972, 0.9976] | 0.001173 | 0.9993 | 0.02667 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=0.5x, matched noise | 3 | 0.7481 | [0.7472, 0.7489] | 0.005504 | 0.9993 | 0 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=1x, noiseless | 3 | 0.9987 | [0.9986, 0.9988] | 0.0008265 | 0.9993 | 0.3133 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=1x, matched noise | 3 | 0.7523 | [0.7515, 0.7532] | 0.005892 | 0.9993 | 0 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=2x, noiseless | 3 | 0.9992 | [0.9991, 0.9993] | 0.0005447 | 0.9993 | 0.52 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=2x, matched noise | 3 | 0.7551 | [0.7542, 0.7558] | 0.004855 | 0.9993 | 0 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=4x, noiseless | 3 | 0.9994 | [0.9993, 0.9995] | 0.0004966 | 0.9993 | 0.7 | 0.01898 | 0.234 |
| random_init_bnid | GP tau=4x, matched noise | 3 | 0.7564 | [0.7555, 0.7573] | 0.005311 | 0.9993 | 0 | 0.01898 | 0.234 |
| random_init_bnid | random linear, noiseless | 3 | 0.9996 | [0.9996, 0.9997] | 0.0004024 | 0.9993 | 0.8467 | 0.01898 | 0.234 |
| random_init_bnid | random linear, matched noise | 3 | 0.7565 | [0.7557, 0.7572] | 0.005236 | 0.9993 | 0 | 0.01898 | 0.234 |
<!-- END GENERATED: baseline.smooth -->

## Interpretation

- **Reading 1 holds for the trained runs.** Observed same-view R^2 minus the matched-noise GP (tau = 1x): 0.948 vs 0.978 (lambda 0.02), 0.947 vs 0.969 (0.05), 0.936 vs 0.954 (0.1), differences -0.030, -0.022, -0.018, all within 0.05 and all *below* the random functions. A random smooth function of the directions with the same bandwidth and noise level is predicted as well as, or slightly better than, the real log-radius. R^2 ~ 0.95 therefore measures smoothness of the radius-direction relation, not a special coupling.
- **Reading 4: the number is uninformative about structure.** `GP tau=0.25x` (noiseless) reaches 0.970 / 0.959 / 0.917, not more than 0.2 below the observed value; even a rough function is recovered by the regressor at N = 4000 (random linear functions: R^2 = 1.00). A high same-view R^2 is nearly automatic for any smooth function on these directions.
- **Random-init, calibrated BN (reading 3, with caution):** observed 0.973 against 0.648 for the matched-noise GP: the real map is much easier to predict than a generic function with the same noise fraction (0.26). That noise fraction (1 - Spearman-Brown reliability) mostly reflects view-to-view variation (see `controls_redundancy.md`), which a same-view regression does not see, so the matched-noise comparison is not meaningful for the random-init groups.
- Caveat: tau_hat is the median bandwidth selected for the real log-radius; the random families are defined relative to it, so the comparison is only as good as that estimate.

## Next steps
