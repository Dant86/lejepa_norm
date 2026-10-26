# E2 (part 1): is N = 4000 saturating? Learning curve of the redundancy R^2
- **Branch / version:** `radius-baselines-controls` @ `<commit>` (filled in with the results commit; the margin and the reading below were committed before any job was submitted)
- **Run datetime (UTC):** <filled in with the results>
- **Command(s):** <filled in with the results>
- **Datasets:** STL-10 `train` (5000 labeled images) and `test` (8000 images) clean images, pooled to 13000 ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10)); the canonical archive was used (via the existing `stl10_train` / `stl10_test` embedding dumps). The 100k `unlabeled` images have no embeddings (the dump layout covers five evaluation sets, not the pretraining images), so N is capped at 11000 training images; nothing is extrapolated beyond it.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` of `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}`, and the random-init networks of `controls_redundancy.md`.
- **Scope:** h is not analyzed; LeJEPA's product is z.

## Question

All redundancy numbers use N = 4000 images (nested 5-fold CV, so 3200 training images per fit). Is that saturating, i.e. does the held-out R^2 of the radius from the direction stop growing with the number of training images, or would more images give a (much) higher R^2?

## Setup (deviations from the base config)

No deviation from `configs/base.yaml`. Pool: the `stl10_train` and `stl10_test` dumps, K = 8 augmented views each (13000 images). One fixed random held-out set of `eval.curve_holdout = 2000` images (the same images for every run: seed 0, independent of the run); training sets of N in {250, 500, 1000, 2000, 4000, 8000, 11000} images are random nested subsets of the other 11000 images (5 draws for N <= 1000, 3 for 2000 and 4000, 2 for 8000, the whole remainder once). Fit: `redundancy.FittedKernelRidge` on `u -> rho_bar` (exp-dot kernel, bandwidth and penalty selected by exact LOO on the training rows, as inside one outer fold of the nested CV in the redundancy table). Deviation for compute (an 11000 x 11000 eigendecomposition per bandwidth is about 10 minutes): for N = 8000 and 11000 the bandwidth is the one selected at N = 4000 on the same nested training draw and only the penalty is re-selected by exact LOO at the larger N. If the best bandwidth shrinks with N this slightly understates R^2 at large N, i.e. it can bias the contrast toward "saturated". Two setups: `same views` (u_hat and rho_bar both from all 8 views, the headline setup) and `disjoint halves` (u_hat of views 0-3 to rho_bar of views 4-7, the setup with no shared view). R^2 is on the held-out set; for N with several draws the per-image squared errors are averaged over draws before the bootstrap (images of the held-out set resampled within each seed, same resample for every N, then averaged over seeds; 1000 replicates). Note the difference to the redundancy table: its N = 4000 CV trains on 3200 images per fold and tests on the remaining 800 of the same images; here N = 4000 trains on 4000 and tests on 2000 different images, so values at N = 4000 differ slightly (more training images; a different, larger test set).

**Margin (fixed before any curve was computed, `eval.curve_margin` in `configs/base.yaml` and `lejepa_norm.config`):** 0.01 in R^2 units. Saturation contrast: `R^2(N) - R^2(4000)`, paired over held-out images, equivalence verdict shared with E1 (`randomization.verdict`): `equivalent` if the 90% interval lies inside (-0.01, +0.01), `different` if not and the 95% interval excludes 0, else `inconclusive`.

## Pre-registered reading (written before any result was computed)

1. **N = 4000 is saturating:** the contrast `R^2(11000) - R^2(4000)` for `same views` is `equivalent` (within 0.01) for every group of trained runs. Then the reported R^2 at N = 4000 is a good estimate of what this regressor can reach on these directions, and the numbers of the redundancy table are not limited by N (to within 0.01).
2. **Not saturating:** the same contrast is `different` with a positive point estimate above 0.01. Then the redundancy R^2 reported at N = 4000 is a lower bound by about that amount (higher N, higher R^2); statements such as "about 5% of the radius variance is not explained" would be too pessimistic, and the unexplained share would have to be quoted at the largest N.
3. **Inconclusive:** the interval is wider than the margin (seed or held-out noise): reported as such; no saturation claim either way.
4. The `disjoint halves` setup is read the same way as a check; it has a lower ceiling (the reliability of the target) and may saturate earlier.
5. For the random-init groups the curve is shown for comparison only; the saturation decision uses the trained groups.
6. The contrasts `R^2(2000) - R^2(4000)` and `R^2(8000) - R^2(4000)` are reported to locate where the curve flattens; they carry no decision.

## Results

## Interpretation

## Next steps
