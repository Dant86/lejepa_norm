# E1-low-label: usage of the radius with scarce labels, lambda = 0.1
- **Branch / version:** `radius-usage-lowlabel` @ `9a46ed3c32e937d98696a5fdb80fee13d79eb963` (analysis code; per-run jobs at `5079b0a`, same evaluation code)
- **Run datetime (UTC):** 2026-10-25T21:02Z (aggregation; the per-run jobs ended 2026-10-25T20:41Z to 20:54Z, SLURM reports CDT = UTC-5, converted)
- **Command(s):**
  per run (SLURM, `sbatch --export=ALL`, 4 CPUs, 16 GB, CPU only; `<repo>` is the checkout of this branch; 9 jobs, one per run, 13-17 minutes each, jobs `radius_marginal_lam0.02_s{0,1,2}`: 2246242, 2246243, 2246244; `lam0.05_s{0,1,2}`: 2246245, 2246246, 2246247; `lam0.1_s{0,1,2}`: 2246249, 2246250, 2246294), `cd <repo> && sbatch --export=ALL slurm/lowlabel.sbatch configs/base.yaml radius_marginal <RUN>` which runs `apps/analyze.py --config configs/base.yaml --lowlabel radius_marginal --lowlabel-run <RUN>` with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=2` (jobs ran at commit `5079b0a`; the evaluation code is unchanged since)
  Aggregation (this note): `cd <repo> && export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=<repo>/src && <python> apps/analyze.py --config configs/base.yaml --lowlabel radius_marginal --lowlabel-lambda 0.1 --note radius_usage/lowlabel_lam0.1.md`
- **Datasets:** STL-10 `train` (the 5000 labeled images; the labeled bank is a class-balanced random subset of it) and `test` (8000 images, evaluation), clean, center views ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10)). The **canonical Stanford archive was used** (via the prepared arrays and the existing `stl10_train` / `stl10_test` embedding dumps); the HF page is linked for reference only.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` (epoch 200) of `radius_marginal_lam0.1_s{0,1,2}` (Variant A).
- **Scope:** h is not analyzed; LeJEPA's product is z. All results are in z-space.
- **Equivalence margin:** 0.5 accuracy points, fixed before looking at any result (`eval.equivalence_margin`).

## Question

If the radius carried a confidence-like signal, it should matter most when labels are scarce, because a reader with little labeled data has little else to go on. Pre-stated hypothesis: randomizing the radius of z hurts the E1 readers more at 50-500 labels than with all 5000 labels. This note is the low-label repeat of E1 (`research_notes/radius_randomization/`) for lambda = 0.1, three training seeds; the cross-lambda reading is in `lowlabel_summary.md`.

## Setup (deviations from the base config)

Deviations from `configs/base.yaml`: none (k-NN `k` is capped by the per-class count, see below; 1000 bootstrap replicates, 5 shuffle repeats per randomized condition, margin 0.5 points = `eval.equivalence_margin`, 10 subsets = `eval.lowlabel_subsets`). Design, fixed in the code (`lejepa_norm.eval.lowlabel`) before any condition effect was looked at:

- **Budgets** (per class / total labels / share of the 5000-image labeled split): 5 / 50 / 1%, 10 / 100 / 2%, 50 / 500 / 10%, 500 / 5000 / 100% (the whole split; its single "subset" is the E1 setting).
- **Subsets:** 10 independent class-balanced random draws per budget (1 at 5000 labels). The draws depend only on the budget and the draw index, so every one of the 9 runs and every condition sees the same labeled images (paired). Subset-to-subset spread is reported (`subset SE`, `sd over subsets`).
- **Conditions, protocols, statistics:** exactly E1 (`lejepa_norm.eval.randomization`, same code path: `condition_cells`, `compare`): baseline, permute radius across images, residual-only shuffle, constant radius (unit norm); *retrained* and *frozen* protocols; paired bootstrap over test images within each seed, averaged over 3 seeds; TOST-style verdict with margin 0.5 points.
- **Readers:** Euclidean k-NN, linear probe, one-hidden-layer MLP (cosine k-NN dropped: radius-invariant by construction).
- **Hyperparameters** (decided before looking at condition effects): selected per subset on the *clean* labeled subset by stratified cross-validation inside the subset (`min(5, per-class count)` folds), then reused for all conditions and protocols. Linear probe `C` in {0.01, 0.1, 1}; MLP weight decay in {1e-4, 1e-2, 1} (the E1 grid); k-NN `k = min(20, per-class count)` (5 with 5 labels per class, 10 with 10, 20 beyond). At 5000 labels the cross-validation runs on the whole split (E1 used a single 20% validation split for the MLP; the linear probe used `C = 1`), so that budget is not bit-identical to E1.
- **Residual shuffle:** the direction -> log-radius regression uses no labels, so it is fitted once per run on the whole unlabeled `train` split (out-of-fold for the bank, E1 code) and the subset's images take their predictions from it. Fitting it on 50 images would be meaningless and would confound the condition with the budget.
- **Subset noise in the verdict:** the bootstrap interval conditions on the drawn subsets. `subset SE` is the standard error over subsets of the mean paired difference (combined over seeds); `verdict (with subsets)` adds it in quadrature to the bootstrap standard error (normal approximation). Because each subset's difference is measured on all test images, the subset SE also contains some test-image noise, i.e. it over-states the subset noise slightly: the second verdict is the stricter one. Training-seed variability beyond averaging 3 seeds is not covered (as in E1).

## Results

<!-- BEGIN GENERATED: analyze.lowlabel_comparison -->
### Paired accuracy change vs the baseline, per lambda and label budget

z-space, center views; labeled bank = a class-balanced random subset of STL-10 train, test set = STL-10 test; the same transformation is applied to bank and test set independently. Differences are in accuracy points (condition minus baseline on the same test images and the same subsets), averaged over training seeds and subsets. `90% CI, test images`: paired bootstrap (1000 replicates, test images resampled within each seed), conditional on the drawn subsets. `subset SE`: standard error of the mean paired difference over subsets. Verdicts, with the equivalence margin set to ±0.5 points before looking at results: `equivalent` if the 90% interval (TOST at 5%) lies inside the margin, `different` if not equivalent and the 95% interval excludes 0, otherwise `inconclusive`; `(with subsets)` adds the subset SE in quadrature to the bootstrap SE (normal approximation).

| lambda | labels | subsets | reader | protocol | condition | baseline acc | condition acc | difference (points) | 90% CI, test images (points) | subset SE (points) | verdict (test images) | verdict (with subsets) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 10 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.5349 | 0.5179 | -1.70 | [-1.84, -1.56] | 0.40 | different | different |
| 0.1 | 50 | 10 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.5349 | 0.5442 | +0.93 | [+0.84, +1.02] | 0.32 | different | different |
| 0.1 | 50 | 10 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.5349 | 0.5998 | +6.49 | [+6.32, +6.67] | 0.63 | different | different |
| 0.1 | 50 | 10 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.5349 | 0.5274 | -0.75 | [-0.82, -0.68] | 0.11 | different | different |
| 0.1 | 50 | 10 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.5349 | 0.5345 | -0.04 | [-0.08, -0.01] | 0.01 | equivalent | equivalent |
| 0.1 | 50 | 10 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.5349 | 0.1110 | -42.39 | [-42.71, -42.06] | 0.75 | different | different |
| 0.1 | 50 | 10 | linear probe | retrained on shuffled train | permute radius across images | 0.6299 | 0.6332 | +0.33 | [+0.28, +0.39] | 0.07 | equivalent | equivalent |
| 0.1 | 50 | 10 | linear probe | retrained on shuffled train | residual-only shuffle | 0.6299 | 0.6302 | +0.03 | [+0.01, +0.06] | 0.03 | equivalent | equivalent |
| 0.1 | 50 | 10 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.6299 | 0.6354 | +0.55 | [+0.49, +0.61] | 0.08 | different | different |
| 0.1 | 50 | 10 | linear probe | trained on clean train, frozen | permute radius across images | 0.6299 | 0.6295 | -0.04 | [-0.07, -0.01] | 0.03 | equivalent | equivalent |
| 0.1 | 50 | 10 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.6299 | 0.6300 | +0.01 | [-0.00, +0.02] | 0.01 | equivalent | equivalent |
| 0.1 | 50 | 10 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.6299 | 0.2057 | -42.41 | [-42.78, -42.05] | 1.56 | different | different |
| 0.1 | 50 | 10 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.6124 | 0.6165 | +0.41 | [+0.35, +0.47] | 0.09 | equivalent | different |
| 0.1 | 50 | 10 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.6124 | 0.6131 | +0.07 | [+0.04, +0.10] | 0.03 | equivalent | equivalent |
| 0.1 | 50 | 10 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.6124 | 0.6181 | +0.57 | [+0.51, +0.63] | 0.08 | different | different |
| 0.1 | 50 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.6124 | 0.6121 | -0.03 | [-0.05, -0.01] | 0.01 | equivalent | equivalent |
| 0.1 | 50 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.6124 | 0.6123 | -0.01 | [-0.02, +0.00] | 0.01 | equivalent | equivalent |
| 0.1 | 50 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.6124 | 0.2342 | -37.82 | [-38.16, -37.49] | 1.27 | different | different |
| 0.1 | 100 | 10 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.6236 | 0.6195 | -0.41 | [-0.56, -0.25] | 0.24 | different | inconclusive |
| 0.1 | 100 | 10 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.6236 | 0.6278 | +0.42 | [+0.33, +0.51] | 0.20 | different | different |
| 0.1 | 100 | 10 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.6236 | 0.6666 | +4.30 | [+4.11, +4.49] | 0.26 | different | different |
| 0.1 | 100 | 10 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.6236 | 0.6205 | -0.32 | [-0.38, -0.25] | 0.06 | equivalent | equivalent |
| 0.1 | 100 | 10 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.6236 | 0.6238 | +0.02 | [-0.02, +0.05] | 0.02 | equivalent | equivalent |
| 0.1 | 100 | 10 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.6236 | 0.1158 | -50.78 | [-51.15, -50.37] | 0.32 | different | different |
| 0.1 | 100 | 10 | linear probe | retrained on shuffled train | permute radius across images | 0.6945 | 0.6958 | +0.13 | [+0.08, +0.18] | 0.05 | equivalent | equivalent |
| 0.1 | 100 | 10 | linear probe | retrained on shuffled train | residual-only shuffle | 0.6945 | 0.6944 | -0.02 | [-0.04, +0.01] | 0.02 | equivalent | equivalent |
| 0.1 | 100 | 10 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.6945 | 0.6969 | +0.24 | [+0.18, +0.29] | 0.05 | equivalent | equivalent |
| 0.1 | 100 | 10 | linear probe | trained on clean train, frozen | permute radius across images | 0.6945 | 0.6944 | -0.01 | [-0.04, +0.01] | 0.02 | equivalent | equivalent |
| 0.1 | 100 | 10 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.6945 | 0.6944 | -0.01 | [-0.02, +0.00] | 0.01 | equivalent | equivalent |
| 0.1 | 100 | 10 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.6945 | 0.1790 | -51.55 | [-51.95, -51.10] | 1.15 | different | different |
| 0.1 | 100 | 10 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.6764 | 0.6795 | +0.32 | [+0.27, +0.36] | 0.05 | equivalent | equivalent |
| 0.1 | 100 | 10 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.6764 | 0.6768 | +0.04 | [+0.01, +0.07] | 0.02 | equivalent | equivalent |
| 0.1 | 100 | 10 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.6764 | 0.6807 | +0.43 | [+0.38, +0.49] | 0.05 | equivalent | different |
| 0.1 | 100 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.6764 | 0.6762 | -0.02 | [-0.04, +0.00] | 0.01 | equivalent | equivalent |
| 0.1 | 100 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.6764 | 0.6763 | -0.00 | [-0.01, +0.00] | 0.01 | equivalent | equivalent |
| 0.1 | 100 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.6764 | 0.2670 | -40.93 | [-41.30, -40.52] | 2.02 | different | different |
| 0.1 | 500 | 10 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.7399 | 0.7376 | -0.23 | [-0.35, -0.11] | 0.06 | equivalent | equivalent |
| 0.1 | 500 | 10 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.7399 | 0.7406 | +0.07 | [-0.00, +0.14] | 0.05 | equivalent | equivalent |
| 0.1 | 500 | 10 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.7399 | 0.7465 | +0.66 | [+0.53, +0.80] | 0.07 | different | different |
| 0.1 | 500 | 10 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.7399 | 0.7415 | +0.16 | [+0.10, +0.21] | 0.03 | equivalent | equivalent |
| 0.1 | 500 | 10 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.7399 | 0.7401 | +0.02 | [-0.02, +0.05] | 0.01 | equivalent | equivalent |
| 0.1 | 500 | 10 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.7399 | 0.1088 | -63.11 | [-63.58, -62.67] | 0.22 | different | different |
| 0.1 | 500 | 10 | linear probe | retrained on shuffled train | permute radius across images | 0.7743 | 0.7737 | -0.07 | [-0.10, -0.03] | 0.02 | equivalent | equivalent |
| 0.1 | 500 | 10 | linear probe | retrained on shuffled train | residual-only shuffle | 0.7743 | 0.7743 | -0.00 | [-0.02, +0.02] | 0.01 | equivalent | equivalent |
| 0.1 | 500 | 10 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.7743 | 0.7739 | -0.04 | [-0.08, -0.00] | 0.03 | equivalent | equivalent |
| 0.1 | 500 | 10 | linear probe | trained on clean train, frozen | permute radius across images | 0.7743 | 0.7745 | +0.02 | [-0.01, +0.04] | 0.01 | equivalent | equivalent |
| 0.1 | 500 | 10 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.7743 | 0.7744 | +0.00 | [-0.01, +0.02] | 0.01 | equivalent | equivalent |
| 0.1 | 500 | 10 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.7743 | 0.2252 | -54.91 | [-55.35, -54.45] | 1.26 | different | different |
| 0.1 | 500 | 10 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.7609 | 0.7618 | +0.09 | [+0.05, +0.13] | 0.02 | equivalent | equivalent |
| 0.1 | 500 | 10 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.7609 | 0.7610 | +0.01 | [-0.01, +0.03] | 0.01 | equivalent | equivalent |
| 0.1 | 500 | 10 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.7609 | 0.7623 | +0.14 | [+0.10, +0.18] | 0.03 | equivalent | equivalent |
| 0.1 | 500 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.7609 | 0.7608 | -0.01 | [-0.02, +0.00] | 0.01 | equivalent | equivalent |
| 0.1 | 500 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.7609 | 0.7609 | -0.00 | [-0.01, +0.00] | 0.00 | equivalent | equivalent |
| 0.1 | 500 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.7609 | 0.4684 | -29.26 | [-29.61, -28.91] | 1.71 | different | different |
| 0.1 | 5000 | 1 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.7886 | 0.7841 | -0.45 | [-0.66, -0.23] | 0.00 | different | different |
| 0.1 | 5000 | 1 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.7886 | 0.7882 | -0.04 | [-0.18, +0.12] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.7886 | 0.7845 | -0.42 | [-0.65, -0.18] | 0.00 | different | different |
| 0.1 | 5000 | 1 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.7886 | 0.7891 | +0.05 | [-0.07, +0.18] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.7886 | 0.7888 | +0.01 | [-0.07, +0.10] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.7886 | 0.1349 | -65.37 | [-65.95, -64.78] | 0.00 | different | different |
| 0.1 | 5000 | 1 | linear probe | retrained on shuffled train | permute radius across images | 0.8161 | 0.8154 | -0.07 | [-0.17, +0.03] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | linear probe | retrained on shuffled train | residual-only shuffle | 0.8161 | 0.8160 | -0.01 | [-0.08, +0.05] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.8161 | 0.8157 | -0.04 | [-0.14, +0.07] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | linear probe | trained on clean train, frozen | permute radius across images | 0.8161 | 0.8152 | -0.09 | [-0.18, +0.01] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.8161 | 0.8157 | -0.04 | [-0.09, +0.01] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.8161 | 0.1952 | -62.09 | [-62.69, -61.48] | 0.00 | different | different |
| 0.1 | 5000 | 1 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.8205 | 0.8200 | -0.05 | [-0.16, +0.06] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.8205 | 0.8198 | -0.07 | [-0.16, +0.02] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.8205 | 0.8199 | -0.06 | [-0.18, +0.09] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.8205 | 0.8202 | -0.03 | [-0.07, +0.03] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.8205 | 0.8206 | +0.01 | [-0.02, +0.04] | 0.00 | equivalent | equivalent |
| 0.1 | 5000 | 1 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.8205 | 0.4920 | -32.85 | [-33.43, -32.29] | 0.00 | different | different |
<!-- END GENERATED: analyze.lowlabel_comparison -->

<!-- BEGIN GENERATED: analyze.lowlabel_summary -->
### Compact view

Accuracy change in points, `difference [90% CI including subset noise] verdict(test images)/verdict(with subsets)` with E = equivalent (margin ±0.5 points), D = different, I = inconclusive. The frozen constant-radius cell (chance level, a distribution-shift artifact) is omitted.

| protocol | reader | condition | lambda 0.1, 50 labels | lambda 0.1, 100 labels | lambda 0.1, 500 labels | lambda 0.1, 5000 labels |
|---|---|---|---|---|---|---|
| retrained on shuffled train | k-NN (euclidean) | permute radius across images | -1.70 [-2.38, -1.03] D/D | -0.41 [-0.83, +0.01] D/I | -0.23 [-0.39, -0.07] E/E | -0.45 [-0.67, -0.24] D/D |
| retrained on shuffled train | k-NN (euclidean) | residual-only shuffle | +0.93 [+0.39, +1.47] D/D | +0.42 [+0.08, +0.76] D/D | +0.07 [-0.04, +0.18] E/E | -0.04 [-0.19, +0.11] E/E |
| retrained on shuffled train | k-NN (euclidean) | constant radius (unit norm) | +6.49 [+5.44, +7.55] D/D | +4.30 [+3.83, +4.77] D/D | +0.66 [+0.48, +0.83] D/D | -0.42 [-0.65, -0.18] D/D |
| retrained on shuffled train | linear probe | permute radius across images | +0.33 [+0.21, +0.46] E/E | +0.13 [+0.04, +0.23] E/E | -0.07 [-0.12, -0.01] E/E | -0.07 [-0.17, +0.03] E/E |
| retrained on shuffled train | linear probe | residual-only shuffle | +0.03 [-0.02, +0.09] E/E | -0.02 [-0.06, +0.02] E/E | -0.00 [-0.03, +0.02] E/E | -0.01 [-0.08, +0.05] E/E |
| retrained on shuffled train | linear probe | constant radius (unit norm) | +0.55 [+0.41, +0.69] D/D | +0.24 [+0.14, +0.33] E/E | -0.04 [-0.10, +0.01] E/E | -0.04 [-0.14, +0.07] E/E |
| retrained on shuffled train | MLP (1 hidden layer) | permute radius across images | +0.41 [+0.25, +0.56] E/D | +0.32 [+0.22, +0.41] E/E | +0.09 [+0.03, +0.14] E/E | -0.05 [-0.16, +0.06] E/E |
| retrained on shuffled train | MLP (1 hidden layer) | residual-only shuffle | +0.07 [+0.01, +0.12] E/E | +0.04 [-0.01, +0.09] E/E | +0.01 [-0.02, +0.04] E/E | -0.07 [-0.16, +0.02] E/E |
| retrained on shuffled train | MLP (1 hidden layer) | constant radius (unit norm) | +0.57 [+0.42, +0.71] D/D | +0.43 [+0.34, +0.53] E/D | +0.14 [+0.08, +0.20] E/E | -0.06 [-0.20, +0.08] E/E |
| trained on clean train, frozen | k-NN (euclidean) | permute radius across images | -0.75 [-0.95, -0.56] D/D | -0.32 [-0.43, -0.20] E/E | +0.16 [+0.08, +0.23] E/E | +0.05 [-0.08, +0.18] E/E |
| trained on clean train, frozen | k-NN (euclidean) | residual-only shuffle | -0.04 [-0.08, +0.00] E/E | +0.02 [-0.04, +0.07] E/E | +0.02 [-0.02, +0.06] E/E | +0.01 [-0.07, +0.10] E/E |
| trained on clean train, frozen | linear probe | permute radius across images | -0.04 [-0.09, +0.01] E/E | -0.01 [-0.06, +0.03] E/E | +0.02 [-0.02, +0.05] E/E | -0.09 [-0.18, +0.01] E/E |
| trained on clean train, frozen | linear probe | residual-only shuffle | +0.01 [-0.01, +0.03] E/E | -0.01 [-0.03, +0.01] E/E | +0.00 [-0.01, +0.02] E/E | -0.04 [-0.09, +0.01] E/E |
| trained on clean train, frozen | MLP (1 hidden layer) | permute radius across images | -0.03 [-0.06, -0.00] E/E | -0.02 [-0.05, +0.01] E/E | -0.01 [-0.03, +0.01] E/E | -0.03 [-0.08, +0.02] E/E |
| trained on clean train, frozen | MLP (1 hidden layer) | residual-only shuffle | -0.01 [-0.03, +0.01] E/E | -0.00 [-0.02, +0.01] E/E | -0.00 [-0.01, +0.01] E/E | +0.01 [-0.02, +0.04] E/E |
<!-- END GENERATED: analyze.lowlabel_summary -->

<!-- BEGIN GENERATED: analyze.lowlabel_accuracy -->
### Accuracy per label budget

| lambda | labels | reader | protocol | condition | accuracy, mean ± sd over seeds | sd over subsets (mean over seeds) |
|---|---|---|---|---|---|---|
| 0.1 | 50 | k-NN (euclidean) | retrained on shuffled train | baseline | 0.5349 ± 0.0037 | 0.0466 |
| 0.1 | 50 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.5179 ± 0.0015 | 0.0442 |
| 0.1 | 50 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.5442 ± 0.0086 | 0.0418 |
| 0.1 | 50 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.5998 ± 0.0032 | 0.0298 |
| 0.1 | 50 | k-NN (euclidean) | trained on clean train, frozen | baseline | 0.5349 ± 0.0037 | 0.0466 |
| 0.1 | 50 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.5274 ± 0.0028 | 0.0484 |
| 0.1 | 50 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.5345 ± 0.0036 | 0.0467 |
| 0.1 | 50 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.1110 ± 0.0055 | 0.0190 |
| 0.1 | 50 | linear probe | retrained on shuffled train | baseline | 0.6299 ± 0.0010 | 0.0304 |
| 0.1 | 50 | linear probe | retrained on shuffled train | permute radius across images | 0.6332 ± 0.0022 | 0.0298 |
| 0.1 | 50 | linear probe | retrained on shuffled train | residual-only shuffle | 0.6302 ± 0.0005 | 0.0307 |
| 0.1 | 50 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.6354 ± 0.0018 | 0.0294 |
| 0.1 | 50 | linear probe | trained on clean train, frozen | baseline | 0.6299 ± 0.0010 | 0.0304 |
| 0.1 | 50 | linear probe | trained on clean train, frozen | permute radius across images | 0.6295 ± 0.0008 | 0.0305 |
| 0.1 | 50 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.6300 ± 0.0011 | 0.0304 |
| 0.1 | 50 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.2057 ± 0.0209 | 0.0804 |
| 0.1 | 50 | MLP (1 hidden layer) | retrained on shuffled train | baseline | 0.6124 ± 0.0040 | 0.0323 |
| 0.1 | 50 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.6165 ± 0.0047 | 0.0302 |
| 0.1 | 50 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.6131 ± 0.0037 | 0.0324 |
| 0.1 | 50 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.6181 ± 0.0046 | 0.0303 |
| 0.1 | 50 | MLP (1 hidden layer) | trained on clean train, frozen | baseline | 0.6124 ± 0.0040 | 0.0323 |
| 0.1 | 50 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.6121 ± 0.0040 | 0.0323 |
| 0.1 | 50 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.6123 ± 0.0041 | 0.0323 |
| 0.1 | 50 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.2342 ± 0.0063 | 0.0642 |
| 0.1 | 100 | k-NN (euclidean) | retrained on shuffled train | baseline | 0.6236 ± 0.0023 | 0.0171 |
| 0.1 | 100 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.6195 ± 0.0029 | 0.0156 |
| 0.1 | 100 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.6278 ± 0.0010 | 0.0151 |
| 0.1 | 100 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.6666 ± 0.0012 | 0.0117 |
| 0.1 | 100 | k-NN (euclidean) | trained on clean train, frozen | baseline | 0.6236 ± 0.0023 | 0.0171 |
| 0.1 | 100 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.6205 ± 0.0038 | 0.0183 |
| 0.1 | 100 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.6238 ± 0.0025 | 0.0165 |
| 0.1 | 100 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.1158 ± 0.0070 | 0.0183 |
| 0.1 | 100 | linear probe | retrained on shuffled train | baseline | 0.6945 ± 0.0020 | 0.0152 |
| 0.1 | 100 | linear probe | retrained on shuffled train | permute radius across images | 0.6958 ± 0.0014 | 0.0146 |
| 0.1 | 100 | linear probe | retrained on shuffled train | residual-only shuffle | 0.6944 ± 0.0021 | 0.0155 |
| 0.1 | 100 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.6969 ± 0.0013 | 0.0148 |
| 0.1 | 100 | linear probe | trained on clean train, frozen | baseline | 0.6945 ± 0.0020 | 0.0152 |
| 0.1 | 100 | linear probe | trained on clean train, frozen | permute radius across images | 0.6944 ± 0.0023 | 0.0155 |
| 0.1 | 100 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.6944 ± 0.0021 | 0.0152 |
| 0.1 | 100 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.1790 ± 0.0079 | 0.0653 |
| 0.1 | 100 | MLP (1 hidden layer) | retrained on shuffled train | baseline | 0.6764 ± 0.0023 | 0.0126 |
| 0.1 | 100 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.6795 ± 0.0013 | 0.0139 |
| 0.1 | 100 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.6768 ± 0.0021 | 0.0126 |
| 0.1 | 100 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.6807 ± 0.0018 | 0.0140 |
| 0.1 | 100 | MLP (1 hidden layer) | trained on clean train, frozen | baseline | 0.6764 ± 0.0023 | 0.0126 |
| 0.1 | 100 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.6762 ± 0.0020 | 0.0127 |
| 0.1 | 100 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.6763 ± 0.0023 | 0.0126 |
| 0.1 | 100 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.2670 ± 0.0156 | 0.1101 |
| 0.1 | 500 | k-NN (euclidean) | retrained on shuffled train | baseline | 0.7399 ± 0.0013 | 0.0049 |
| 0.1 | 500 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.7376 ± 0.0003 | 0.0045 |
| 0.1 | 500 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.7406 ± 0.0007 | 0.0051 |
| 0.1 | 500 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.7465 ± 0.0008 | 0.0044 |
| 0.1 | 500 | k-NN (euclidean) | trained on clean train, frozen | baseline | 0.7399 ± 0.0013 | 0.0049 |
| 0.1 | 500 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.7415 ± 0.0015 | 0.0047 |
| 0.1 | 500 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.7401 ± 0.0014 | 0.0049 |
| 0.1 | 500 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.1088 ± 0.0041 | 0.0102 |
| 0.1 | 500 | linear probe | retrained on shuffled train | baseline | 0.7743 ± 0.0021 | 0.0061 |
| 0.1 | 500 | linear probe | retrained on shuffled train | permute radius across images | 0.7737 ± 0.0019 | 0.0054 |
| 0.1 | 500 | linear probe | retrained on shuffled train | residual-only shuffle | 0.7743 ± 0.0020 | 0.0060 |
| 0.1 | 500 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.7739 ± 0.0019 | 0.0055 |
| 0.1 | 500 | linear probe | trained on clean train, frozen | baseline | 0.7743 ± 0.0021 | 0.0061 |
| 0.1 | 500 | linear probe | trained on clean train, frozen | permute radius across images | 0.7745 ± 0.0021 | 0.0058 |
| 0.1 | 500 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.7744 ± 0.0021 | 0.0060 |
| 0.1 | 500 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.2252 ± 0.0094 | 0.0674 |
| 0.1 | 500 | MLP (1 hidden layer) | retrained on shuffled train | baseline | 0.7609 ± 0.0011 | 0.0068 |
| 0.1 | 500 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.7618 ± 0.0010 | 0.0065 |
| 0.1 | 500 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.7610 ± 0.0011 | 0.0070 |
| 0.1 | 500 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.7623 ± 0.0011 | 0.0067 |
| 0.1 | 500 | MLP (1 hidden layer) | trained on clean train, frozen | baseline | 0.7609 ± 0.0011 | 0.0068 |
| 0.1 | 500 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.7608 ± 0.0010 | 0.0068 |
| 0.1 | 500 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.7609 ± 0.0011 | 0.0069 |
| 0.1 | 500 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.4684 ± 0.0106 | 0.0914 |
| 0.1 | 5000 | k-NN (euclidean) | retrained on shuffled train | baseline | 0.7886 ± 0.0012 | 0.0000 |
| 0.1 | 5000 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.7841 ± 0.0019 | 0.0000 |
| 0.1 | 5000 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.7882 ± 0.0018 | 0.0000 |
| 0.1 | 5000 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.7845 ± 0.0025 | 0.0000 |
| 0.1 | 5000 | k-NN (euclidean) | trained on clean train, frozen | baseline | 0.7886 ± 0.0012 | 0.0000 |
| 0.1 | 5000 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.7891 ± 0.0004 | 0.0000 |
| 0.1 | 5000 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.7888 ± 0.0006 | 0.0000 |
| 0.1 | 5000 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.1349 ± 0.0063 | 0.0000 |
| 0.1 | 5000 | linear probe | retrained on shuffled train | baseline | 0.8161 ± 0.0006 | 0.0000 |
| 0.1 | 5000 | linear probe | retrained on shuffled train | permute radius across images | 0.8154 ± 0.0012 | 0.0000 |
| 0.1 | 5000 | linear probe | retrained on shuffled train | residual-only shuffle | 0.8160 ± 0.0008 | 0.0000 |
| 0.1 | 5000 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.8157 ± 0.0006 | 0.0000 |
| 0.1 | 5000 | linear probe | trained on clean train, frozen | baseline | 0.8161 ± 0.0006 | 0.0000 |
| 0.1 | 5000 | linear probe | trained on clean train, frozen | permute radius across images | 0.8152 ± 0.0021 | 0.0000 |
| 0.1 | 5000 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.8157 ± 0.0012 | 0.0000 |
| 0.1 | 5000 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.1952 ± 0.0400 | 0.0000 |
| 0.1 | 5000 | MLP (1 hidden layer) | retrained on shuffled train | baseline | 0.8205 ± 0.0027 | 0.0000 |
| 0.1 | 5000 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.8200 ± 0.0031 | 0.0000 |
| 0.1 | 5000 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.8198 ± 0.0030 | 0.0000 |
| 0.1 | 5000 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.8199 ± 0.0023 | 0.0000 |
| 0.1 | 5000 | MLP (1 hidden layer) | trained on clean train, frozen | baseline | 0.8205 ± 0.0027 | 0.0000 |
| 0.1 | 5000 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.8202 ± 0.0030 | 0.0000 |
| 0.1 | 5000 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.8206 ± 0.0028 | 0.0000 |
| 0.1 | 5000 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.4920 ± 0.0709 | 0.0000 |
<!-- END GENERATED: analyze.lowlabel_accuracy -->

<!-- BEGIN GENERATED: analyze.lowlabel_hyper -->
### Hyperparameters selected by cross-validation inside each subset

Linear probe: inverse regularization C; MLP: weight decay.

| lambda | labels | reader | selected (value: count over seeds x subsets) |
|---|---|---|---|
| 0.1 | 50 | linear probe | 0.01: 25, 0.1: 2, 1: 3 |
| 0.1 | 50 | MLP (1 hidden layer) | 0.0001: 11, 1: 19 |
| 0.1 | 100 | linear probe | 0.01: 27, 0.1: 3 |
| 0.1 | 100 | MLP (1 hidden layer) | 0.0001: 13, 0.01: 1, 1: 16 |
| 0.1 | 500 | linear probe | 0.01: 29, 0.1: 1 |
| 0.1 | 500 | MLP (1 hidden layer) | 0.0001: 9, 0.01: 2, 1: 19 |
| 0.1 | 5000 | linear probe | 0.1: 2, 1: 1 |
| 0.1 | 5000 | MLP (1 hidden layer) | 1: 3 |
<!-- END GENERATED: analyze.lowlabel_hyper -->

<!-- BEGIN GENERATED: analyze.lowlabel_figures -->
### Figures

Per reader and lambda: accuracy vs number of labeled images (retrained protocol; mean over seeds and subsets, bars = standard deviation over subsets), then the paired accuracy change vs the baseline (bars = 95% paired bootstrap interval over test images; the frozen constant-radius cell is omitted, it is at chance).

![accuracy vs labels, k-NN (euclidean)](lowlabel_lam0.1_accuracy_k_NN.png)

![accuracy change, k-NN (euclidean)](lowlabel_lam0.1_change_k_NN.png)

![accuracy vs labels, linear probe](lowlabel_lam0.1_accuracy_linear_probe.png)

![accuracy change, linear probe](lowlabel_lam0.1_change_linear_probe.png)

![accuracy vs labels, MLP (1 hidden layer)](lowlabel_lam0.1_accuracy_MLP.png)

![accuracy change, MLP (1 hidden layer)](lowlabel_lam0.1_change_MLP.png)
<!-- END GENERATED: analyze.lowlabel_figures -->

## Interpretation

Highlights for lambda 0.1: the learned readers are insensitive to the radius (residual shuffle `E/E` everywhere). Euclidean k-NN shows the largest radius effects of the three lambdas: unit norm gains +6.49 (50 labels) and +4.30 (100 labels), +0.66 at 500, and loses 0.42 at 5000; permuting costs 1.70 at 50 labels; the residual shuffle *raises* accuracy (+0.93 at 50 labels, +0.42 at 100).

## Next steps

- Cosine k-NN is the natural reader at low labels (it is the `constant` condition); whether its advantage over Euclidean k-NN at 50 labels also holds for other embedding methods is not tested here.
- The confidence-like reading of the radius is tested more directly by selective classification (radius as a confidence score), not by randomization; see the other radius-usage notes.
- Training-seed variability: only 3 seeds per lambda; more seeds would be needed to bound the lambda trend of the k-NN effects.
