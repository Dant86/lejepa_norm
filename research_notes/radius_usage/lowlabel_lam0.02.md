# E1-low-label: usage of the radius with scarce labels, lambda = 0.02
- **Branch / version:** `radius-usage-lowlabel` @ `9a46ed3c32e937d98696a5fdb80fee13d79eb963` (analysis code; per-run jobs at `5079b0a`, same evaluation code)
- **Run datetime (UTC):** 2026-10-25T21:02Z (aggregation; the per-run jobs ended 2026-10-25T20:41Z to 20:54Z, SLURM reports CDT = UTC-5, converted)
- **Command(s):**
  per run (SLURM, `sbatch --export=ALL`, 4 CPUs, 16 GB, CPU only; `<repo>` is the checkout of this branch; 9 jobs, one per run, 13-17 minutes each, jobs `radius_marginal_lam0.02_s{0,1,2}`: 2246242, 2246243, 2246244; `lam0.05_s{0,1,2}`: 2246245, 2246246, 2246247; `lam0.1_s{0,1,2}`: 2246249, 2246250, 2246294), `cd <repo> && sbatch --export=ALL slurm/lowlabel.sbatch configs/base.yaml radius_marginal <RUN>` which runs `apps/analyze.py --config configs/base.yaml --lowlabel radius_marginal --lowlabel-run <RUN>` with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=2` (jobs ran at commit `5079b0a`; the evaluation code is unchanged since)
  Aggregation (this note): `cd <repo> && export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=<repo>/src && <python> apps/analyze.py --config configs/base.yaml --lowlabel radius_marginal --lowlabel-lambda 0.02 --note radius_usage/lowlabel_lam0.02.md`
- **Datasets:** STL-10 `train` (the 5000 labeled images; the labeled bank is a class-balanced random subset of it) and `test` (8000 images, evaluation), clean, center views ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10)). The **canonical Stanford archive was used** (via the prepared arrays and the existing `stl10_train` / `stl10_test` embedding dumps); the HF page is linked for reference only.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` (epoch 200) of `radius_marginal_lam0.02_s{0,1,2}` (Variant A).
- **Scope:** h is not analyzed; LeJEPA's product is z. All results are in z-space.
- **Equivalence margin:** 0.5 accuracy points, fixed before looking at any result (`eval.equivalence_margin`).

## Question

If the radius carried a confidence-like signal, it should matter most when labels are scarce, because a reader with little labeled data has little else to go on. Pre-stated hypothesis: randomizing the radius of z hurts the E1 readers more at 50-500 labels than with all 5000 labels. This note is the low-label repeat of E1 (`research_notes/radius_randomization/`) for lambda = 0.02, three training seeds; the cross-lambda reading is in `lowlabel_summary.md`.

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
| 0.02 | 50 | 10 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.6093 | 0.5926 | -1.67 | [-1.80, -1.53] | 0.24 | different | different |
| 0.02 | 50 | 10 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.6093 | 0.6070 | -0.23 | [-0.30, -0.17] | 0.08 | equivalent | equivalent |
| 0.02 | 50 | 10 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.6093 | 0.6247 | +1.54 | [+1.38, +1.69] | 0.25 | different | different |
| 0.02 | 50 | 10 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.6093 | 0.6083 | -0.10 | [-0.16, -0.03] | 0.07 | equivalent | equivalent |
| 0.02 | 50 | 10 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.6093 | 0.6090 | -0.03 | [-0.06, -0.00] | 0.02 | equivalent | equivalent |
| 0.02 | 50 | 10 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.6093 | 0.1229 | -48.64 | [-49.06, -48.24] | 0.67 | different | different |
| 0.02 | 50 | 10 | linear probe | retrained on shuffled train | permute radius across images | 0.6625 | 0.6641 | +0.16 | [+0.10, +0.22] | 0.09 | equivalent | equivalent |
| 0.02 | 50 | 10 | linear probe | retrained on shuffled train | residual-only shuffle | 0.6625 | 0.6623 | -0.03 | [-0.05, -0.00] | 0.02 | equivalent | equivalent |
| 0.02 | 50 | 10 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.6625 | 0.6665 | +0.40 | [+0.33, +0.46] | 0.10 | equivalent | different |
| 0.02 | 50 | 10 | linear probe | trained on clean train, frozen | permute radius across images | 0.6625 | 0.6621 | -0.05 | [-0.08, -0.01] | 0.04 | equivalent | equivalent |
| 0.02 | 50 | 10 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.6625 | 0.6623 | -0.02 | [-0.04, -0.01] | 0.01 | equivalent | equivalent |
| 0.02 | 50 | 10 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.6625 | 0.1865 | -47.61 | [-48.02, -47.20] | 1.51 | different | different |
| 0.02 | 50 | 10 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.6493 | 0.6531 | +0.38 | [+0.32, +0.44] | 0.10 | equivalent | different |
| 0.02 | 50 | 10 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.6493 | 0.6490 | -0.03 | [-0.06, -0.01] | 0.02 | equivalent | equivalent |
| 0.02 | 50 | 10 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.6493 | 0.6558 | +0.65 | [+0.58, +0.71] | 0.10 | different | different |
| 0.02 | 50 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.6493 | 0.6491 | -0.03 | [-0.05, -0.00] | 0.02 | equivalent | equivalent |
| 0.02 | 50 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.6493 | 0.6492 | -0.01 | [-0.02, -0.00] | 0.01 | equivalent | equivalent |
| 0.02 | 50 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.6493 | 0.2705 | -37.89 | [-38.24, -37.52] | 1.60 | different | different |
| 0.02 | 100 | 10 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.6659 | 0.6570 | -0.89 | [-1.03, -0.75] | 0.19 | different | different |
| 0.02 | 100 | 10 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.6659 | 0.6640 | -0.19 | [-0.26, -0.12] | 0.07 | equivalent | equivalent |
| 0.02 | 100 | 10 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.6659 | 0.6715 | +0.56 | [+0.40, +0.73] | 0.22 | different | different |
| 0.02 | 100 | 10 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.6659 | 0.6670 | +0.11 | [+0.04, +0.18] | 0.05 | equivalent | equivalent |
| 0.02 | 100 | 10 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.6659 | 0.6654 | -0.05 | [-0.08, -0.01] | 0.02 | equivalent | equivalent |
| 0.02 | 100 | 10 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.6659 | 0.1366 | -52.93 | [-53.35, -52.51] | 0.66 | different | different |
| 0.02 | 100 | 10 | linear probe | retrained on shuffled train | permute radius across images | 0.7077 | 0.7081 | +0.05 | [-0.02, +0.10] | 0.06 | equivalent | equivalent |
| 0.02 | 100 | 10 | linear probe | retrained on shuffled train | residual-only shuffle | 0.7077 | 0.7075 | -0.02 | [-0.04, +0.01] | 0.02 | equivalent | equivalent |
| 0.02 | 100 | 10 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.7077 | 0.7090 | +0.14 | [+0.07, +0.20] | 0.06 | equivalent | equivalent |
| 0.02 | 100 | 10 | linear probe | trained on clean train, frozen | permute radius across images | 0.7077 | 0.7079 | +0.03 | [-0.01, +0.06] | 0.02 | equivalent | equivalent |
| 0.02 | 100 | 10 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.7077 | 0.7075 | -0.02 | [-0.03, -0.00] | 0.01 | equivalent | equivalent |
| 0.02 | 100 | 10 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.7077 | 0.1434 | -56.43 | [-56.87, -55.97] | 0.96 | different | different |
| 0.02 | 100 | 10 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.6890 | 0.6936 | +0.46 | [+0.41, +0.52] | 0.07 | different | different |
| 0.02 | 100 | 10 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.6890 | 0.6891 | +0.01 | [-0.01, +0.04] | 0.02 | equivalent | equivalent |
| 0.02 | 100 | 10 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.6890 | 0.6944 | +0.55 | [+0.49, +0.61] | 0.06 | different | different |
| 0.02 | 100 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.6890 | 0.6891 | +0.02 | [-0.00, +0.04] | 0.02 | equivalent | equivalent |
| 0.02 | 100 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.6890 | 0.6889 | -0.00 | [-0.01, +0.01] | 0.01 | equivalent | equivalent |
| 0.02 | 100 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.6890 | 0.2142 | -47.47 | [-47.87, -47.09] | 2.10 | different | different |
| 0.02 | 500 | 10 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.7293 | 0.7225 | -0.68 | [-0.80, -0.57] | 0.08 | different | different |
| 0.02 | 500 | 10 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.7293 | 0.7285 | -0.08 | [-0.14, -0.02] | 0.04 | equivalent | equivalent |
| 0.02 | 500 | 10 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.7293 | 0.7259 | -0.34 | [-0.49, -0.22] | 0.07 | equivalent | different |
| 0.02 | 500 | 10 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.7293 | 0.7308 | +0.15 | [+0.08, +0.21] | 0.03 | equivalent | equivalent |
| 0.02 | 500 | 10 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.7293 | 0.7287 | -0.06 | [-0.09, -0.03] | 0.02 | equivalent | equivalent |
| 0.02 | 500 | 10 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.7293 | 0.1215 | -60.78 | [-61.26, -60.34] | 0.51 | different | different |
| 0.02 | 500 | 10 | linear probe | retrained on shuffled train | permute radius across images | 0.7601 | 0.7586 | -0.16 | [-0.20, -0.10] | 0.03 | equivalent | equivalent |
| 0.02 | 500 | 10 | linear probe | retrained on shuffled train | residual-only shuffle | 0.7601 | 0.7602 | +0.00 | [-0.02, +0.03] | 0.01 | equivalent | equivalent |
| 0.02 | 500 | 10 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.7601 | 0.7594 | -0.08 | [-0.13, -0.02] | 0.03 | equivalent | equivalent |
| 0.02 | 500 | 10 | linear probe | trained on clean train, frozen | permute radius across images | 0.7601 | 0.7599 | -0.03 | [-0.06, +0.01] | 0.03 | equivalent | equivalent |
| 0.02 | 500 | 10 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.7601 | 0.7601 | -0.00 | [-0.02, +0.01] | 0.01 | equivalent | equivalent |
| 0.02 | 500 | 10 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.7601 | 0.1803 | -57.98 | [-58.43, -57.52] | 0.89 | different | different |
| 0.02 | 500 | 10 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.7508 | 0.7509 | +0.01 | [-0.04, +0.05] | 0.03 | equivalent | equivalent |
| 0.02 | 500 | 10 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.7508 | 0.7507 | -0.01 | [-0.04, +0.01] | 0.01 | equivalent | equivalent |
| 0.02 | 500 | 10 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.7508 | 0.7509 | +0.01 | [-0.04, +0.06] | 0.03 | equivalent | equivalent |
| 0.02 | 500 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.7508 | 0.7506 | -0.03 | [-0.05, -0.00] | 0.01 | equivalent | equivalent |
| 0.02 | 500 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.7508 | 0.7508 | -0.01 | [-0.02, +0.00] | 0.00 | equivalent | equivalent |
| 0.02 | 500 | 10 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.7508 | 0.3137 | -43.71 | [-44.10, -43.28] | 1.96 | different | different |
| 0.02 | 5000 | 1 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.7774 | 0.7704 | -0.70 | [-0.91, -0.49] | 0.00 | different | different |
| 0.02 | 5000 | 1 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.7774 | 0.7769 | -0.04 | [-0.18, +0.10] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.7774 | 0.7735 | -0.38 | [-0.62, -0.15] | 0.00 | different | different |
| 0.02 | 5000 | 1 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.7774 | 0.7769 | -0.05 | [-0.22, +0.13] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.7774 | 0.7777 | +0.03 | [-0.08, +0.13] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.7774 | 0.1299 | -64.75 | [-65.37, -64.13] | 0.00 | different | different |
| 0.02 | 5000 | 1 | linear probe | retrained on shuffled train | permute radius across images | 0.8042 | 0.8036 | -0.06 | [-0.18, +0.07] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | linear probe | retrained on shuffled train | residual-only shuffle | 0.8042 | 0.8039 | -0.03 | [-0.09, +0.05] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.8042 | 0.8046 | +0.05 | [-0.09, +0.19] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | linear probe | trained on clean train, frozen | permute radius across images | 0.8042 | 0.8001 | -0.41 | [-0.56, -0.27] | 0.00 | different | different |
| 0.02 | 5000 | 1 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.8042 | 0.8037 | -0.05 | [-0.11, +0.02] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.8042 | 0.1247 | -67.94 | [-68.53, -67.37] | 0.00 | different | different |
| 0.02 | 5000 | 1 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.8051 | 0.8037 | -0.14 | [-0.30, +0.00] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.8051 | 0.8058 | +0.07 | [-0.04, +0.19] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.8051 | 0.8033 | -0.18 | [-0.35, +0.00] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.8051 | 0.8038 | -0.12 | [-0.20, -0.05] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.8051 | 0.8046 | -0.05 | [-0.09, -0.01] | 0.00 | equivalent | equivalent |
| 0.02 | 5000 | 1 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.8051 | 0.3782 | -42.68 | [-43.30, -42.11] | 0.00 | different | different |
<!-- END GENERATED: analyze.lowlabel_comparison -->

<!-- BEGIN GENERATED: analyze.lowlabel_summary -->
### Compact view

Accuracy change in points, `difference [90% CI including subset noise] verdict(test images)/verdict(with subsets)` with E = equivalent (margin ±0.5 points), D = different, I = inconclusive. The frozen constant-radius cell (chance level, a distribution-shift artifact) is omitted.

| protocol | reader | condition | lambda 0.02, 50 labels | lambda 0.02, 100 labels | lambda 0.02, 500 labels | lambda 0.02, 5000 labels |
|---|---|---|---|---|---|---|
| retrained on shuffled train | k-NN (euclidean) | permute radius across images | -1.67 [-2.09, -1.25] D/D | -0.89 [-1.23, -0.55] D/D | -0.68 [-0.86, -0.51] D/D | -0.70 [-0.91, -0.49] D/D |
| retrained on shuffled train | k-NN (euclidean) | residual-only shuffle | -0.23 [-0.38, -0.09] E/E | -0.19 [-0.32, -0.06] E/E | -0.08 [-0.17, +0.00] E/E | -0.04 [-0.18, +0.10] E/E |
| retrained on shuffled train | k-NN (euclidean) | constant radius (unit norm) | +1.54 [+1.10, +1.97] D/D | +0.56 [+0.17, +0.96] D/D | -0.34 [-0.52, -0.17] E/D | -0.38 [-0.62, -0.15] D/D |
| retrained on shuffled train | linear probe | permute radius across images | +0.16 [-0.00, +0.32] E/E | +0.05 [-0.07, +0.16] E/E | -0.16 [-0.22, -0.09] E/E | -0.06 [-0.18, +0.07] E/E |
| retrained on shuffled train | linear probe | residual-only shuffle | -0.03 [-0.06, +0.01] E/E | -0.02 [-0.05, +0.02] E/E | +0.00 [-0.03, +0.04] E/E | -0.03 [-0.10, +0.05] E/E |
| retrained on shuffled train | linear probe | constant radius (unit norm) | +0.40 [+0.22, +0.57] E/D | +0.14 [+0.02, +0.25] E/E | -0.08 [-0.15, -0.00] E/E | +0.05 [-0.09, +0.19] E/E |
| retrained on shuffled train | MLP (1 hidden layer) | permute radius across images | +0.38 [+0.21, +0.55] E/D | +0.46 [+0.33, +0.59] D/D | +0.01 [-0.06, +0.07] E/E | -0.14 [-0.29, +0.01] E/E |
| retrained on shuffled train | MLP (1 hidden layer) | residual-only shuffle | -0.03 [-0.08, +0.01] E/E | +0.01 [-0.03, +0.06] E/E | -0.01 [-0.04, +0.02] E/E | +0.07 [-0.04, +0.19] E/E |
| retrained on shuffled train | MLP (1 hidden layer) | constant radius (unit norm) | +0.65 [+0.46, +0.83] D/D | +0.55 [+0.43, +0.66] D/D | +0.01 [-0.06, +0.07] E/E | -0.18 [-0.36, -0.00] E/E |
| trained on clean train, frozen | k-NN (euclidean) | permute radius across images | -0.10 [-0.24, +0.04] E/E | +0.11 [-0.00, +0.22] E/E | +0.15 [+0.06, +0.23] E/E | -0.05 [-0.22, +0.13] E/E |
| trained on clean train, frozen | k-NN (euclidean) | residual-only shuffle | -0.03 [-0.08, +0.01] E/E | -0.05 [-0.09, -0.00] E/E | -0.06 [-0.11, -0.01] E/E | +0.03 [-0.07, +0.13] E/E |
| trained on clean train, frozen | linear probe | permute radius across images | -0.05 [-0.12, +0.03] E/E | +0.03 [-0.03, +0.08] E/E | -0.03 [-0.08, +0.03] E/E | -0.41 [-0.55, -0.27] D/D |
| trained on clean train, frozen | linear probe | residual-only shuffle | -0.02 [-0.05, -0.00] E/E | -0.02 [-0.04, +0.00] E/E | -0.00 [-0.03, +0.02] E/E | -0.05 [-0.12, +0.02] E/E |
| trained on clean train, frozen | MLP (1 hidden layer) | permute radius across images | -0.03 [-0.07, +0.02] E/E | +0.02 [-0.02, +0.06] E/E | -0.03 [-0.06, +0.01] E/E | -0.12 [-0.20, -0.05] E/E |
| trained on clean train, frozen | MLP (1 hidden layer) | residual-only shuffle | -0.01 [-0.03, +0.00] E/E | -0.00 [-0.02, +0.01] E/E | -0.01 [-0.02, +0.01] E/E | -0.05 [-0.09, -0.01] E/E |
<!-- END GENERATED: analyze.lowlabel_summary -->

<!-- BEGIN GENERATED: analyze.lowlabel_accuracy -->
### Accuracy per label budget

| lambda | labels | reader | protocol | condition | accuracy, mean ± sd over seeds | sd over subsets (mean over seeds) |
|---|---|---|---|---|---|---|
| 0.02 | 50 | k-NN (euclidean) | retrained on shuffled train | baseline | 0.6093 ± 0.0041 | 0.0267 |
| 0.02 | 50 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.5926 ± 0.0049 | 0.0254 |
| 0.02 | 50 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.6070 ± 0.0057 | 0.0265 |
| 0.02 | 50 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.6247 ± 0.0041 | 0.0260 |
| 0.02 | 50 | k-NN (euclidean) | trained on clean train, frozen | baseline | 0.6093 ± 0.0041 | 0.0267 |
| 0.02 | 50 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.6083 ± 0.0049 | 0.0256 |
| 0.02 | 50 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.6090 ± 0.0040 | 0.0264 |
| 0.02 | 50 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.1229 ± 0.0122 | 0.0313 |
| 0.02 | 50 | linear probe | retrained on shuffled train | baseline | 0.6625 ± 0.0055 | 0.0222 |
| 0.02 | 50 | linear probe | retrained on shuffled train | permute radius across images | 0.6641 ± 0.0047 | 0.0212 |
| 0.02 | 50 | linear probe | retrained on shuffled train | residual-only shuffle | 0.6623 ± 0.0053 | 0.0226 |
| 0.02 | 50 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.6665 ± 0.0044 | 0.0224 |
| 0.02 | 50 | linear probe | trained on clean train, frozen | baseline | 0.6625 ± 0.0055 | 0.0222 |
| 0.02 | 50 | linear probe | trained on clean train, frozen | permute radius across images | 0.6621 ± 0.0055 | 0.0230 |
| 0.02 | 50 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.6623 ± 0.0055 | 0.0221 |
| 0.02 | 50 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.1865 ± 0.0078 | 0.0773 |
| 0.02 | 50 | MLP (1 hidden layer) | retrained on shuffled train | baseline | 0.6493 ± 0.0035 | 0.0293 |
| 0.02 | 50 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.6531 ± 0.0041 | 0.0258 |
| 0.02 | 50 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.6490 ± 0.0037 | 0.0291 |
| 0.02 | 50 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.6558 ± 0.0042 | 0.0262 |
| 0.02 | 50 | MLP (1 hidden layer) | trained on clean train, frozen | baseline | 0.6493 ± 0.0035 | 0.0293 |
| 0.02 | 50 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.6491 ± 0.0035 | 0.0297 |
| 0.02 | 50 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.6492 ± 0.0034 | 0.0293 |
| 0.02 | 50 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.2705 ± 0.0272 | 0.0883 |
| 0.02 | 100 | k-NN (euclidean) | retrained on shuffled train | baseline | 0.6659 ± 0.0043 | 0.0133 |
| 0.02 | 100 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.6570 ± 0.0029 | 0.0096 |
| 0.02 | 100 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.6640 ± 0.0032 | 0.0137 |
| 0.02 | 100 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.6715 ± 0.0017 | 0.0108 |
| 0.02 | 100 | k-NN (euclidean) | trained on clean train, frozen | baseline | 0.6659 ± 0.0043 | 0.0133 |
| 0.02 | 100 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.6670 ± 0.0035 | 0.0129 |
| 0.02 | 100 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.6654 ± 0.0037 | 0.0130 |
| 0.02 | 100 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.1366 ± 0.0162 | 0.0378 |
| 0.02 | 100 | linear probe | retrained on shuffled train | baseline | 0.7077 ± 0.0038 | 0.0135 |
| 0.02 | 100 | linear probe | retrained on shuffled train | permute radius across images | 0.7081 ± 0.0036 | 0.0122 |
| 0.02 | 100 | linear probe | retrained on shuffled train | residual-only shuffle | 0.7075 ± 0.0037 | 0.0134 |
| 0.02 | 100 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.7090 ± 0.0036 | 0.0127 |
| 0.02 | 100 | linear probe | trained on clean train, frozen | baseline | 0.7077 ± 0.0038 | 0.0135 |
| 0.02 | 100 | linear probe | trained on clean train, frozen | permute radius across images | 0.7079 ± 0.0035 | 0.0136 |
| 0.02 | 100 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.7075 ± 0.0036 | 0.0134 |
| 0.02 | 100 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.1434 ± 0.0105 | 0.0506 |
| 0.02 | 100 | MLP (1 hidden layer) | retrained on shuffled train | baseline | 0.6890 ± 0.0022 | 0.0141 |
| 0.02 | 100 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.6936 ± 0.0022 | 0.0134 |
| 0.02 | 100 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.6891 ± 0.0019 | 0.0140 |
| 0.02 | 100 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.6944 ± 0.0021 | 0.0138 |
| 0.02 | 100 | MLP (1 hidden layer) | trained on clean train, frozen | baseline | 0.6890 ± 0.0022 | 0.0141 |
| 0.02 | 100 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.6891 ± 0.0020 | 0.0143 |
| 0.02 | 100 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.6889 ± 0.0021 | 0.0141 |
| 0.02 | 100 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.2142 ± 0.0197 | 0.1193 |
| 0.02 | 500 | k-NN (euclidean) | retrained on shuffled train | baseline | 0.7293 ± 0.0009 | 0.0078 |
| 0.02 | 500 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.7225 ± 0.0017 | 0.0062 |
| 0.02 | 500 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.7285 ± 0.0013 | 0.0073 |
| 0.02 | 500 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.7259 ± 0.0014 | 0.0064 |
| 0.02 | 500 | k-NN (euclidean) | trained on clean train, frozen | baseline | 0.7293 ± 0.0009 | 0.0078 |
| 0.02 | 500 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.7308 ± 0.0016 | 0.0068 |
| 0.02 | 500 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.7287 ± 0.0013 | 0.0078 |
| 0.02 | 500 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.1215 ± 0.0027 | 0.0257 |
| 0.02 | 500 | linear probe | retrained on shuffled train | baseline | 0.7601 ± 0.0009 | 0.0059 |
| 0.02 | 500 | linear probe | retrained on shuffled train | permute radius across images | 0.7586 ± 0.0009 | 0.0061 |
| 0.02 | 500 | linear probe | retrained on shuffled train | residual-only shuffle | 0.7602 ± 0.0007 | 0.0060 |
| 0.02 | 500 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.7594 ± 0.0008 | 0.0061 |
| 0.02 | 500 | linear probe | trained on clean train, frozen | baseline | 0.7601 ± 0.0009 | 0.0059 |
| 0.02 | 500 | linear probe | trained on clean train, frozen | permute radius across images | 0.7599 ± 0.0009 | 0.0062 |
| 0.02 | 500 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.7601 ± 0.0009 | 0.0060 |
| 0.02 | 500 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.1803 ± 0.0105 | 0.0460 |
| 0.02 | 500 | MLP (1 hidden layer) | retrained on shuffled train | baseline | 0.7508 ± 0.0011 | 0.0058 |
| 0.02 | 500 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.7509 ± 0.0012 | 0.0055 |
| 0.02 | 500 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.7507 ± 0.0011 | 0.0058 |
| 0.02 | 500 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.7509 ± 0.0014 | 0.0057 |
| 0.02 | 500 | MLP (1 hidden layer) | trained on clean train, frozen | baseline | 0.7508 ± 0.0011 | 0.0058 |
| 0.02 | 500 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.7506 ± 0.0012 | 0.0060 |
| 0.02 | 500 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.7508 ± 0.0011 | 0.0057 |
| 0.02 | 500 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.3137 ± 0.0133 | 0.1048 |
| 0.02 | 5000 | k-NN (euclidean) | retrained on shuffled train | baseline | 0.7774 ± 0.0012 | 0.0000 |
| 0.02 | 5000 | k-NN (euclidean) | retrained on shuffled train | permute radius across images | 0.7703 ± 0.0020 | 0.0000 |
| 0.02 | 5000 | k-NN (euclidean) | retrained on shuffled train | residual-only shuffle | 0.7769 ± 0.0011 | 0.0000 |
| 0.02 | 5000 | k-NN (euclidean) | retrained on shuffled train | constant radius (unit norm) | 0.7735 ± 0.0026 | 0.0000 |
| 0.02 | 5000 | k-NN (euclidean) | trained on clean train, frozen | baseline | 0.7774 ± 0.0012 | 0.0000 |
| 0.02 | 5000 | k-NN (euclidean) | trained on clean train, frozen | permute radius across images | 0.7769 ± 0.0011 | 0.0000 |
| 0.02 | 5000 | k-NN (euclidean) | trained on clean train, frozen | residual-only shuffle | 0.7777 ± 0.0015 | 0.0000 |
| 0.02 | 5000 | k-NN (euclidean) | trained on clean train, frozen | constant radius (unit norm) | 0.1299 ± 0.0113 | 0.0000 |
| 0.02 | 5000 | linear probe | retrained on shuffled train | baseline | 0.8042 ± 0.0022 | 0.0000 |
| 0.02 | 5000 | linear probe | retrained on shuffled train | permute radius across images | 0.8036 ± 0.0010 | 0.0000 |
| 0.02 | 5000 | linear probe | retrained on shuffled train | residual-only shuffle | 0.8039 ± 0.0019 | 0.0000 |
| 0.02 | 5000 | linear probe | retrained on shuffled train | constant radius (unit norm) | 0.8046 ± 0.0020 | 0.0000 |
| 0.02 | 5000 | linear probe | trained on clean train, frozen | baseline | 0.8042 ± 0.0022 | 0.0000 |
| 0.02 | 5000 | linear probe | trained on clean train, frozen | permute radius across images | 0.8001 ± 0.0026 | 0.0000 |
| 0.02 | 5000 | linear probe | trained on clean train, frozen | residual-only shuffle | 0.8037 ± 0.0022 | 0.0000 |
| 0.02 | 5000 | linear probe | trained on clean train, frozen | constant radius (unit norm) | 0.1247 ± 0.0346 | 0.0000 |
| 0.02 | 5000 | MLP (1 hidden layer) | retrained on shuffled train | baseline | 0.8051 ± 0.0017 | 0.0000 |
| 0.02 | 5000 | MLP (1 hidden layer) | retrained on shuffled train | permute radius across images | 0.8037 ± 0.0016 | 0.0000 |
| 0.02 | 5000 | MLP (1 hidden layer) | retrained on shuffled train | residual-only shuffle | 0.8058 ± 0.0017 | 0.0000 |
| 0.02 | 5000 | MLP (1 hidden layer) | retrained on shuffled train | constant radius (unit norm) | 0.8033 ± 0.0013 | 0.0000 |
| 0.02 | 5000 | MLP (1 hidden layer) | trained on clean train, frozen | baseline | 0.8051 ± 0.0017 | 0.0000 |
| 0.02 | 5000 | MLP (1 hidden layer) | trained on clean train, frozen | permute radius across images | 0.8038 ± 0.0017 | 0.0000 |
| 0.02 | 5000 | MLP (1 hidden layer) | trained on clean train, frozen | residual-only shuffle | 0.8046 ± 0.0014 | 0.0000 |
| 0.02 | 5000 | MLP (1 hidden layer) | trained on clean train, frozen | constant radius (unit norm) | 0.3782 ± 0.0748 | 0.0000 |
<!-- END GENERATED: analyze.lowlabel_accuracy -->

<!-- BEGIN GENERATED: analyze.lowlabel_hyper -->
### Hyperparameters selected by cross-validation inside each subset

Linear probe: inverse regularization C; MLP: weight decay.

| lambda | labels | reader | selected (value: count over seeds x subsets) |
|---|---|---|---|
| 0.02 | 50 | linear probe | 0.01: 26, 0.1: 2, 1: 2 |
| 0.02 | 50 | MLP (1 hidden layer) | 0.0001: 24, 0.01: 1, 1: 5 |
| 0.02 | 100 | linear probe | 0.01: 27, 0.1: 3 |
| 0.02 | 100 | MLP (1 hidden layer) | 0.0001: 20, 1: 10 |
| 0.02 | 500 | linear probe | 0.01: 16, 0.1: 14 |
| 0.02 | 500 | MLP (1 hidden layer) | 0.0001: 4, 0.01: 2, 1: 24 |
| 0.02 | 5000 | linear probe | 1: 3 |
| 0.02 | 5000 | MLP (1 hidden layer) | 1: 3 |
<!-- END GENERATED: analyze.lowlabel_hyper -->

<!-- BEGIN GENERATED: analyze.lowlabel_figures -->
### Figures

Per reader and lambda: accuracy vs number of labeled images (retrained protocol; mean over seeds and subsets, bars = standard deviation over subsets), then the paired accuracy change vs the baseline (bars = 95% paired bootstrap interval over test images; the frozen constant-radius cell is omitted, it is at chance).

![accuracy vs labels, k-NN (euclidean)](lowlabel_lam0.02_accuracy_k_NN.png)

![accuracy change, k-NN (euclidean)](lowlabel_lam0.02_change_k_NN.png)

![accuracy vs labels, linear probe](lowlabel_lam0.02_accuracy_linear_probe.png)

![accuracy change, linear probe](lowlabel_lam0.02_change_linear_probe.png)

![accuracy vs labels, MLP (1 hidden layer)](lowlabel_lam0.02_accuracy_MLP.png)

![accuracy change, MLP (1 hidden layer)](lowlabel_lam0.02_change_MLP.png)
<!-- END GENERATED: analyze.lowlabel_figures -->

## Interpretation

Highlights for lambda 0.02 (compact view above; full text in `lowlabel_summary.md`): the learned readers are insensitive to the radius at every budget (linear and MLP: all residual-shuffle cells `E/E`, no condition lowers accuracy by more than 0.2 points; the positive `different` cells are +0.4 to +0.65 points). Euclidean k-NN gains from removing the radius at 50 and 100 labels (+1.54, +0.56) and loses slightly at 5000 (-0.38); permuting hurts it (-1.67 at 50 labels, -0.70 at 5000); the residual shuffle is neutral. This is the smallest lambda and the smallest k-NN effects of the three.

## Next steps

- Cosine k-NN is the natural reader at low labels (it is the `constant` condition); whether its advantage over Euclidean k-NN at 50 labels also holds for other embedding methods is not tested here.
- The confidence-like reading of the radius is tested more directly by selective classification (radius as a confidence score), not by randomization; see the other radius-usage notes.
- Training-seed variability: only 3 seeds per lambda; more seeds would be needed to bound the lambda trend of the k-NN effects.
