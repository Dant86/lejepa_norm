# radius_view_signal: summary and interpretation
- **Branch / version:** `radius-view-signal` @ `6ff28d6` (code; notes committed after)
- **Run datetime (UTC):** 2026-10-26T04:01Z to 05:20Z (units and aggregation, see commands)
- **Command(s):**
  Per-run units (CPU, 8 CPUs, 64 GB, about 15 to 20 min each; jobs 2248755 to 2248758, 2248941, 2248943 to 2248945 and 2248964, 2026-10-26T04:01Z to 04:46Z; checkout of this branch at `6ff28d6`, `PYTHONPATH=<repo>/src`, `.env` with `LEJEPA_NORM_NOTES_DIR` pointing at this checkout's `research_notes/`):
  `sbatch --export=ALL slurm/view_signal.sbatch configs/base.yaml radius_marginal radius_marginal_lam<lambda>_s<seed>` for lambda in 0.02, 0.05, 0.1 and seed in 0, 1, 2 (writes `<runs_dir>/<run>/view_signal.npz`).
  Aggregation (this note and the three per-lambda notes; job 2248965, 8 CPUs, 32 GB, 33 min, finished about 2026-10-26T05:20Z):
  `python apps/analyze.py --config configs/base.yaml --view-signal radius_marginal --note radius_view_signal/view_signal.md`
  History: a first pass of the units (jobs 2248076, 2248257 to 2248260, 2248336 to 2248338, 2248365, 2248547 at commit `de18063`, aggregation job 2248628) lacked the post hoc additions marked below; they were added after seeing it and everything was recomputed. The numbers of the pre-registered comparisons were identical in both passes (all steps are deterministic).
- **Datasets:** STL-10 ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10); the canonical archive was used, HF is for reference) `train` (clean labeled bank) and `test` (clean twins); STL-10-C built locally from STL-10 `test` (10 corruptions x 5 severities, 400k images; no HF mirror).
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` (epoch 200) of `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}` (Variant A, trained on A100s in earlier work; this study is CPU-only, no GPU was used).
- **Scope:** h is not analyzed; LeJEPA's product is z. All results are z-space.
- **Margins (fixed before the results):** 0.5 accuracy points; 0.01 Spearman. Pre-registration: `preregistration.md` (committed before any analysis code was run).

## Question

E6 and the shift study found that the radius moves under corruption while frozen readers do not use it. Is there nevertheless a usable signal in the radius about the *view*: how much of the clean-to-corrupted displacement is radial (A), would restoring the clean radius recover accuracy (B), does the radius residual tell the corruption severity beyond the direction (C), and does weighting or picking the views of an image by their radius help multi-view classification (D)? All z only, STL-10 `test` and STL-10-C against a clean STL-10 `train` bank, readers frozen on the clean bank, nine runs (lambda 0.02 / 0.05 / 0.1 x seeds 0 / 1 / 2). Per-lambda detail tables and figures: `view_signal_lam<lambda>.md`; definitions and the predictions written in advance: `preregistration.md`.

## Setup (deviations from the base config)

No training; `configs/base.yaml` (k = 20, 1000 bootstrap replicates, margin 0.5 accuracy points, 0.01 Spearman). Statistics are paired bootstrap over test images within seed, averaged over the three seeds (seed variability beyond three seeds and the dependence among the severities of one corruption are not modeled). Post hoc additions, made after the first pass of the results and not in the pre-registration (all labeled "post hoc" in the tables): in A the radius-change share `sum (r_x - r_c)^2 / sum ||z_x - z_c||^2` and the median cosine between the clean and corrupted directions; in D the `random single view` reference and the `highest-residual` / `lowest-residual` single views with their contrasts. Retrieval was not run in D (classification only). Lambda 0.05 numbers are quoted first below; the other lambdas agree unless stated.

## Results

<!-- BEGIN GENERATED: analyze.view_summary -->
### Summary across lambda (overall rows)

**B. Oracle radius** (all corruptions)

| lambda | level | reader | contrast | accuracy | reference accuracy | difference (points) | 90% CI | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | all corruptions | k-NN (euclidean) | oracle radius - baseline | 0.3693 | 0.3648 | +0.45 | [+0.42, +0.48] | [+0.41, +0.49] | equivalent |
| 0.02 | all corruptions | k-NN (euclidean) | scaled constant - baseline | 0.3689 | 0.3648 | +0.40 | [+0.37, +0.43] | [+0.36, +0.44] | equivalent |
| 0.02 | all corruptions | k-NN (euclidean) | oracle direction - clean twin | 0.7764 | 0.7774 | -0.09 | [-0.22, +0.03] | [-0.25, +0.06] | equivalent |
| 0.02 | all corruptions | k-NN (euclidean) | baseline - clean twin (total corruption loss) | 0.3648 | 0.7774 | -41.25 | [-41.64, -40.89] | [-41.74, -40.79] | different |
| 0.02 | all corruptions | linear probe | oracle radius - baseline | 0.3775 | 0.3727 | +0.48 | [+0.45, +0.51] | [+0.45, +0.52] | different |
| 0.02 | all corruptions | linear probe | scaled constant - baseline | 0.3747 | 0.3727 | +0.20 | [+0.16, +0.23] | [+0.15, +0.23] | equivalent |
| 0.02 | all corruptions | linear probe | oracle direction - clean twin | 0.8013 | 0.8040 | -0.27 | [-0.37, -0.17] | [-0.39, -0.15] | equivalent |
| 0.02 | all corruptions | linear probe | baseline - clean twin (total corruption loss) | 0.3727 | 0.8040 | -43.13 | [-43.50, -42.76] | [-43.58, -42.69] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | oracle radius - baseline | 0.3757 | 0.3738 | +0.19 | [+0.18, +0.21] | [+0.18, +0.21] | equivalent |
| 0.02 | all corruptions | MLP (1 hidden layer) | scaled constant - baseline | 0.3754 | 0.3738 | +0.16 | [+0.14, +0.17] | [+0.14, +0.17] | equivalent |
| 0.02 | all corruptions | MLP (1 hidden layer) | oracle direction - clean twin | 0.8032 | 0.8037 | -0.05 | [-0.11, +0.01] | [-0.13, +0.02] | equivalent |
| 0.02 | all corruptions | MLP (1 hidden layer) | baseline - clean twin (total corruption loss) | 0.3738 | 0.8037 | -43.00 | [-43.37, -42.65] | [-43.44, -42.57] | different |
| 0.05 | all corruptions | k-NN (euclidean) | oracle radius - baseline | 0.3716 | 0.3677 | +0.39 | [+0.36, +0.42] | [+0.36, +0.43] | equivalent |
| 0.05 | all corruptions | k-NN (euclidean) | scaled constant - baseline | 0.3710 | 0.3677 | +0.33 | [+0.30, +0.36] | [+0.29, +0.37] | equivalent |
| 0.05 | all corruptions | k-NN (euclidean) | oracle direction - clean twin | 0.7875 | 0.7870 | +0.04 | [-0.08, +0.15] | [-0.10, +0.18] | equivalent |
| 0.05 | all corruptions | k-NN (euclidean) | baseline - clean twin (total corruption loss) | 0.3677 | 0.7870 | -41.93 | [-42.27, -41.57] | [-42.34, -41.49] | different |
| 0.05 | all corruptions | linear probe | oracle radius - baseline | 0.3780 | 0.3724 | +0.56 | [+0.53, +0.59] | [+0.53, +0.60] | different |
| 0.05 | all corruptions | linear probe | scaled constant - baseline | 0.3752 | 0.3724 | +0.28 | [+0.25, +0.31] | [+0.24, +0.31] | equivalent |
| 0.05 | all corruptions | linear probe | oracle direction - clean twin | 0.8144 | 0.8175 | -0.30 | [-0.40, -0.20] | [-0.42, -0.18] | equivalent |
| 0.05 | all corruptions | linear probe | baseline - clean twin (total corruption loss) | 0.3724 | 0.8175 | -44.51 | [-44.85, -44.13] | [-44.93, -44.07] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | oracle radius - baseline | 0.3790 | 0.3777 | +0.14 | [+0.12, +0.15] | [+0.12, +0.15] | equivalent |
| 0.05 | all corruptions | MLP (1 hidden layer) | scaled constant - baseline | 0.3787 | 0.3777 | +0.10 | [+0.09, +0.12] | [+0.09, +0.12] | equivalent |
| 0.05 | all corruptions | MLP (1 hidden layer) | oracle direction - clean twin | 0.8152 | 0.8155 | -0.03 | [-0.09, +0.03] | [-0.11, +0.03] | equivalent |
| 0.05 | all corruptions | MLP (1 hidden layer) | baseline - clean twin (total corruption loss) | 0.3777 | 0.8155 | -43.79 | [-44.13, -43.45] | [-44.20, -43.36] | different |
| 0.1 | all corruptions | k-NN (euclidean) | oracle radius - baseline | 0.3695 | 0.3663 | +0.32 | [+0.30, +0.35] | [+0.29, +0.36] | equivalent |
| 0.1 | all corruptions | k-NN (euclidean) | scaled constant - baseline | 0.3687 | 0.3663 | +0.24 | [+0.21, +0.28] | [+0.21, +0.28] | equivalent |
| 0.1 | all corruptions | k-NN (euclidean) | oracle direction - clean twin | 0.7895 | 0.7886 | +0.08 | [-0.02, +0.18] | [-0.04, +0.20] | equivalent |
| 0.1 | all corruptions | k-NN (euclidean) | baseline - clean twin (total corruption loss) | 0.3663 | 0.7886 | -42.24 | [-42.62, -41.88] | [-42.68, -41.83] | different |
| 0.1 | all corruptions | linear probe | oracle radius - baseline | 0.3789 | 0.3742 | +0.47 | [+0.45, +0.49] | [+0.44, +0.50] | equivalent |
| 0.1 | all corruptions | linear probe | scaled constant - baseline | 0.3767 | 0.3742 | +0.25 | [+0.23, +0.27] | [+0.23, +0.28] | equivalent |
| 0.1 | all corruptions | linear probe | oracle direction - clean twin | 0.8100 | 0.8121 | -0.21 | [-0.30, -0.12] | [-0.31, -0.11] | equivalent |
| 0.1 | all corruptions | linear probe | baseline - clean twin (total corruption loss) | 0.3742 | 0.8121 | -43.79 | [-44.14, -43.42] | [-44.22, -43.36] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | oracle radius - baseline | 0.3842 | 0.3828 | +0.14 | [+0.13, +0.15] | [+0.13, +0.16] | equivalent |
| 0.1 | all corruptions | MLP (1 hidden layer) | scaled constant - baseline | 0.3837 | 0.3828 | +0.09 | [+0.08, +0.10] | [+0.07, +0.10] | equivalent |
| 0.1 | all corruptions | MLP (1 hidden layer) | oracle direction - clean twin | 0.8174 | 0.8175 | -0.02 | [-0.06, +0.03] | [-0.07, +0.03] | equivalent |
| 0.1 | all corruptions | MLP (1 hidden layer) | baseline - clean twin (total corruption loss) | 0.3828 | 0.8175 | -43.48 | [-43.82, -43.14] | [-43.87, -43.06] | different |

**D. Aggregation** (all corruptions and clean test)

| lambda | level | reader | contrast | accuracy | reference accuracy | difference (points) | 90% CI | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | all corruptions | k-NN (euclidean) | radius-weighted - mean | 0.3660 | 0.3637 | +0.23 | [+0.21, +0.25] | [+0.21, +0.25] | equivalent |
| 0.02 | all corruptions | k-NN (euclidean) | inverse-radius-weighted - mean | 0.3615 | 0.3637 | -0.22 | [-0.24, -0.20] | [-0.24, -0.20] | equivalent |
| 0.02 | all corruptions | k-NN (euclidean) | residual-weighted - mean | 0.3715 | 0.3637 | +0.78 | [+0.74, +0.82] | [+0.73, +0.83] | different |
| 0.02 | all corruptions | k-NN (euclidean) | top-radius view - mean | 0.3702 | 0.3637 | +0.65 | [+0.58, +0.73] | [+0.57, +0.75] | different |
| 0.02 | all corruptions | k-NN (euclidean) | bottom-radius view - mean | 0.3263 | 0.3637 | -3.74 | [-3.82, -3.66] | [-3.83, -3.65] | different |
| 0.02 | all corruptions | k-NN (euclidean) | center view only (context) - mean | 0.3710 | 0.3637 | +0.73 | [+0.67, +0.80] | [+0.66, +0.82] | different |
| 0.02 | all corruptions | k-NN (euclidean) | random single view (post hoc context) - mean | 0.3480 | 0.3637 | -1.57 | [-1.63, -1.52] | [-1.64, -1.51] | different |
| 0.02 | all corruptions | k-NN (euclidean) | highest-residual view (post hoc) - mean | 0.3533 | 0.3637 | -1.04 | [-1.10, -0.97] | [-1.12, -0.96] | different |
| 0.02 | all corruptions | k-NN (euclidean) | lowest-residual view (post hoc) - mean | 0.3333 | 0.3637 | -3.04 | [-3.12, -2.97] | [-3.14, -2.96] | different |
| 0.02 | all corruptions | k-NN (euclidean) | top-radius view - bottom-radius view | 0.3702 | 0.3263 | +4.39 | [+4.28, +4.50] | [+4.26, +4.52] | different |
| 0.02 | all corruptions | k-NN (euclidean) | top-radius view - random single view (post hoc) | 0.3702 | 0.3480 | +2.23 | [+2.15, +2.30] | [+2.14, +2.31] | different |
| 0.02 | all corruptions | k-NN (euclidean) | bottom-radius view - random single view (post hoc) | 0.3263 | 0.3480 | -2.16 | [-2.24, -2.09] | [-2.25, -2.08] | different |
| 0.02 | all corruptions | k-NN (euclidean) | highest-residual view - lowest-residual view (post hoc) | 0.3533 | 0.3333 | +2.00 | [+1.92, +2.11] | [+1.90, +2.13] | different |
| 0.02 | all corruptions | k-NN (euclidean) | highest-residual view - random single view (post hoc) | 0.3533 | 0.3480 | +0.54 | [+0.47, +0.61] | [+0.46, +0.61] | different |
| 0.02 | all corruptions | k-NN (euclidean) | lowest-residual view - random single view (post hoc) | 0.3333 | 0.3480 | -1.47 | [-1.54, -1.40] | [-1.56, -1.38] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | radius-weighted - mean | 0.7862 | 0.7856 | +0.06 | [-0.02, +0.13] | [-0.03, +0.14] | equivalent |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | inverse-radius-weighted - mean | 0.7851 | 0.7856 | -0.05 | [-0.12, +0.03] | [-0.14, +0.05] | equivalent |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | residual-weighted - mean | 0.7858 | 0.7856 | +0.02 | [-0.15, +0.19] | [-0.18, +0.22] | equivalent |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | top-radius view - mean | 0.7738 | 0.7856 | -1.18 | [-1.47, -0.90] | [-1.52, -0.87] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | bottom-radius view - mean | 0.7404 | 0.7856 | -4.52 | [-4.85, -4.17] | [-4.90, -4.11] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | center view only (context) - mean | 0.7735 | 0.7856 | -1.20 | [-1.47, -0.94] | [-1.54, -0.90] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | random single view (post hoc context) - mean | 0.7650 | 0.7856 | -2.06 | [-2.35, -1.78] | [-2.40, -1.72] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | highest-residual view (post hoc) - mean | 0.7662 | 0.7856 | -1.94 | [-2.23, -1.66] | [-2.29, -1.60] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | lowest-residual view (post hoc) - mean | 0.7499 | 0.7856 | -3.57 | [-3.90, -3.23] | [-3.95, -3.18] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | top-radius view - bottom-radius view | 0.7738 | 0.7404 | +3.33 | [+2.92, +3.70] | [+2.85, +3.80] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | top-radius view - random single view (post hoc) | 0.7738 | 0.7650 | +0.88 | [+0.55, +1.21] | [+0.50, +1.29] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | bottom-radius view - random single view (post hoc) | 0.7404 | 0.7650 | -2.45 | [-2.80, -2.08] | [-2.84, -2.01] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | highest-residual view - lowest-residual view (post hoc) | 0.7662 | 0.7499 | +1.63 | [+1.24, +2.01] | [+1.19, +2.09] | different |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | highest-residual view - random single view (post hoc) | 0.7662 | 0.7650 | +0.12 | [-0.20, +0.45] | [-0.27, +0.50] | equivalent |
| 0.02 | clean test (no corruption) | k-NN (euclidean) | lowest-residual view - random single view (post hoc) | 0.7499 | 0.7650 | -1.51 | [-1.86, -1.14] | [-1.93, -1.08] | different |
| 0.02 | all corruptions | linear probe | radius-weighted - mean | 0.3675 | 0.3646 | +0.30 | [+0.28, +0.32] | [+0.28, +0.32] | equivalent |
| 0.02 | all corruptions | linear probe | inverse-radius-weighted - mean | 0.3616 | 0.3646 | -0.30 | [-0.32, -0.28] | [-0.32, -0.28] | equivalent |
| 0.02 | all corruptions | linear probe | residual-weighted - mean | 0.3753 | 0.3646 | +1.07 | [+1.03, +1.11] | [+1.02, +1.12] | different |
| 0.02 | all corruptions | linear probe | top-radius view - mean | 0.3861 | 0.3646 | +2.15 | [+2.05, +2.25] | [+2.03, +2.26] | different |
| 0.02 | all corruptions | linear probe | bottom-radius view - mean | 0.3262 | 0.3646 | -3.84 | [-3.91, -3.76] | [-3.93, -3.74] | different |
| 0.02 | all corruptions | linear probe | center view only (context) - mean | 0.3764 | 0.3646 | +1.18 | [+1.11, +1.25] | [+1.09, +1.27] | different |
| 0.02 | all corruptions | linear probe | random single view (post hoc context) - mean | 0.3533 | 0.3646 | -1.12 | [-1.18, -1.07] | [-1.19, -1.06] | different |
| 0.02 | all corruptions | linear probe | highest-residual view (post hoc) - mean | 0.3618 | 0.3646 | -0.28 | [-0.35, -0.20] | [-0.36, -0.19] | equivalent |
| 0.02 | all corruptions | linear probe | lowest-residual view (post hoc) - mean | 0.3344 | 0.3646 | -3.01 | [-3.09, -2.94] | [-3.10, -2.92] | different |
| 0.02 | all corruptions | linear probe | top-radius view - bottom-radius view | 0.3861 | 0.3262 | +5.99 | [+5.86, +6.12] | [+5.82, +6.14] | different |
| 0.02 | all corruptions | linear probe | top-radius view - random single view (post hoc) | 0.3861 | 0.3533 | +3.28 | [+3.19, +3.37] | [+3.16, +3.38] | different |
| 0.02 | all corruptions | linear probe | bottom-radius view - random single view (post hoc) | 0.3262 | 0.3533 | -2.71 | [-2.79, -2.63] | [-2.80, -2.62] | different |
| 0.02 | all corruptions | linear probe | highest-residual view - lowest-residual view (post hoc) | 0.3618 | 0.3344 | +2.74 | [+2.64, +2.84] | [+2.62, +2.85] | different |
| 0.02 | all corruptions | linear probe | highest-residual view - random single view (post hoc) | 0.3618 | 0.3533 | +0.85 | [+0.78, +0.91] | [+0.77, +0.93] | different |
| 0.02 | all corruptions | linear probe | lowest-residual view - random single view (post hoc) | 0.3344 | 0.3533 | -1.89 | [-1.96, -1.82] | [-1.97, -1.80] | different |
| 0.02 | clean test (no corruption) | linear probe | radius-weighted - mean | 0.8145 | 0.8140 | +0.05 | [-0.01, +0.11] | [-0.02, +0.12] | equivalent |
| 0.02 | clean test (no corruption) | linear probe | inverse-radius-weighted - mean | 0.8135 | 0.8140 | -0.05 | [-0.11, +0.01] | [-0.12, +0.02] | equivalent |
| 0.02 | clean test (no corruption) | linear probe | residual-weighted - mean | 0.8124 | 0.8140 | -0.16 | [-0.31, -0.02] | [-0.34, +0.01] | equivalent |
| 0.02 | clean test (no corruption) | linear probe | top-radius view - mean | 0.7969 | 0.8140 | -1.71 | [-2.00, -1.43] | [-2.05, -1.37] | different |
| 0.02 | clean test (no corruption) | linear probe | bottom-radius view - mean | 0.7575 | 0.8140 | -5.65 | [-6.00, -5.32] | [-6.05, -5.24] | different |
| 0.02 | clean test (no corruption) | linear probe | center view only (context) - mean | 0.8045 | 0.8140 | -0.95 | [-1.21, -0.69] | [-1.26, -0.64] | different |
| 0.02 | clean test (no corruption) | linear probe | random single view (post hoc context) - mean | 0.7950 | 0.8140 | -1.90 | [-2.18, -1.64] | [-2.24, -1.60] | different |
| 0.02 | clean test (no corruption) | linear probe | highest-residual view (post hoc) - mean | 0.7906 | 0.8140 | -2.34 | [-2.64, -2.07] | [-2.69, -2.02] | different |
| 0.02 | clean test (no corruption) | linear probe | lowest-residual view (post hoc) - mean | 0.7724 | 0.8140 | -4.16 | [-4.50, -3.82] | [-4.55, -3.76] | different |
| 0.02 | clean test (no corruption) | linear probe | top-radius view - bottom-radius view | 0.7969 | 0.7575 | +3.94 | [+3.53, +4.32] | [+3.45, +4.40] | different |
| 0.02 | clean test (no corruption) | linear probe | top-radius view - random single view (post hoc) | 0.7969 | 0.7950 | +0.18 | [-0.13, +0.51] | [-0.18, +0.56] | inconclusive |
| 0.02 | clean test (no corruption) | linear probe | bottom-radius view - random single view (post hoc) | 0.7575 | 0.7950 | -3.75 | [-4.10, -3.38] | [-4.17, -3.33] | different |
| 0.02 | clean test (no corruption) | linear probe | highest-residual view - lowest-residual view (post hoc) | 0.7906 | 0.7724 | +1.83 | [+1.41, +2.22] | [+1.30, +2.28] | different |
| 0.02 | clean test (no corruption) | linear probe | highest-residual view - random single view (post hoc) | 0.7906 | 0.7950 | -0.44 | [-0.77, -0.12] | [-0.82, -0.05] | different |
| 0.02 | clean test (no corruption) | linear probe | lowest-residual view - random single view (post hoc) | 0.7724 | 0.7950 | -2.27 | [-2.63, -1.90] | [-2.71, -1.84] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | radius-weighted - mean | 0.3704 | 0.3675 | +0.28 | [+0.27, +0.30] | [+0.26, +0.30] | equivalent |
| 0.02 | all corruptions | MLP (1 hidden layer) | inverse-radius-weighted - mean | 0.3648 | 0.3675 | -0.27 | [-0.29, -0.26] | [-0.29, -0.26] | equivalent |
| 0.02 | all corruptions | MLP (1 hidden layer) | residual-weighted - mean | 0.3756 | 0.3675 | +0.80 | [+0.76, +0.85] | [+0.75, +0.86] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | top-radius view - mean | 0.3791 | 0.3675 | +1.15 | [+1.06, +1.25] | [+1.04, +1.27] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | bottom-radius view - mean | 0.3279 | 0.3675 | -3.96 | [-4.04, -3.88] | [-4.05, -3.87] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | center view only (context) - mean | 0.3747 | 0.3675 | +0.72 | [+0.64, +0.79] | [+0.63, +0.80] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | random single view (post hoc context) - mean | 0.3526 | 0.3675 | -1.49 | [-1.55, -1.43] | [-1.56, -1.41] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | highest-residual view (post hoc) - mean | 0.3578 | 0.3675 | -0.98 | [-1.04, -0.90] | [-1.05, -0.88] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | lowest-residual view (post hoc) - mean | 0.3370 | 0.3675 | -3.05 | [-3.13, -2.98] | [-3.15, -2.96] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | top-radius view - bottom-radius view | 0.3791 | 0.3279 | +5.11 | [+4.98, +5.24] | [+4.96, +5.27] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | top-radius view - random single view (post hoc) | 0.3791 | 0.3526 | +2.64 | [+2.56, +2.72] | [+2.54, +2.74] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | bottom-radius view - random single view (post hoc) | 0.3279 | 0.3526 | -2.47 | [-2.55, -2.40] | [-2.56, -2.38] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | highest-residual view - lowest-residual view (post hoc) | 0.3578 | 0.3370 | +2.08 | [+1.98, +2.18] | [+1.96, +2.20] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | highest-residual view - random single view (post hoc) | 0.3578 | 0.3526 | +0.51 | [+0.45, +0.58] | [+0.44, +0.60] | different |
| 0.02 | all corruptions | MLP (1 hidden layer) | lowest-residual view - random single view (post hoc) | 0.3370 | 0.3526 | -1.56 | [-1.64, -1.50] | [-1.65, -1.48] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | radius-weighted - mean | 0.8109 | 0.8108 | +0.01 | [-0.06, +0.08] | [-0.07, +0.09] | equivalent |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | inverse-radius-weighted - mean | 0.8103 | 0.8108 | -0.05 | [-0.12, +0.02] | [-0.13, +0.03] | equivalent |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | residual-weighted - mean | 0.8108 | 0.8108 | +0.00 | [-0.16, +0.15] | [-0.19, +0.18] | equivalent |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - mean | 0.7999 | 0.8108 | -1.09 | [-1.39, -0.80] | [-1.45, -0.76] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | bottom-radius view - mean | 0.7552 | 0.8108 | -5.57 | [-5.93, -5.21] | [-6.01, -5.15] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | center view only (context) - mean | 0.8030 | 0.8108 | -0.78 | [-1.04, -0.51] | [-1.09, -0.45] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | random single view (post hoc context) - mean | 0.7928 | 0.8108 | -1.80 | [-2.10, -1.52] | [-2.15, -1.46] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view (post hoc) - mean | 0.7898 | 0.8108 | -2.11 | [-2.43, -1.81] | [-2.47, -1.75] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | lowest-residual view (post hoc) - mean | 0.7703 | 0.8108 | -4.05 | [-4.42, -3.69] | [-4.50, -3.63] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - bottom-radius view | 0.7999 | 0.7552 | +4.48 | [+4.06, +4.89] | [+3.95, +4.98] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - random single view (post hoc) | 0.7999 | 0.7928 | +0.71 | [+0.37, +1.05] | [+0.31, +1.11] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | bottom-radius view - random single view (post hoc) | 0.7552 | 0.7928 | -3.76 | [-4.12, -3.38] | [-4.21, -3.30] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view - lowest-residual view (post hoc) | 0.7898 | 0.7703 | +1.95 | [+1.52, +2.35] | [+1.45, +2.43] | different |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view - random single view (post hoc) | 0.7898 | 0.7928 | -0.30 | [-0.63, +0.03] | [-0.71, +0.09] | inconclusive |
| 0.02 | clean test (no corruption) | MLP (1 hidden layer) | lowest-residual view - random single view (post hoc) | 0.7703 | 0.7928 | -2.25 | [-2.62, -1.87] | [-2.69, -1.79] | different |
| 0.05 | all corruptions | k-NN (euclidean) | radius-weighted - mean | 0.3725 | 0.3713 | +0.12 | [+0.10, +0.13] | [+0.10, +0.14] | equivalent |
| 0.05 | all corruptions | k-NN (euclidean) | inverse-radius-weighted - mean | 0.3700 | 0.3713 | -0.13 | [-0.15, -0.11] | [-0.15, -0.11] | equivalent |
| 0.05 | all corruptions | k-NN (euclidean) | residual-weighted - mean | 0.3755 | 0.3713 | +0.42 | [+0.38, +0.47] | [+0.37, +0.48] | equivalent |
| 0.05 | all corruptions | k-NN (euclidean) | top-radius view - mean | 0.3691 | 0.3713 | -0.22 | [-0.30, -0.14] | [-0.31, -0.13] | equivalent |
| 0.05 | all corruptions | k-NN (euclidean) | bottom-radius view - mean | 0.3374 | 0.3713 | -3.39 | [-3.47, -3.31] | [-3.49, -3.29] | different |
| 0.05 | all corruptions | k-NN (euclidean) | center view only (context) - mean | 0.3742 | 0.3713 | +0.28 | [+0.22, +0.35] | [+0.21, +0.36] | equivalent |
| 0.05 | all corruptions | k-NN (euclidean) | random single view (post hoc context) - mean | 0.3522 | 0.3713 | -1.91 | [-1.97, -1.85] | [-1.98, -1.84] | different |
| 0.05 | all corruptions | k-NN (euclidean) | highest-residual view (post hoc) - mean | 0.3606 | 0.3713 | -1.07 | [-1.14, -1.01] | [-1.15, -0.99] | different |
| 0.05 | all corruptions | k-NN (euclidean) | lowest-residual view (post hoc) - mean | 0.3408 | 0.3713 | -3.05 | [-3.14, -2.97] | [-3.16, -2.96] | different |
| 0.05 | all corruptions | k-NN (euclidean) | top-radius view - bottom-radius view | 0.3691 | 0.3374 | +3.17 | [+3.07, +3.28] | [+3.04, +3.31] | different |
| 0.05 | all corruptions | k-NN (euclidean) | top-radius view - random single view (post hoc) | 0.3691 | 0.3522 | +1.69 | [+1.62, +1.77] | [+1.60, +1.78] | different |
| 0.05 | all corruptions | k-NN (euclidean) | bottom-radius view - random single view (post hoc) | 0.3374 | 0.3522 | -1.48 | [-1.56, -1.41] | [-1.57, -1.39] | different |
| 0.05 | all corruptions | k-NN (euclidean) | highest-residual view - lowest-residual view (post hoc) | 0.3606 | 0.3408 | +1.98 | [+1.88, +2.09] | [+1.86, +2.10] | different |
| 0.05 | all corruptions | k-NN (euclidean) | highest-residual view - random single view (post hoc) | 0.3606 | 0.3522 | +0.84 | [+0.77, +0.91] | [+0.75, +0.91] | different |
| 0.05 | all corruptions | k-NN (euclidean) | lowest-residual view - random single view (post hoc) | 0.3408 | 0.3522 | -1.14 | [-1.22, -1.06] | [-1.23, -1.05] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | radius-weighted - mean | 0.7920 | 0.7921 | -0.00 | [-0.07, +0.06] | [-0.09, +0.08] | equivalent |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | inverse-radius-weighted - mean | 0.7924 | 0.7921 | +0.03 | [-0.05, +0.11] | [-0.07, +0.12] | equivalent |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | residual-weighted - mean | 0.7929 | 0.7921 | +0.08 | [-0.09, +0.27] | [-0.13, +0.29] | equivalent |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | top-radius view - mean | 0.7806 | 0.7921 | -1.15 | [-1.42, -0.85] | [-1.48, -0.80] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | bottom-radius view - mean | 0.7497 | 0.7921 | -4.24 | [-4.59, -3.87] | [-4.65, -3.79] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | center view only (context) - mean | 0.7836 | 0.7921 | -0.85 | [-1.11, -0.59] | [-1.15, -0.54] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | random single view (post hoc context) - mean | 0.7733 | 0.7921 | -1.87 | [-2.18, -1.60] | [-2.24, -1.55] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | highest-residual view (post hoc) - mean | 0.7778 | 0.7921 | -1.43 | [-1.72, -1.14] | [-1.78, -1.07] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | lowest-residual view (post hoc) - mean | 0.7511 | 0.7921 | -4.10 | [-4.44, -3.75] | [-4.50, -3.68] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | top-radius view - bottom-radius view | 0.7806 | 0.7497 | +3.09 | [+2.68, +3.52] | [+2.59, +3.57] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | top-radius view - random single view (post hoc) | 0.7806 | 0.7733 | +0.72 | [+0.40, +1.04] | [+0.34, +1.10] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | bottom-radius view - random single view (post hoc) | 0.7497 | 0.7733 | -2.37 | [-2.73, -1.98] | [-2.80, -1.90] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | highest-residual view - lowest-residual view (post hoc) | 0.7778 | 0.7511 | +2.67 | [+2.28, +3.08] | [+2.19, +3.15] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | highest-residual view - random single view (post hoc) | 0.7778 | 0.7733 | +0.45 | [+0.11, +0.77] | [+0.02, +0.85] | different |
| 0.05 | clean test (no corruption) | k-NN (euclidean) | lowest-residual view - random single view (post hoc) | 0.7511 | 0.7733 | -2.22 | [-2.60, -1.86] | [-2.66, -1.78] | different |
| 0.05 | all corruptions | linear probe | radius-weighted - mean | 0.3720 | 0.3696 | +0.24 | [+0.22, +0.25] | [+0.22, +0.25] | equivalent |
| 0.05 | all corruptions | linear probe | inverse-radius-weighted - mean | 0.3673 | 0.3696 | -0.22 | [-0.24, -0.21] | [-0.24, -0.21] | equivalent |
| 0.05 | all corruptions | linear probe | residual-weighted - mean | 0.3781 | 0.3696 | +0.85 | [+0.80, +0.90] | [+0.79, +0.90] | different |
| 0.05 | all corruptions | linear probe | top-radius view - mean | 0.3856 | 0.3696 | +1.60 | [+1.50, +1.70] | [+1.48, +1.73] | different |
| 0.05 | all corruptions | linear probe | bottom-radius view - mean | 0.3282 | 0.3696 | -4.14 | [-4.22, -4.06] | [-4.24, -4.05] | different |
| 0.05 | all corruptions | linear probe | center view only (context) - mean | 0.3757 | 0.3696 | +0.61 | [+0.53, +0.69] | [+0.52, +0.70] | different |
| 0.05 | all corruptions | linear probe | random single view (post hoc context) - mean | 0.3534 | 0.3696 | -1.62 | [-1.68, -1.56] | [-1.70, -1.55] | different |
| 0.05 | all corruptions | linear probe | highest-residual view (post hoc) - mean | 0.3665 | 0.3696 | -0.31 | [-0.39, -0.23] | [-0.40, -0.21] | equivalent |
| 0.05 | all corruptions | linear probe | lowest-residual view (post hoc) - mean | 0.3359 | 0.3696 | -3.37 | [-3.45, -3.29] | [-3.47, -3.28] | different |
| 0.05 | all corruptions | linear probe | top-radius view - bottom-radius view | 0.3856 | 0.3282 | +5.74 | [+5.60, +5.89] | [+5.58, +5.91] | different |
| 0.05 | all corruptions | linear probe | top-radius view - random single view (post hoc) | 0.3856 | 0.3534 | +3.22 | [+3.13, +3.31] | [+3.11, +3.33] | different |
| 0.05 | all corruptions | linear probe | bottom-radius view - random single view (post hoc) | 0.3282 | 0.3534 | -2.52 | [-2.61, -2.44] | [-2.62, -2.42] | different |
| 0.05 | all corruptions | linear probe | highest-residual view - lowest-residual view (post hoc) | 0.3665 | 0.3359 | +3.06 | [+2.95, +3.17] | [+2.93, +3.19] | different |
| 0.05 | all corruptions | linear probe | highest-residual view - random single view (post hoc) | 0.3665 | 0.3534 | +1.31 | [+1.24, +1.39] | [+1.22, +1.40] | different |
| 0.05 | all corruptions | linear probe | lowest-residual view - random single view (post hoc) | 0.3359 | 0.3534 | -1.75 | [-1.82, -1.67] | [-1.84, -1.66] | different |
| 0.05 | clean test (no corruption) | linear probe | radius-weighted - mean | 0.8258 | 0.8258 | +0.00 | [-0.05, +0.06] | [-0.06, +0.07] | equivalent |
| 0.05 | clean test (no corruption) | linear probe | inverse-radius-weighted - mean | 0.8256 | 0.8258 | -0.02 | [-0.08, +0.04] | [-0.09, +0.05] | equivalent |
| 0.05 | clean test (no corruption) | linear probe | residual-weighted - mean | 0.8235 | 0.8258 | -0.23 | [-0.41, -0.07] | [-0.44, -0.03] | equivalent |
| 0.05 | clean test (no corruption) | linear probe | top-radius view - mean | 0.8080 | 0.8258 | -1.78 | [-2.09, -1.49] | [-2.13, -1.43] | different |
| 0.05 | clean test (no corruption) | linear probe | bottom-radius view - mean | 0.7654 | 0.8258 | -6.04 | [-6.40, -5.68] | [-6.49, -5.61] | different |
| 0.05 | clean test (no corruption) | linear probe | center view only (context) - mean | 0.8169 | 0.8258 | -0.89 | [-1.15, -0.63] | [-1.20, -0.59] | different |
| 0.05 | clean test (no corruption) | linear probe | random single view (post hoc context) - mean | 0.8015 | 0.8258 | -2.43 | [-2.72, -2.14] | [-2.77, -2.08] | different |
| 0.05 | clean test (no corruption) | linear probe | highest-residual view (post hoc) - mean | 0.8044 | 0.8258 | -2.14 | [-2.45, -1.86] | [-2.51, -1.80] | different |
| 0.05 | clean test (no corruption) | linear probe | lowest-residual view (post hoc) - mean | 0.7730 | 0.8258 | -5.28 | [-5.62, -4.94] | [-5.70, -4.85] | different |
| 0.05 | clean test (no corruption) | linear probe | top-radius view - bottom-radius view | 0.8080 | 0.7654 | +4.26 | [+3.81, +4.69] | [+3.72, +4.79] | different |
| 0.05 | clean test (no corruption) | linear probe | top-radius view - random single view (post hoc) | 0.8080 | 0.8015 | +0.65 | [+0.31, +0.99] | [+0.25, +1.04] | different |
| 0.05 | clean test (no corruption) | linear probe | bottom-radius view - random single view (post hoc) | 0.7654 | 0.8015 | -3.61 | [-4.00, -3.24] | [-4.07, -3.17] | different |
| 0.05 | clean test (no corruption) | linear probe | highest-residual view - lowest-residual view (post hoc) | 0.8044 | 0.7730 | +3.14 | [+2.72, +3.54] | [+2.62, +3.60] | different |
| 0.05 | clean test (no corruption) | linear probe | highest-residual view - random single view (post hoc) | 0.8044 | 0.8015 | +0.29 | [-0.05, +0.60] | [-0.11, +0.66] | inconclusive |
| 0.05 | clean test (no corruption) | linear probe | lowest-residual view - random single view (post hoc) | 0.7730 | 0.8015 | -2.85 | [-3.24, -2.51] | [-3.29, -2.44] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | radius-weighted - mean | 0.3768 | 0.3748 | +0.20 | [+0.18, +0.21] | [+0.18, +0.21] | equivalent |
| 0.05 | all corruptions | MLP (1 hidden layer) | inverse-radius-weighted - mean | 0.3727 | 0.3748 | -0.21 | [-0.22, -0.20] | [-0.23, -0.19] | equivalent |
| 0.05 | all corruptions | MLP (1 hidden layer) | residual-weighted - mean | 0.3804 | 0.3748 | +0.56 | [+0.51, +0.61] | [+0.50, +0.62] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | top-radius view - mean | 0.3765 | 0.3748 | +0.17 | [+0.08, +0.26] | [+0.06, +0.28] | equivalent |
| 0.05 | all corruptions | MLP (1 hidden layer) | bottom-radius view - mean | 0.3356 | 0.3748 | -3.92 | [-4.00, -3.84] | [-4.01, -3.82] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | center view only (context) - mean | 0.3800 | 0.3748 | +0.52 | [+0.45, +0.59] | [+0.43, +0.60] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | random single view (post hoc context) - mean | 0.3549 | 0.3748 | -1.99 | [-2.06, -1.93] | [-2.07, -1.92] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | highest-residual view (post hoc) - mean | 0.3633 | 0.3748 | -1.15 | [-1.23, -1.07] | [-1.24, -1.06] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | lowest-residual view (post hoc) - mean | 0.3408 | 0.3748 | -3.40 | [-3.47, -3.32] | [-3.48, -3.30] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | top-radius view - bottom-radius view | 0.3765 | 0.3356 | +4.08 | [+3.96, +4.21] | [+3.94, +4.23] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | top-radius view - random single view (post hoc) | 0.3765 | 0.3549 | +2.16 | [+2.08, +2.24] | [+2.06, +2.26] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | bottom-radius view - random single view (post hoc) | 0.3356 | 0.3549 | -1.92 | [-2.01, -1.84] | [-2.02, -1.83] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | highest-residual view - lowest-residual view (post hoc) | 0.3633 | 0.3408 | +2.25 | [+2.14, +2.35] | [+2.12, +2.37] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | highest-residual view - random single view (post hoc) | 0.3633 | 0.3549 | +0.84 | [+0.77, +0.92] | [+0.76, +0.93] | different |
| 0.05 | all corruptions | MLP (1 hidden layer) | lowest-residual view - random single view (post hoc) | 0.3408 | 0.3549 | -1.40 | [-1.47, -1.32] | [-1.49, -1.31] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | radius-weighted - mean | 0.8254 | 0.8250 | +0.04 | [-0.01, +0.09] | [-0.03, +0.10] | equivalent |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | inverse-radius-weighted - mean | 0.8247 | 0.8250 | -0.03 | [-0.10, +0.03] | [-0.12, +0.05] | equivalent |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | residual-weighted - mean | 0.8230 | 0.8250 | -0.20 | [-0.35, -0.04] | [-0.39, -0.01] | equivalent |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - mean | 0.8070 | 0.8250 | -1.80 | [-2.11, -1.51] | [-2.17, -1.45] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | bottom-radius view - mean | 0.7671 | 0.8250 | -5.79 | [-6.17, -5.45] | [-6.23, -5.38] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | center view only (context) - mean | 0.8145 | 0.8250 | -1.05 | [-1.32, -0.78] | [-1.37, -0.72] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | random single view (post hoc context) - mean | 0.8030 | 0.8250 | -2.20 | [-2.50, -1.93] | [-2.55, -1.88] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view (post hoc) - mean | 0.8028 | 0.8250 | -2.22 | [-2.51, -1.94] | [-2.58, -1.89] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | lowest-residual view (post hoc) - mean | 0.7759 | 0.8250 | -4.91 | [-5.26, -4.58] | [-5.36, -4.52] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - bottom-radius view | 0.8070 | 0.7671 | +3.99 | [+3.55, +4.43] | [+3.45, +4.51] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - random single view (post hoc) | 0.8070 | 0.8030 | +0.40 | [+0.05, +0.75] | [-0.02, +0.83] | inconclusive |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | bottom-radius view - random single view (post hoc) | 0.7671 | 0.8030 | -3.59 | [-3.98, -3.21] | [-4.07, -3.14] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view - lowest-residual view (post hoc) | 0.8028 | 0.7759 | +2.69 | [+2.29, +3.12] | [+2.20, +3.16] | different |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view - random single view (post hoc) | 0.8028 | 0.8030 | -0.01 | [-0.34, +0.32] | [-0.40, +0.38] | equivalent |
| 0.05 | clean test (no corruption) | MLP (1 hidden layer) | lowest-residual view - random single view (post hoc) | 0.7759 | 0.8030 | -2.70 | [-3.08, -2.32] | [-3.15, -2.23] | different |
| 0.1 | all corruptions | k-NN (euclidean) | radius-weighted - mean | 0.3672 | 0.3663 | +0.08 | [+0.07, +0.10] | [+0.06, +0.10] | equivalent |
| 0.1 | all corruptions | k-NN (euclidean) | inverse-radius-weighted - mean | 0.3654 | 0.3663 | -0.10 | [-0.11, -0.08] | [-0.12, -0.08] | equivalent |
| 0.1 | all corruptions | k-NN (euclidean) | residual-weighted - mean | 0.3675 | 0.3663 | +0.12 | [+0.07, +0.16] | [+0.06, +0.17] | equivalent |
| 0.1 | all corruptions | k-NN (euclidean) | top-radius view - mean | 0.3606 | 0.3663 | -0.57 | [-0.65, -0.50] | [-0.66, -0.49] | different |
| 0.1 | all corruptions | k-NN (euclidean) | bottom-radius view - mean | 0.3346 | 0.3663 | -3.17 | [-3.26, -3.09] | [-3.27, -3.07] | different |
| 0.1 | all corruptions | k-NN (euclidean) | center view only (context) - mean | 0.3716 | 0.3663 | +0.53 | [+0.46, +0.60] | [+0.44, +0.61] | different |
| 0.1 | all corruptions | k-NN (euclidean) | random single view (post hoc context) - mean | 0.3463 | 0.3663 | -2.00 | [-2.06, -1.94] | [-2.08, -1.93] | different |
| 0.1 | all corruptions | k-NN (euclidean) | highest-residual view (post hoc) - mean | 0.3521 | 0.3663 | -1.43 | [-1.49, -1.37] | [-1.51, -1.36] | different |
| 0.1 | all corruptions | k-NN (euclidean) | lowest-residual view (post hoc) - mean | 0.3390 | 0.3663 | -2.73 | [-2.81, -2.65] | [-2.82, -2.63] | different |
| 0.1 | all corruptions | k-NN (euclidean) | top-radius view - bottom-radius view | 0.3606 | 0.3346 | +2.60 | [+2.49, +2.71] | [+2.46, +2.73] | different |
| 0.1 | all corruptions | k-NN (euclidean) | top-radius view - random single view (post hoc) | 0.3606 | 0.3463 | +1.43 | [+1.35, +1.50] | [+1.34, +1.52] | different |
| 0.1 | all corruptions | k-NN (euclidean) | bottom-radius view - random single view (post hoc) | 0.3346 | 0.3463 | -1.17 | [-1.25, -1.10] | [-1.26, -1.08] | different |
| 0.1 | all corruptions | k-NN (euclidean) | highest-residual view - lowest-residual view (post hoc) | 0.3521 | 0.3390 | +1.30 | [+1.20, +1.40] | [+1.18, +1.42] | different |
| 0.1 | all corruptions | k-NN (euclidean) | highest-residual view - random single view (post hoc) | 0.3521 | 0.3463 | +0.57 | [+0.50, +0.64] | [+0.49, +0.65] | different |
| 0.1 | all corruptions | k-NN (euclidean) | lowest-residual view - random single view (post hoc) | 0.3390 | 0.3463 | -0.73 | [-0.81, -0.65] | [-0.82, -0.63] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | radius-weighted - mean | 0.7925 | 0.7928 | -0.02 | [-0.10, +0.05] | [-0.11, +0.06] | equivalent |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | inverse-radius-weighted - mean | 0.7933 | 0.7928 | +0.05 | [-0.02, +0.11] | [-0.03, +0.13] | equivalent |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | residual-weighted - mean | 0.7895 | 0.7928 | -0.33 | [-0.54, -0.15] | [-0.56, -0.11] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | top-radius view - mean | 0.7789 | 0.7928 | -1.39 | [-1.70, -1.12] | [-1.74, -1.07] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | bottom-radius view - mean | 0.7495 | 0.7928 | -4.33 | [-4.70, -3.97] | [-4.76, -3.92] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | center view only (context) - mean | 0.7845 | 0.7928 | -0.83 | [-1.11, -0.58] | [-1.16, -0.53] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | random single view (post hoc context) - mean | 0.7745 | 0.7928 | -1.83 | [-2.15, -1.52] | [-2.22, -1.47] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | highest-residual view (post hoc) - mean | 0.7781 | 0.7928 | -1.47 | [-1.75, -1.18] | [-1.82, -1.12] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | lowest-residual view (post hoc) - mean | 0.7506 | 0.7928 | -4.22 | [-4.60, -3.90] | [-4.65, -3.81] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | top-radius view - bottom-radius view | 0.7789 | 0.7495 | +2.94 | [+2.55, +3.34] | [+2.45, +3.44] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | top-radius view - random single view (post hoc) | 0.7789 | 0.7745 | +0.44 | [+0.10, +0.76] | [+0.03, +0.82] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | bottom-radius view - random single view (post hoc) | 0.7495 | 0.7745 | -2.50 | [-2.88, -2.12] | [-2.96, -2.05] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | highest-residual view - lowest-residual view (post hoc) | 0.7781 | 0.7506 | +2.75 | [+2.34, +3.16] | [+2.27, +3.28] | different |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | highest-residual view - random single view (post hoc) | 0.7781 | 0.7745 | +0.36 | [+0.03, +0.70] | [-0.05, +0.76] | inconclusive |
| 0.1 | clean test (no corruption) | k-NN (euclidean) | lowest-residual view - random single view (post hoc) | 0.7506 | 0.7745 | -2.39 | [-2.77, -2.01] | [-2.85, -1.95] | different |
| 0.1 | all corruptions | linear probe | radius-weighted - mean | 0.3702 | 0.3684 | +0.19 | [+0.17, +0.20] | [+0.17, +0.20] | equivalent |
| 0.1 | all corruptions | linear probe | inverse-radius-weighted - mean | 0.3665 | 0.3684 | -0.18 | [-0.20, -0.17] | [-0.20, -0.17] | equivalent |
| 0.1 | all corruptions | linear probe | residual-weighted - mean | 0.3764 | 0.3684 | +0.80 | [+0.75, +0.86] | [+0.74, +0.87] | different |
| 0.1 | all corruptions | linear probe | top-radius view - mean | 0.3760 | 0.3684 | +0.76 | [+0.65, +0.87] | [+0.64, +0.89] | different |
| 0.1 | all corruptions | linear probe | bottom-radius view - mean | 0.3279 | 0.3684 | -4.05 | [-4.13, -3.96] | [-4.16, -3.95] | different |
| 0.1 | all corruptions | linear probe | center view only (context) - mean | 0.3776 | 0.3684 | +0.92 | [+0.83, +1.02] | [+0.81, +1.03] | different |
| 0.1 | all corruptions | linear probe | random single view (post hoc context) - mean | 0.3496 | 0.3684 | -1.87 | [-1.93, -1.81] | [-1.95, -1.80] | different |
| 0.1 | all corruptions | linear probe | highest-residual view (post hoc) - mean | 0.3651 | 0.3684 | -0.33 | [-0.41, -0.24] | [-0.42, -0.22] | equivalent |
| 0.1 | all corruptions | linear probe | lowest-residual view (post hoc) - mean | 0.3370 | 0.3684 | -3.14 | [-3.22, -3.06] | [-3.23, -3.04] | different |
| 0.1 | all corruptions | linear probe | top-radius view - bottom-radius view | 0.3760 | 0.3279 | +4.81 | [+4.66, +4.96] | [+4.63, +4.99] | different |
| 0.1 | all corruptions | linear probe | top-radius view - random single view (post hoc) | 0.3760 | 0.3496 | +2.63 | [+2.54, +2.73] | [+2.52, +2.74] | different |
| 0.1 | all corruptions | linear probe | bottom-radius view - random single view (post hoc) | 0.3279 | 0.3496 | -2.18 | [-2.27, -2.09] | [-2.29, -2.08] | different |
| 0.1 | all corruptions | linear probe | highest-residual view - lowest-residual view (post hoc) | 0.3651 | 0.3370 | +2.81 | [+2.70, +2.92] | [+2.68, +2.95] | different |
| 0.1 | all corruptions | linear probe | highest-residual view - random single view (post hoc) | 0.3651 | 0.3496 | +1.54 | [+1.47, +1.62] | [+1.45, +1.64] | different |
| 0.1 | all corruptions | linear probe | lowest-residual view - random single view (post hoc) | 0.3370 | 0.3496 | -1.27 | [-1.35, -1.19] | [-1.37, -1.17] | different |
| 0.1 | clean test (no corruption) | linear probe | radius-weighted - mean | 0.8215 | 0.8207 | +0.08 | [+0.02, +0.15] | [+0.01, +0.15] | equivalent |
| 0.1 | clean test (no corruption) | linear probe | inverse-radius-weighted - mean | 0.8191 | 0.8207 | -0.15 | [-0.22, -0.09] | [-0.23, -0.07] | equivalent |
| 0.1 | clean test (no corruption) | linear probe | residual-weighted - mean | 0.8180 | 0.8207 | -0.27 | [-0.44, -0.09] | [-0.47, -0.06] | equivalent |
| 0.1 | clean test (no corruption) | linear probe | top-radius view - mean | 0.8029 | 0.8207 | -1.78 | [-2.07, -1.48] | [-2.13, -1.44] | different |
| 0.1 | clean test (no corruption) | linear probe | bottom-radius view - mean | 0.7532 | 0.8207 | -6.74 | [-7.10, -6.38] | [-7.17, -6.29] | different |
| 0.1 | clean test (no corruption) | linear probe | center view only (context) - mean | 0.8104 | 0.8207 | -1.03 | [-1.30, -0.74] | [-1.38, -0.70] | different |
| 0.1 | clean test (no corruption) | linear probe | random single view (post hoc context) - mean | 0.7947 | 0.8207 | -2.59 | [-2.88, -2.30] | [-2.96, -2.25] | different |
| 0.1 | clean test (no corruption) | linear probe | highest-residual view (post hoc) - mean | 0.7993 | 0.8207 | -2.14 | [-2.41, -1.86] | [-2.46, -1.80] | different |
| 0.1 | clean test (no corruption) | linear probe | lowest-residual view (post hoc) - mean | 0.7642 | 0.8207 | -5.64 | [-6.02, -5.29] | [-6.08, -5.22] | different |
| 0.1 | clean test (no corruption) | linear probe | top-radius view - bottom-radius view | 0.8029 | 0.7532 | +4.96 | [+4.54, +5.40] | [+4.43, +5.48] | different |
| 0.1 | clean test (no corruption) | linear probe | top-radius view - random single view (post hoc) | 0.8029 | 0.7947 | +0.81 | [+0.46, +1.14] | [+0.38, +1.20] | different |
| 0.1 | clean test (no corruption) | linear probe | bottom-radius view - random single view (post hoc) | 0.7532 | 0.7947 | -4.15 | [-4.53, -3.77] | [-4.61, -3.70] | different |
| 0.1 | clean test (no corruption) | linear probe | highest-residual view - lowest-residual view (post hoc) | 0.7993 | 0.7642 | +3.50 | [+3.08, +3.93] | [+3.00, +4.03] | different |
| 0.1 | clean test (no corruption) | linear probe | highest-residual view - random single view (post hoc) | 0.7993 | 0.7947 | +0.45 | [+0.12, +0.79] | [+0.08, +0.85] | different |
| 0.1 | clean test (no corruption) | linear probe | lowest-residual view - random single view (post hoc) | 0.7642 | 0.7947 | -3.05 | [-3.45, -2.68] | [-3.52, -2.58] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | radius-weighted - mean | 0.3790 | 0.3777 | +0.12 | [+0.11, +0.14] | [+0.11, +0.14] | equivalent |
| 0.1 | all corruptions | MLP (1 hidden layer) | inverse-radius-weighted - mean | 0.3764 | 0.3777 | -0.13 | [-0.15, -0.12] | [-0.15, -0.12] | equivalent |
| 0.1 | all corruptions | MLP (1 hidden layer) | residual-weighted - mean | 0.3812 | 0.3777 | +0.34 | [+0.29, +0.40] | [+0.28, +0.40] | equivalent |
| 0.1 | all corruptions | MLP (1 hidden layer) | top-radius view - mean | 0.3738 | 0.3777 | -0.40 | [-0.49, -0.30] | [-0.51, -0.28] | equivalent |
| 0.1 | all corruptions | MLP (1 hidden layer) | bottom-radius view - mean | 0.3368 | 0.3777 | -4.09 | [-4.17, -4.01] | [-4.19, -3.99] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | center view only (context) - mean | 0.3867 | 0.3777 | +0.89 | [+0.81, +0.97] | [+0.79, +0.99] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | random single view (post hoc context) - mean | 0.3538 | 0.3777 | -2.39 | [-2.46, -2.33] | [-2.47, -2.32] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | highest-residual view (post hoc) - mean | 0.3643 | 0.3777 | -1.35 | [-1.43, -1.27] | [-1.45, -1.25] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | lowest-residual view (post hoc) - mean | 0.3429 | 0.3777 | -3.48 | [-3.56, -3.39] | [-3.58, -3.38] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | top-radius view - bottom-radius view | 0.3738 | 0.3368 | +3.69 | [+3.56, +3.82] | [+3.54, +3.85] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | top-radius view - random single view (post hoc) | 0.3738 | 0.3538 | +2.00 | [+1.91, +2.08] | [+1.89, +2.09] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | bottom-radius view - random single view (post hoc) | 0.3368 | 0.3538 | -1.70 | [-1.78, -1.62] | [-1.80, -1.60] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | highest-residual view - lowest-residual view (post hoc) | 0.3643 | 0.3429 | +2.14 | [+2.02, +2.25] | [+2.00, +2.27] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | highest-residual view - random single view (post hoc) | 0.3643 | 0.3538 | +1.05 | [+0.97, +1.12] | [+0.96, +1.14] | different |
| 0.1 | all corruptions | MLP (1 hidden layer) | lowest-residual view - random single view (post hoc) | 0.3429 | 0.3538 | -1.09 | [-1.16, -1.01] | [-1.18, -0.99] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | radius-weighted - mean | 0.8272 | 0.8267 | +0.05 | [+0.00, +0.10] | [-0.00, +0.11] | equivalent |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | inverse-radius-weighted - mean | 0.8262 | 0.8267 | -0.05 | [-0.11, +0.01] | [-0.12, +0.02] | equivalent |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | residual-weighted - mean | 0.8223 | 0.8267 | -0.43 | [-0.61, -0.27] | [-0.65, -0.24] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - mean | 0.8089 | 0.8267 | -1.78 | [-2.09, -1.48] | [-2.14, -1.43] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | bottom-radius view - mean | 0.7661 | 0.8267 | -6.05 | [-6.43, -5.68] | [-6.51, -5.60] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | center view only (context) - mean | 0.8172 | 0.8267 | -0.95 | [-1.22, -0.70] | [-1.27, -0.65] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | random single view (post hoc context) - mean | 0.8006 | 0.8267 | -2.61 | [-2.91, -2.31] | [-2.98, -2.25] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view (post hoc) - mean | 0.8064 | 0.8267 | -2.02 | [-2.31, -1.74] | [-2.36, -1.68] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | lowest-residual view (post hoc) - mean | 0.7738 | 0.8267 | -5.28 | [-5.63, -4.91] | [-5.72, -4.85] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - bottom-radius view | 0.8089 | 0.7661 | +4.27 | [+3.86, +4.70] | [+3.79, +4.75] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | top-radius view - random single view (post hoc) | 0.8089 | 0.8006 | +0.83 | [+0.50, +1.15] | [+0.45, +1.21] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | bottom-radius view - random single view (post hoc) | 0.7661 | 0.8006 | -3.45 | [-3.83, -3.07] | [-3.88, -3.00] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view - lowest-residual view (post hoc) | 0.8064 | 0.7738 | +3.26 | [+2.84, +3.68] | [+2.76, +3.77] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | highest-residual view - random single view (post hoc) | 0.8064 | 0.8006 | +0.58 | [+0.25, +0.93] | [+0.17, +0.98] | different |
| 0.1 | clean test (no corruption) | MLP (1 hidden layer) | lowest-residual view - random single view (post hoc) | 0.7738 | 0.8006 | -2.68 | [-3.06, -2.32] | [-3.12, -2.24] | different |

**A. Geometry** (all corruptions and null)

| lambda | level | radial share F_rad | 95% CI | median r_x / r_c | radius-change share (post hoc) | 95% CI (radius change) | median cos(u_x, u_c) (post hoc) |
|---|---|---|---|---|---|---|---|
| 0.02 | all corruptions | 0.5219 | [0.5201, 0.5235] | 0.915 | 0.0549 | [0.0541, 0.0557] | 0.423 |
| 0.02 | null: clean image -> other clean image | 0.5174 | [0.5154, 0.5195] |  | 0.0446 | [0.0438, 0.0454] | -0.057 |
| 0.05 | all corruptions | 0.4877 | [0.4864, 0.4892] | 0.931 | 0.0361 | [0.0356, 0.0366] | 0.342 |
| 0.05 | null: clean image -> other clean image | 0.5072 | [0.5056, 0.5089] |  | 0.0314 | [0.0309, 0.0320] | -0.033 |
| 0.1 | all corruptions | 0.4827 | [0.4813, 0.4839] | 0.931 | 0.0283 | [0.0279, 0.0287] | 0.287 |
| 0.1 | null: clean image -> other clean image | 0.5035 | [0.5020, 0.5051] |  | 0.0234 | [0.0230, 0.0239] | -0.022 |

**C. Severity** (mean over corruptions)

| lambda | corruption | Spearman(res, severity), signed | held-out Spearman: direction | held-out Spearman: radius | held-out Spearman: direction + radius | direction + radius - direction | verdict |
|---|---|---|---|---|---|---|---|
| 0.02 | mean over corruptions | -0.192 | +0.909 | +0.270 | +0.909 | +0.001 |  |
| 0.05 | mean over corruptions | -0.210 | +0.909 | +0.358 | +0.910 | +0.001 |  |
| 0.1 | mean over corruptions | -0.431 | +0.906 | +0.457 | +0.908 | +0.002 |  |
<!-- END GENERATED: analyze.view_summary -->

## Interpretation

## Next steps

## Prediction scorecard (pre-registered predictions, `preregistration.md`)

| Prediction | Outcome |
|---|---|
| A1: displacement mostly tangential, `F_rad < 0.10` | **Not supported.** `F_rad` is 0.48 to 0.52 over all corruptions (per corruption 0.25 to 0.54 at lambda 0.05 / 0.1), the same as the null displacement between two different clean images (0.50 to 0.52). |
| A2: `F_rad` grows with severity for noise, blur and JPEG but stays below 0.5 | **Not supported as stated** (lambda 0.05 values). It grows with severity for brightness (0.08 -> 0.31), contrast (0.15 -> 0.47) and JPEG (0.33 -> 0.53), not monotonically for noise (0.46 -> 0.55 -> 0.54), and exceeds 0.5 for noise and JPEG. |
| A3: null radial share similar to weak corruptions | **Not supported:** the null (0.50) is above the weakest corruptions (brightness 0.25, contrast 0.39). |
| B1: oracle radius equivalent (within 0.5 points) for k-NN and MLP, recovery CI includes 0 and stays below 0.10 | **Supported for the equivalence and the bound, not for "includes 0":** k-NN +0.32 to +0.45 and MLP +0.14 to +0.19 points, all equivalent; the recovered share of the 41 to 44 point corruption loss is 0.3% to 1.1%, CIs excluding 0 but far below 0.10. |
| B2: linear probe small non-zero effect, below 1 point, in the direction "oracle radius helps" | **Supported:** +0.47 to +0.56 overall (`different` at lambda 0.02 and 0.05, equivalent at 0.1), up to +1.2 points for single corruptions (speckle noise, lambda 0.05). |
| B3: oracle direction within 0.5 points of the clean accuracy | **Supported:** all readers, all lambdas, |difference| <= 0.3 (equivalent). |
| B4: scaled constant equivalent to the baseline | **Supported:** +0.09 to +0.40, equivalent. |
| C1: `|Spearman(res, severity)| >= 0.3` for noise, blur, JPEG, below 0.3 for most others | **Not supported, lambda dependent.** Signed Spearman is negative almost everywhere; blur -0.45 to -0.67 at all lambdas; noise between -0.15 and +0.12 at lambda 0.02 / 0.05 but -0.3 to -0.6 at 0.1; JPEG -0.21 to -0.38; contrast and pixelate -0.33 to -0.40 at lambda 0.05 / 0.1 although predicted below 0.3. |
| C2: direction predicts severity much better than the radius, and direction + radius adds at most 0.01 for most corruptions | **Supported (stronger than predicted):** held-out Spearman direction 0.91, radius model 0.27 to 0.46 (mean over corruptions), direction + radius 0.91; the difference is +0.000 to +0.006 and `does not add` for all 10 corruptions at all 3 lambdas (the expected minority that `adds` did not appear). |
| D1: radius-weighted and inverse-radius weighting equivalent to the mean | **Supported:** |difference| <= 0.31 points, equivalent for every reader, lambda, on corrupted and clean images. These weightings are mild (effective number of views 7.94 of 8), so this is a weak test. |
| D2: residual-weighted equivalent or no worse than -1 point | **Holds in the letter, but the effect is a small gain on corrupted images:** +0.1 to +0.8 points for k-NN and MLP, +0.8 to +1.1 for the linear probe (`different`), on clean images within -0.43 to +0.08 (two cells `different`: k-NN -0.33 and MLP -0.43 at lambda 0.1). |
| D3: top - bottom radius view equivalent, both single views worse than the mean | **Not supported.** Top - bottom is +2.6 to +6.0 points on corrupted and +2.9 to +5.0 on clean images (`different`, all readers and lambdas). On corrupted images the top-radius view is not worse than the mean (-0.6 to +2.2 points) while the bottom-radius view is 3.2 to 4.1 points worse. |

## Interpretation

What the data say, with the post hoc parts marked:

- **The radius is not where the corruption damage is (A, B).** Restoring the clean twin's radius on a corrupted direction recovers about 1% of the corruption loss for every reader (0.3% to 1.3%), and giving the clean direction the corrupted radius costs nothing measurable. The radial share `F_rad` is about 0.5 but equals the share of a displacement between unrelated images, so the pre-registered statistic does not separate "radial displacement" from "content destroyed": when the new vector is nearly orthogonal to the old one, the component along the old direction is just `-r_c`, giving `r_c^2 / (r_c^2 + r_x^2)`, about 0.5. The post hoc radius-change share is the clearer reading: a change of radius alone explains only 2.8% to 5.5% of the squared displacement (null: 2.3% to 4.5%), and the corrupted direction keeps a median cosine of only 0.29 to 0.42 with the clean one. Corruption moves the direction; the median radius changes by 7% to 8.5% (shrinks).
- **The radius tracks corruption severity but adds nothing to the direction (C).** The radius model alone predicts severity with held-out Spearman 0.27 to 0.46 on average (up to 0.7 for Gaussian blur at lambda 0.05) but the direction model reaches 0.91, and adding the radius features changes it by at most +0.006 (`does not add` in all 30 corruption x lambda cells). So the radius is informative about severity but redundant with the direction.
- **The per-view radius does rank views by quality (D), which the pre-registration predicted it would not.** Picking the highest-radius of 8 augmented views beats the lowest-radius one by 2.6 to 6.0 points; relative to a random single view (post hoc), the lowest-radius view is 1.2 to 4.2 points worse on every setting and the highest-radius view is 0.2 to 0.9 points better on clean images (some cells inconclusive) and 1.4 to 3.3 better on corrupted ones. The direction-independent part of the radius carries part of this (post hoc): highest-residual minus lowest-residual is +1.3 to +3.5 points, and the lowest-residual view is 0.7 to 3.1 points worse than a random view. So low radius, and in particular low radius given the direction, flags a bad view; a high radius is a weaker positive signal. The averaged embedding still beats every single-view pick on clean images, and mild radius weightings of the average do not move accuracy beyond 0.3 points, so the signal is not obviously exploitable by simple weighting; stronger weighting (e.g. dropping the lowest-radius views) was not tested.
- **Net for the nuisance framing.** Consistent with earlier results: the radius carries no recoverable class or corruption-repair information for the frozen readers (A, B) and nothing about severity beyond the direction (C). Not consistent with "pure nuisance": it is a reliability signal for individual views (D), as the earlier instance-retrieval and OOD results (view-to-view signature, smaller radius for OOD images) already suggested. Together these fit "radius as a per-view confidence / quality indicator that is largely predictable from the direction but not entirely", not "radius as noise".

Caveats: one dataset and architecture; three seeds (not in the intervals); all single-view comparisons are for K = 8 augmented global views of one image, so "view quality" here means how well the augmentation preserves what the readers need, and a low radius may be a proxy for a cropping or blur property of the view rather than a learned confidence; the single-view-pick contrasts and the post hoc analyses are not corrected for multiple comparisons; the corrupted-image averages weigh all 10 corruptions and 5 severities equally, and the baseline accuracy under corruption is only about 37% so those cells are dominated by severe corruptions.

## Next steps

- Test stronger radius-based view selection (drop the k lowest-radius views of the 8, learned weights) to see whether the quality signal can be used.
- Check whether the low-radius views are low-quality because of crop size / blur severity by regressing the per-view radius on the augmentation parameters (needs the augmentation parameters in the dumps).
- Repeat part D on the ViT backbones once E5 exists, and with a second dataset.
