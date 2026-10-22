# Variant A lambda sweep: does the embedding radius follow chi_256 under SIGReg?
- **Branch / version:** `radius-marginal` (note); runs executed at `e25af9c1c86a19a8e4c479cd06ffd43e21401978` (merge of PR #23 on `master`, which includes the per-view SIGReg fix #21, the GPU augmentation #22 and `slurm/sweep.sbatch` #23)
- **Run datetime (UTC):** 2026-10-18T23:53:52Z (start of array task 3, `radius_marginal_lam0.05_s0`; the other tasks of the array started independently, see each run's `slurm-train-<jobid>.log` for its own start time and any restarts)
- **Command(s):** (job array `2186161`, submitted from the cluster checkout at the commit above)
  `LAMBDAS="0.02 0.05 0.1" SEEDS="0 1 2" sbatch --array=0-8 --cpus-per-task=4 slurm/sweep.sbatch configs/base.yaml radius_marginal`
- **Analysis (run locally on 2026-10-19T03:00Z, branch `radius-marginal` after merging `master` at `6d616fc`, which contains `analyze --sweep`):** the nine run directories were copied from the cluster without checkpoints, then
  `rsync -avz --prune-empty-dirs --include='radius_marginal_lam*_s*/' --include='metrics.jsonl' --include='config.resolved.yaml' --include='slurm-train-*.log' --exclude='*' fe.ds:<runs_dir>/ <local_runs_dir>/`
  `LEJEPA_NORM_RUNS_DIR=<local_runs_dir> LEJEPA_NORM_NOTES_DIR=<repo>/research_notes uv run python apps/analyze.py --sweep radius_marginal --note radius_marginal/lambda_sweep.md`
- **Per-run diagnostic (2026-10-19, same branch):** the final checkpoints were embedded on STL-10 `test` and analyzed per run on the cluster (checkout at `master` after #22/#24), the per-run notes copied back, then summarized:
  `for lam in 0.02 0.05 0.1; do for s in 0 1 2; do run=radius_marginal_lam${lam}_s${s}; jid=$(sbatch --parsable --time=00:30:00 --cpus-per-task=4 slurm/embed.sbatch configs/base.yaml $run last.pt --sets stl10_test); sbatch --dependency=afterok:$jid --time=00:30:00 --cpus-per-task=4 slurm/analyze.sbatch configs/base.yaml $run radius_marginal/diag/$run.md --no-figures --set eval.num_bootstrap=50; done; done`
  `rsync -avz fe.ds:<notes_dir>/radius_marginal/diag/ <local_diag_dir>/`
  `uv run python research_notes/radius_marginal/summarize_diag.py --diag-dir <local_diag_dir> --runs-dir <local_runs_dir> --note research_notes/radius_marginal/lambda_sweep.md`
- **Datasets:** STL-10 ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF](https://huggingface.co/datasets/tanganke/stl10)); pretraining on the `unlabeled` + `train` splits (labels unused), k-NN monitoring on `train` (bank) and `test`. The **canonical Stanford archive was used for all splits**; the Hugging Face mirror has no `unlabeled` split and is linked for reference only.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used)

## Question
[LeJEPA](https://arxiv.org/abs/2511.08544) regularizes the population distribution of the projector output `z` towards N(0, I_d), d = 256. Under that target the radius `r = ||z||` follows a chi_d distribution, concentrated near sqrt(d - 1/2) ~ 15.98 with standard deviation ~ 0.71. This sweep asks the marginal-radius question only (the image-conditional questions are M3b):

1. Does the radius of the trained model actually match chi_256? We measure `kappa = Var(||z||^2) / (2d)` (target 1 for an exact Gaussian) and the mean and standard deviation of `||z||`, over training and at the end.
2. How do these depend on the SIGReg weight `lambda` in {0.02, 0.05, 0.1}, and how large is the seed-to-seed spread (3 seeds per setting)?

**Hypotheses, recorded before the runs finished.** 5-epoch pilots with the per-view SIGReg loss (infrastructure checks, not results) ended with `kappa_z ~ 8`, `||z||` mean ~ 14.9 and standard deviation ~ 2.1, and `kappa_z` was no longer falling. So I expect that (a) `kappa_z` will **not** reach ~1 within 200 epochs at lambda = 0.05, and (b) larger lambda will give smaller `kappa_z`. Neither is established; the pilots used a 5-epoch schedule whose learning rate decays to zero within the run.

## Setup (deviations from the base config)
- Base config `configs/base.yaml` (ResNet-18 backbone, projector 2048-2048-256, 2 global 96 px + 4 local 48 px views, batch 256, AdamW lr 5e-4 scaled by batch/256, weight decay 0.05, 10 warmup epochs then cosine, 200 epochs, bf16 autocast with the loss in fp32).
- Only `loss.lambda` and `seed` vary: the 3 x 3 grid {0.02, 0.05, 0.1} x {0, 1, 2}, one independent job-array task per run, run names `radius_marginal_lam<lambda>_s<seed>`.
- Hardware: one A100 per run (`--gres=gpu:a100:1`), 4 CPUs, `--cpus-per-task=4`. Augmentation runs on the GPU, so `data.num_workers` does not affect the results.
- Implementation points that differ from the original Technical Design and from earlier pilots:
  - SIGReg is computed **per view** (N = per-view batch size, shared random directions, averaged over views), as in the paper and the official code, not on all views pooled (see `research_notes/sigreg_reference/reference_comparison.md`). Earlier pilots used the pooled loss and are not comparable.
  - Per-channel normalization uses statistics computed from the prepared data (`prepare_data --stage stats`): mean (0.4409, 0.4279, 0.3868), std (0.2683, 0.2610, 0.2687).
  - Augmentation is the batched GPU pipeline (`research_notes/gpu_augmentation/pipeline_comparison.md` documents its agreement with the PIL pipeline; views are ~1.3/255 brighter on average).
- Runs are preemptable (`--resume auto`); a requeued task resumes from its run directory. The run's `slurm-train-<jobid>.log` records every restart.

## Results
Training-log results of all nine runs (200/200 epochs each), generated by `apps/analyze.py --sweep`. Every `kappa_z` and `||z||` figure here is measured on **all six views of a batch pooled** (2 global 96 px + 4 local 48 px), while SIGReg acts per view. The histogram of `r` against chi_256 needs embedding dumps and is not part of this note yet.

<!-- BEGIN GENERATED: sweep.summary -->
### Sweep `radius_marginal`: per lambda (mean ± std over seeds)

Each metric is taken at the last epoch that every seed of that lambda has logged it; the `epoch` column is that epoch for `kappa_z`, and sparsely logged metrics (k-NN) may come from an earlier one.

| lambda | seeds | epoch | kappa_z | z_norm_mean | z_norm_std | kappa_h | knn_acc_h | knn_acc_z | loss_dist | loss_inv |
|---|---|---|---|---|---|---|---|---|---|---|
| target (N(0, I)) |  |  | 1 | 15.98 | 0.7071 |  |  |  |  |  |
| 0.02 | 3 | 200 | 26.29 ± 0.097 | 14.85 ± 0.0023 | 3.467 ± 0.0046 | 0.4277 ± 0.044 | 0.7934 ± 0.0046 | 0.7734 ± 0.0019 | 4.24 ± 0.0052 | 0.1611 ± 0.00061 |
| 0.05 | 3 | 200 | 20.21 ± 0.069 | 15.45 ± 0.00098 | 3.041 ± 0.0048 | 0.09602 ± 0.0063 | 0.7946 ± 0.0044 | 0.7839 ± 0.0008 | 2.036 ± 0.0023 | 0.221 ± 0.00078 |
| 0.1 | 3 | 200 | 14.64 ± 0.067 | 15.69 ± 0.0027 | 2.615 ± 0.0038 | 0.1188 ± 0.0048 | 0.7908 ± 0.001 | 0.7844 ± 0.0026 | 1.285 ± 0.0018 | 0.2722 ± 0.00063 |
<!-- END GENERATED: sweep.summary -->

<!-- BEGIN GENERATED: sweep.runs -->
### Sweep runs

| run | lambda | seed | epochs | status | kappa_z | kappa_h | z_norm_mean | z_norm_std | knn_acc_h | knn_acc_z | loss_dist | loss_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| radius_marginal_lam0.02_s0 | 0.02 | 0 | 200/200 | complete | 26.22 | 0.4616 | 14.85 | 3.463 | 0.7984 | 0.7756 | 4.239 | 0.1616 |
| radius_marginal_lam0.02_s1 | 0.02 | 1 | 200/200 | complete | 26.4 | 0.4432 | 14.85 | 3.472 | 0.7893 | 0.772 | 4.246 | 0.1604 |
| radius_marginal_lam0.02_s2 | 0.02 | 2 | 200/200 | complete | 26.25 | 0.3783 | 14.85 | 3.467 | 0.7925 | 0.7726 | 4.236 | 0.1613 |
| radius_marginal_lam0.05_s0 | 0.05 | 0 | 200/200 | complete | 20.14 | 0.09285 | 15.45 | 3.036 | 0.7981 | 0.783 | 2.033 | 0.2219 |
| radius_marginal_lam0.05_s1 | 0.05 | 1 | 200/200 | complete | 20.21 | 0.1032 | 15.45 | 3.042 | 0.7896 | 0.7843 | 2.037 | 0.2204 |
| radius_marginal_lam0.05_s2 | 0.05 | 2 | 200/200 | complete | 20.28 | 0.09198 | 15.45 | 3.045 | 0.796 | 0.7845 | 2.037 | 0.2208 |
| radius_marginal_lam0.1_s0 | 0.1 | 0 | 200/200 | complete | 14.65 | 0.1236 | 15.69 | 2.615 | 0.7903 | 0.7845 | 1.286 | 0.2727 |
| radius_marginal_lam0.1_s1 | 0.1 | 1 | 200/200 | complete | 14.57 | 0.1139 | 15.68 | 2.612 | 0.792 | 0.7869 | 1.287 | 0.2715 |
| radius_marginal_lam0.1_s2 | 0.1 | 2 | 200/200 | complete | 14.7 | 0.1189 | 15.69 | 2.619 | 0.7901 | 0.7817 | 1.283 | 0.2725 |
<!-- END GENERATED: sweep.runs -->

<!-- BEGIN GENERATED: sweep.figures -->
### Sweep figures

![kappa_z during training](lambda_sweep_curve_kappa_z.png)

![mean ||z|| during training](lambda_sweep_curve_z_norm_mean.png)

![std of ||z|| during training](lambda_sweep_curve_z_norm_std.png)

![online k-NN accuracy (backbone h)](lambda_sweep_curve_knn_acc_h.png)
<!-- END GENERATED: sweep.figures -->

<!-- BEGIN GENERATED: diag.summary -->
### Test-time diagnostic of the final checkpoints, per lambda (mean ± std over seeds)

`*_train`: pooled over all six views, mean over the last epoch's training steps. `*_center` / `*_aug`: embeddings of STL-10 test (8,000 images), un-augmented view and global-view augmentation. `r2_over_d_center` is E||z||^2 / d (1 for N(0, I)); `rel_spread_aug` is std(||z||) / mean(||z||).

| lambda | kappa_z_train | kappa_z_center | kappa_z_aug | r2_over_d_center | mean_r_aug | std_r_aug | rel_spread_aug | seeds | kappa_h_train | kappa_h_center | kappa_h_aug |
|---|---|---|---|---|---|---|---|---|---|---|---|
| target (N(0, I)) | 1 | 1 | 1 | 1 | 15.98 | 0.7071 | 0.0442 |  |  |  |  |
| 0.02 | 26.29 ± 0.097 | 24.8 ± 1.7 | 26.74 ± 1.5 | 0.9947 ± 0.0084 | 15.13 ± 0.1 | 3.541 ± 0.068 | 0.2341 ± 0.0052 | 3 | 0.4277 ± 0.044 | 32.62 ± 29 | 26.54 ± 22 |
| 0.05 | 20.21 ± 0.069 | 17.38 ± 0.62 | 19.61 ± 1.1 | 1.012 ± 0.0076 | 15.47 ± 0.1 | 3.047 ± 0.065 | 0.197 ± 0.0039 | 3 | 0.09602 ± 0.0063 | 0.0585 ± 0.008 | 0.07838 ± 0.017 |
| 0.1 | 14.64 ± 0.067 | 12.77 ± 0.34 | 14.4 ± 0.57 | 1.011 ± 0.011 | 15.52 ± 0.064 | 2.631 ± 0.044 | 0.1695 ± 0.0025 | 3 | 0.1188 ± 0.0048 | 0.1183 ± 0.011 | 0.1432 ± 0.01 |
<!-- END GENERATED: diag.summary -->

<!-- BEGIN GENERATED: diag.runs -->
### Test-time diagnostic, per run

| run | lambda | seed | kappa_z_train | kappa_z_center | kappa_z_aug | r2_over_d_center | mean_r_aug | std_r_aug | rel_spread_aug | kappa_h_train | kappa_h_center | kappa_h_aug |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| radius_marginal_lam0.02_s0 | 0.02 | 0 | 26.22 | 23.08 | 25.14 | 0.9924 | 15.17 | 3.462 | 0.2282 | 0.4616 | 65.96 | 51.42 |
| radius_marginal_lam0.02_s1 | 0.02 | 1 | 26.4 | 26.39 | 28.15 | 1.004 | 15.2 | 3.587 | 0.236 | 0.4432 | 16.35 | 12.77 |
| radius_marginal_lam0.02_s2 | 0.02 | 2 | 26.25 | 24.92 | 26.93 | 0.9877 | 15.01 | 3.573 | 0.238 | 0.3783 | 15.55 | 15.44 |
| radius_marginal_lam0.05_s0 | 0.05 | 0 | 20.14 | 17.41 | 19.54 | 1.021 | 15.56 | 3.026 | 0.1945 | 0.09285 | 0.05574 | 0.07386 |
| radius_marginal_lam0.05_s1 | 0.05 | 1 | 20.21 | 17.99 | 20.78 | 1.009 | 15.49 | 3.12 | 0.2014 | 0.1032 | 0.06749 | 0.09718 |
| radius_marginal_lam0.05_s2 | 0.05 | 2 | 20.28 | 16.75 | 18.52 | 1.007 | 15.36 | 2.996 | 0.1951 | 0.09198 | 0.05228 | 0.06411 |
| radius_marginal_lam0.1_s0 | 0.1 | 0 | 14.65 | 12.64 | 14.56 | 1.001 | 15.48 | 2.65 | 0.1712 | 0.1236 | 0.1084 | 0.1346 |
| radius_marginal_lam0.1_s1 | 0.1 | 1 | 14.57 | 13.15 | 14.87 | 1.023 | 15.59 | 2.662 | 0.1708 | 0.1139 | 0.1302 | 0.1544 |
| radius_marginal_lam0.1_s2 | 0.1 | 2 | 14.7 | 12.51 | 13.76 | 1.01 | 15.48 | 2.58 | 0.1667 | 0.1189 | 0.1164 | 0.1406 |
<!-- END GENERATED: diag.runs -->

## Interpretation
**Both pre-registered hypotheses are supported.**
- (a) `kappa_z` does not reach 1. At the end of training it is 26.3, 20.2 and 14.6 for lambda = 0.02, 0.05, 0.1 (table above); even the best setting is about 15x the Gaussian value, and the radius standard deviation (3.47, 3.04, 2.62) is 3.7-4.9x the chi_256 value 0.71. The mean `||z||` (14.85, 15.45, 15.69) approaches but does not reach 15.98.
- (b) Larger lambda gives smaller `kappa_z`, monotonically and with no overlap between settings: the seed-to-seed spread is at most 0.1 in `kappa_z`, against gaps of 5.6 to 6.1 between settings. The three seeds of each setting are nearly identical on every logged quantity.

**How it evolves** (figure `kappa_z during training`). For every lambda `kappa_z` drops to a minimum of about 6-9 around epochs 3-4, then **rises steadily** through training and flattens only near the end, as the cosine learning rate decays. Larger lambda slows the rise but does not stop it. The per-view SIGReg term (`loss_dist`) keeps falling throughout, so the radius spread grows while the regularizer is being reduced.

**What the numbers do and do not say**
- **The pooling explanation is ruled out; the radial spread is real in the views SIGReg controls.** At lambda = 0.1 the per-view SIGReg value is 1.285, against a noise floor of about 1.05 for exact N(0, I) samples, yet `kappa_z` is 14.6. The test-time diagnostic (table `Test-time diagnostic ... per lambda`) measures `kappa_z` on STL-10 test with the global-view augmentation: 26.7 +- 1.5, 19.6 +- 1.1 and 14.4 +- 0.57 for lambda = 0.02, 0.05, 0.1, against 26.3, 20.2 and 14.6 for the pooled training value, an agreement within about 3%. If global and local views sat at different radii, pooling them would have inflated the training value above the global-only value, so mixing is not what produces it (local views were not measured separately). The remaining explanation consistent with the data is that the sliced Epps-Pulley statistic (t in [0, 3], 1024 random slices, N = 256 per view) has little power against a radial spread of this size.
- **Scale is matched, fluctuation is not.** On the un-augmented test view `E||z||^2 / d` is 0.995, 1.012 and 1.011 (1 for N(0, I)) and the mean radius on that view is 15.6-16.0, close to 15.98: SIGReg controls the overall scale and per-coordinate variance. What it does not constrain is the fluctuation of `||z||`: its relative spread is 0.234, 0.197 and 0.170 (augmented views) against 0.044 for an exact Gaussian. A back-of-envelope reading, not measured: if `z` were a Gaussian scale mixture with a 17% relative scale spread, a random 1-D projection would have an excess kurtosis of about 0.35, comparable to the standard error of a sample kurtosis from 256 samples (about 0.31); averaging over slices does not reduce that sampling error because all slices share the same samples. Center views have a 7-11% lower `kappa_z` than augmented views.
- **Cost of more regularization.** The invariance loss rises with lambda (0.161, 0.221, 0.272), as expected from the trade-off. Online k-NN accuracy on the backbone feature shows no cost in this range: 0.793 +- 0.005, 0.795 +- 0.004, 0.791 +- 0.001 (mean +- std over seeds); on the projector output it is 0.773, 0.784, 0.784. The three settings are indistinguishable on `knn_acc_h` at this seed count.
- **Backbone feature `h` (open observation, not investigated).** `kappa_h` over training is small (0.43 at lambda = 0.02, about 0.1 at 0.05 and 0.1). On the test embeddings the two higher lambdas agree (0.06-0.15), but at **lambda = 0.02 all three seeds** have a heavy-tailed `||h||` on test images: `kappa_h` on the un-augmented view is 66.0, 16.4 and 15.6 (mean `||h||` about 4.9, standard deviation 2.5-3.7), far above the training value. The training statistic and the test statistic disagree only at the lowest lambda, and `z` is unaffected. Candidate explanations (not tested): a few test images with very large `||h||` dominating a fourth-moment statistic, or a train/eval difference such as BatchNorm statistics. Left as a note; the per-image norms can be read from the `stl10_test` dumps.
- **Extrapolation.** The curve of final `kappa_z` against lambda (26.3, 20.2, 14.6) flattens in lambda; with three points I do not extrapolate to the lambda that would give `kappa_z` = 1, if one exists.

**Run logistics** (from the job logs): all nine runs completed 200 epochs with no requeue, about 2.45 h each (median epoch 44 s, about 2,360-2,420 images/s on A100 nodes, 4 CPUs, GPU augmentation). One run (`lam0.05_s0`) ran on node `h001` instead of a `j00x` node and was about 8% slower (47.7 s per epoch); its metrics lie within the seed spread of its setting.

**Caveats.** One backbone, dataset, augmentation recipe and schedule. Training metrics are epoch means on augmented training views (all six views pooled); the diagnostic uses 8,000 STL-10 test images with the un-augmented view and the global-view augmentation. Local views were not measured separately (per-view-type logging was added in PR #25, after these runs). Seed-to-seed spread at test time (about 4-6% standard deviation in `kappa_z`) is larger than in the training logs (0.3%), as expected from estimating a fourth-moment statistic on 8,000 images. No test of the image-conditional structure of the radius yet (M3b).

## Next steps
1. **Done: test-time diagnostic** of the nine final checkpoints (above). Per-view-type (global vs. local) `kappa_z` and `||z||` logging for future runs is implemented in PR #25 and is not present in these nine runs.
2. **Radius histogram against chi_256** at convergence for each lambda, from the existing `stl10_test` embedding dumps (not produced yet).
3. **Larger lambda** (e.g. 0.25 and 0.5) is deprioritized for now: the sweep already gives the trend (`kappa_z` 26.3, 20.2, 14.6) and the diagnostic shows the spread is real; revisit if the write-up needs the lambda at which `kappa_z` approaches 1.
4. **Open observation:** the heavy-tailed `||h||` at lambda = 0.02 (above); not pursued.
5. M3b (`radius-conditional`): embed the final checkpoints on all evaluation sets and run the image-conditional analyses (ICC, redundancy R^2, difficulty correlations, downstream evaluations) on Variant A, then the go/no-go. The `stl10_test` dumps of the nine runs already exist; the remaining sets can be added with `embed --resume`.
