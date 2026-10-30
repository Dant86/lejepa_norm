# radius_usage / svhn_control: is the radius signal on SVHN just low-level image statistics?

- **Branch / version:** radius-usage-svhn-control @ a1e6f73 (analysis code; this note is committed right after)
- **Run datetime (UTC):** 2026-10-26T01:47:00Z
- **Command(s):** see the header of `svhn_control_lam0.05.md` (jobs 2247653 stats, 2247794-2247802 per-run units, 2247980 aggregation; CPU only, no GPU).
- **Datasets:** STL-10 ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10)), DTD `test` ([canonical](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [HF `tanganke/dtd`](https://huggingface.co/datasets/tanganke/dtd)), SVHN `test` ([canonical](http://ufldl.stanford.edu/housenumbers/), [HF `ufldl-stanford/svhn`](https://huggingface.co/datasets/ufldl-stanford/svhn)); canonical sources were used, HF pages are for reference only.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` of `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}`.

**Status: pre-registration (committed in db1123a before any result was computed) followed by the results and interpretation at the end of this file. Generated per-lambda tables: `svhn_control_lam{0.02,0.05,0.1}.md`.**

h is not analyzed; LeJEPA's product is z.

## Question

The shift note (`shift_summary.md`) found that on SVHN the raw radius alone separates OOD from ID (AUROC 0.16 to 0.24 with "high = OOD", i.e. 0.76 to 0.84 when oriented: OOD images have the *smaller* radius) and that adding the radius deviation to the direction score raises SVHN AUROC (+0.12 / +0.06 / +0.04 for lambda 0.02 / 0.05 / 0.1) but not DTD. Open caveat from that note: the SVHN effect may reflect low-level image statistics (small, low-texture digit crops give a smaller radius) rather than semantic novelty. This study tests that, z only, center views only, no training and no new embeddings.

## Pre-registration (fixed before looking at any result)

Runs `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}` (`last.pt`), center-view z dumps of `stl10_train` (bank), `stl10_test` (ID), `dtd_test`, `svhn_test`; seeds averaged, 95% percentile intervals of 1000 image-bootstrap replicates (ID and OOD resampled separately, paired across scores); AUROC margin 0.01 (as in the shift note); R^2 margins below. The verdict rule for a difference `A - B` is the TOST analogue of the shift note: `A higher` iff the lower 95% bound exceeds the margin, `equivalent` iff the 90% interval is inside (-margin, +margin), `B higher` iff the upper 95% bound is below -margin, else `inconclusive`.

**Low-level statistics** (`lejepa_norm.eval.image_stats`, 15 per image, computed from the raw prepared 96x96 uint8 images as fed to the model, on the [0, 1] image; the model's per-channel normalization is an affine map per channel and does not change any analysis below): mean luminance, contrast (luminance std), per-channel mean and std (6), mean HSV saturation, finite-difference gradient (edge) energy, high-frequency fraction of the non-DC power spectrum (radial frequency above 0.25 cycles/pixel), Laplacian variance (blur), colorfulness (Hasler and Suesstrunk), JPEG (quality 75) and PNG (compress level 6) byte sizes as entropy proxies. The five positive heavy-tailed ones (gradient energy, high-frequency fraction, Laplacian variance, JPEG and PNG bytes) are log-transformed; all are standardized with the bank (STL-10 train). Every model of the statistics is fitted on the bank only (ID only); no OOD label or OOD image is used to fit anything.

**(a) How much of the ID log-radius do the statistics explain?** Fit on the bank, held-out R^2 on STL-10 test (hyperparameters by cross-validation inside the bank): ridge on the statistics; gradient boosting (`HistGradientBoostingRegressor`, 300 iterations, early stopping) on the statistics; the direction -> log-radius kernel ridge of E1/E6 (`randomization.fit_radius_model`); and direction prediction plus a statistics model fitted on the bank's out-of-fold direction residual (increment of the statistics beyond the direction). The statistics *explain log-radius materially* iff the lower 95% bound of their R^2 exceeds 0.05; comparison with the direction and the increment use margin 0.01 R^2.

**(b) OOD AUROC of the statistics alone (unsupervised).** Scores, higher = more OOD: Mahalanobis distance of the statistics to the bank (Ledoit-Wolf covariance); the statistics-predicted log-radius (gradient boosting fitted on the bank), low = OOD; per statistic the two-sided deviation `|z|` and the one-sided score oriented by the sign of the statistic's bank correlation with log-radius (single-statistic rows are descriptive). Compared with the radius alone (raw `||z||` with high = OOD, the orientation-free deviation `|log r - mu| / sd`, and `-log r`, whose orientation was read off the shift note's AUROC below 0.5) and with the direction-only k-NN (all from the shift study's scores, reused unchanged). Primary comparison: statistics Mahalanobis vs radius deviation (both orientation-free).

**(c) Does the radius still add to the direction score after removing the statistics?** The log-radius is regressed on the statistics (fit on the bank; primary model gradient boosting, secondary ridge; bank residuals out of fold); the residual deviation `|res - mu| / sd` (bank mu, sd) replaces the radius deviation in the shift study's equal-weight combination. Primary criterion: the lower 95% bound of `AUROC(direction + residual deviation) - AUROC(direction)` exceeds 0.01. The signed residual with weight -1 is exploratory, like its counterpart in the shift note.

**(d) Does direction + statistics (no radius) match direction + radius?** Equal-weight sum of the standardized direction score and the standardized statistics Mahalanobis distance. Primary comparison: `AUROC(direction + radius deviation) - AUROC(direction + statistics)`: `equivalent` or `B higher` means the statistics replace the radius; `A higher` means the radius carries information the statistics do not. Also: the radius deviation added on top of direction + statistics.

**Matched control.** Matching space: bank-standardized mean luminance, contrast and log Laplacian variance (blur). Each SVHN image is matched to its nearest STL-10 test image (with replacement), giving a statistics-matched ID reference. `closed` = `1 - mean(rho_svhn - rho_matched) / (mean(rho_svhn) - mean(rho_id))` (rho = log-radius) is the fraction of the SVHN-vs-STL-10 log-radius gap that matching removes (bootstrap over SVHN images, ID images treated as fixed). Also the AUROC of `-log r` for SVHN against (i) all STL-10 test, (ii) the matched reference with match multiplicities as weights, (iii) the 10% of STL-10 test images closest to SVHN in the matching space. The radius is *still informative on matched images* iff the 95% interval of AUROC (ii) lies outside [0.45, 0.55]; *not informative* iff entirely inside. The balance of the matching statistics and the match multiplicities are reported.

**Overall reading for SVHN, per lambda (fixed in advance).** `explained by low-level statistics`: the residual combination of (c) does not add (verdict not `A higher`), (d) is `equivalent` or `B higher`, and the matched AUROC (ii) is not informative. `independent of low-level statistics`: (c) adds, (d) is `A higher` and (ii) is informative. Anything else: `partly explained`. The fraction of the shift combination's AUROC gain that survives the removal of the statistics is reported alongside. DTD is reported but not decisive (the radius did not add there).

## Code

`lejepa_norm.eval.image_stats`, `lejepa_norm.eval.svhn_control` (reuses `ood_radius` scores/ROC/bootstrap, `randomization` bootstrap helpers and direction -> radius map), `plotting.bar_plot`, `apps/analyze.py --svhn-control PREFIX [--svhn-control-stats | --svhn-control-run RUN]`. Tests: `tests/test_svhn_control.py`.

## Results (summary; numbers from the generated per-lambda notes, seed means, 95% CIs there)

| lambda | R^2 of log-radius from statistics (GB) | AUROC SVHN: statistics Mahalanobis | AUROC SVHN: radius deviation | (c) direction + residual dev. - direction | (d) radius comb. - statistics comb. | matched AUROC (low radius, matched ref.) | pre-registered reading (SVHN) |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.269 | 1.000 | 0.719 | +0.258 (A higher) | -0.332 (B higher) | 0.970 (informative) | partly explained |
| 0.05 | 0.217 | 1.000 | 0.646 | +0.187 (A higher) | -0.363 (B higher) | 0.874 (informative) | partly explained |
| 0.1 | 0.209 | 1.000 | 0.620 | +0.184 (A higher) | -0.364 (B higher) | 0.906 (informative) | partly explained |

DTD (not decisive): the radius deviation does not add to the direction score (shift combination - direction is -0.09 to -0.12), statistics Mahalanobis beats the radius deviation by about +0.30 AUROC, and the overall reading is again `partly explained`.

## Interpretation

1. **The low-level statistics explain only a minority of the ID log-radius** (held-out R^2 0.21 to 0.27, lower bound far above 0.05) against 0.95 for the direction; they add 0.001 to 0.002 R^2 beyond the direction. So the radius is mostly a function of the direction, not of these statistics.
2. **The pre-registered rule gives `partly explained` for SVHN at every lambda** (residual combination still adds, matched AUROC informative, but the radius combination does not match the statistics combination).
3. **The control itself is degenerate on SVHN and its verdicts should not be read as a clean answer.** SVHN is built at 96x96 by upsampling 32x32 digits (`data/layout.py`), so Laplacian variance, gradient energy, high-frequency fraction and JPEG/PNG sizes separate it from STL-10 almost perfectly (single-statistic AUROC 0.995 to 1.000; the statistics Mahalanobis score has AUROC 1.000). Consequences: (i) (d) is `B higher` because the statistics comparator is saturated at 1.0, and the radius deviation on top of it is `equivalent` only because of that ceiling; (ii) the matched control fails its premise: matching balance on the blur statistic is off by about 3 SD, one STL-10 image takes up to 23,901 SVHN matches, and the matching closes only 11% (lam 0.05; -15% at 0.02) of the log-radius gap, so "matched AUROC informative" is not evidence of independence from the statistics; (iii) the "residual given statistics" scores are evaluated where the statistics model extrapolates far off the bank's support, so their AUROC (0.82 to 0.89) and the 'fraction of gain retained' (2.1 to 4.6, above 1) reflect extrapolation error, not a clean radius signal.
4. What can be said: SVHN's smaller radius is accompanied by a trivially detectable low-level difference (upsampling blur), so the shift note's SVHN gain **cannot be attributed to semantic novelty**, and this control **cannot rule the low-level explanation out either**. The earlier framing (radius carries OOD signal on SVHN) should carry the caveat that SVHN differs from STL-10 in resolution provenance. DTD, where the statistics do not trivially separate the sets (Mahalanobis AUROC 0.84, below the direction-only kNN 0.90), shows no radius gain to begin with.

## Next steps

- A clean SVHN control needs an OOD set whose low-level statistics overlap STL-10's (e.g. SVHN downsampled to 32x32 and back, blurred STL-10 as the ID reference, or another natural-image OOD set at native resolution) and a re-embedding; not done here. Alternatively, restrict to SVHN images inside the bank's statistics support and re-run (b)-(d).
- Deviation from pre-registration: none in the computations; the degeneracy was found after the results and is reported as such rather than altering the rule.
