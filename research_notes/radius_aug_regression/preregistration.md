# radius_aug_regression: pre-registration
- **Branch / version:** `radius-aug-regression` (this file is committed before any analysis code is run; its git history is the proof of the order)
- **Written (UTC):** 2026-10-30T16:20:00Z
- **Command(s):** none yet; this note defines the study. Commands that produce results are recorded in `aug_regression.md`.
- **Datasets:** STL-10 ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10); canonical archive used) `test` (8000 images), clean only, K = 8 random global views per image plus the center view; `train` as the labeled bank for the probe in part C.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` of `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}` (Variant A).
- **Scope:** z is the main outcome (`log ||z||`); `log ||h||` is a secondary check. LeJEPA's product is z.

## Question

Instance retrieval, view-signal and the confidence study showed that the radius is a stable per-image signature (ICC of log-radius about 0.87) and that, within an image, the highest-radius view is a better view than the lowest (+2.6 to +6.0 accuracy points; the pre-registered prediction was the opposite). None of those studies could say *why* a view has a low radius, because the dumps did not record the augmentation of each view. This study records it and asks: **how much of the within-image variation of the radius is explained by the augmentation parameters of the view, and does the radius carry information about the view's quality beyond those parameters?** Working hypothesis (not a result): the radius reflects how much image evidence the encoder sees (large, sharp, well-exposed crops give a large radius). The image-level 87% of the radius variance is not addressed here.

## Setup

**Data.** `apps/embed.py` is extended to save, for every augmented view, the parameters drawn by `data.gpu_views.sample_params` (crop box, flip, color-jitter gate and factors, grayscale, blur gate and sigma) next to the embeddings; the nine final checkpoints are re-embedded on `stl10_test` (`eval.num_views=8`). Re-embedding is cheap (8000 images x 9 views per checkpoint); the views are new draws, not those of the existing dumps.

**Unit and outcome.** One row per (image `i`, augmented view `v`), 64,000 rows per checkpoint. `y_iv = log ||z_iv||`. Every analysis is run per checkpoint (nine of them); results are pooled over seeds only for a lambda where the signs of the main effects agree across the three seeds, and never across lambdas without per-lambda numbers.

**Covariates `x_iv`.** Crop: log area fraction (`height * width / (H * W)` of the drawn box), log aspect ratio, offset of the crop center from the image center (normalized, Euclidean). Flip (binary). Color jitter: gate (binary), brightness, contrast, saturation factors and `|hue|` (set to their neutral values when the gate is off). Grayscale (binary). Blur: gate (binary) and sigma (0 when off). All continuous covariates are standardized over the pooled rows of the checkpoint.

**Primary model (image fixed effects).** `y_iv = alpha_i + beta' x_iv + eps_iv`, fitted by within-image demeaning. `alpha_i` absorbs everything fixed about the image (content, class, low-level statistics), so `beta` is identified from variation across views of one image only. Log area enters through a natural cubic spline (4 degrees of freedom) and the model includes the crop-area x blur-gate interaction (small crops are upsampled and look blurry regardless of the blur gate). Uncertainty: percentile intervals from a bootstrap that resamples images (all 8 views together), 1000 replicates, 90% and 95%.

**Reported quantities.**
1. Within-image `R^2`: share of the within-image variance of `y` explained by `x`. Alongside, the ICC (share of total variance that is within-image) so the explained share of the *total* variance is also stated.
2. Coefficients per parameter (the spline as an effect curve of radius against crop area). Coefficients are conditional on the other parameters; they are not the total effect of an augmentation type.
3. Block partial `R^2` per augmentation type (crop, color jitter, grayscale, blur, flip): the drop in within-image `R^2` when the whole block (and its interactions) is removed. This is the only quantity reported per augmentation type.
4. Ceiling: gradient-boosted trees on the same `x` with image-grouped 5-fold cross-validation, within-image `R^2` on held-out images, against the same quantity for the (cross-validated) linear spline model.
5. Image-specific crop slopes: random slope on log area (mixed model, image random intercept and slope); the correlation between an image's slope and its intercept `alpha_i`.

## Part C: does the radius carry view information beyond the augmentation?

For each row, `yhat_iv = beta' x_iv` (cross-fitted: coefficients estimated on the other image folds, 5 folds grouped by image) and `r_iv = (y_iv - alpha_i) - yhat_iv` (the within-image radius residual). Two view-level outcomes: (C-a) whether a frozen linear probe, fitted on the clean `train` center-view z, classifies the view correctly; (C-b) the angle between the view's direction and the image's mean direction over its eight augmented views (a smaller angle is a more typical view). Models with image fixed effects (linear probability for C-a; linear for C-b) on `yhat` alone, `r` alone and both; and the incremental within-image contribution of `r` (and of `yhat`) to the other. For C-a the within-image AUC of each score (pairs of views of the same image with different correctness) is also reported. Intervals: image bootstrap as above. Margins: 0.01 within-image AUC; a standardized-coefficient change of the `yhat` effect is judged by its interval.

## Predictions (written before looking)

- **P1.** Crop area has a positive, saturating effect on the radius (the radius rises with area and flattens for large crops); blur sigma and the blur gate have a negative effect; color jitter parameters have small effects.
- **P2.** `x` explains at least 30% of the within-image variance of `y` (cross-validated, linear spline model); the trees do not exceed the linear model by more than 0.10 in `R^2`. If `x` explains less than 10%, the hypothesis that the radius reflects the amount of image evidence the view carries is rejected for this checkpoint and the view-level radius variation is mostly something that these augmentations do not specify.
- **P3.** Controlling for `yhat` removes at least half of the within-image radius effect on view correctness (C-a): the within-image AUC of `r` is below half-way from 0.5 to the AUC of `yhat + r`. If `r` retains most of it (the AUC of `r` is above half-way), the radius carries view information that the augmentation parameters do not.
- **P4.** The signs of the crop-area, blur-gate and sigma effects agree across the three seeds of every lambda. Where they do not, per-seed results are reported and not pooled.
- **P5.** The angle to the image's mean direction (C-b) is predicted by `yhat` (a view that differs strongly from the typical view is a small, blurry one).

Reading rules: P2 holds if the cross-validated within-image `R^2` of the linear model is at least 0.30 for at least 7 of 9 checkpoints. The "radius = amount of image evidence" hypothesis is supported if P1, P2 and P3 hold; it is partly supported if P1 and P2 hold and P3 fails; it is rejected if P2 fails.

## Limits stated in advance

Augmentation parameters are correlated (small crops are blurrier after resizing; the interaction term and the tree ceiling are the only guards), so block partial `R^2` shares overlap and do not sum to the total. The sign of the crop effect is a prediction, not a finding. Only clean `stl10_test` is analyzed here; corruption-severity slopes and one-augmentation-at-a-time embeddings are separate studies. Three seeds per lambda do not cover seed variability beyond that. The 87% image-level part of the radius is untouched.

## Reporting

All tables are generated by `apps/analyze.py` (a new `--aug-regression` mode), per checkpoint and per lambda, in `aug_regression.md` in this folder; figures (effect curves, block `R^2`, AUC bars) are PNGs next to it. Deviations from this file are listed in the results note.
