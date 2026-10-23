# E1 summary: is the radius used by downstream readers? (lambda 0.02 / 0.05 / 0.1)
- **Branch / version:** `radius-randomization` @ `0f0cd4459d52e558887a1b2a3b21ededaa76905d` (code, merged as PR #32); this summary and the per-lambda notes on `radius-randomization-notes`
- **Run datetime (UTC):** 2026-10-23T15:21:33Z to 2026-10-23T16:19:16Z (ends of SLURM jobs 2236228, 2236229, 2236227; the cluster reports CDT, converted)
- **Command(s):** the three analysis runs, exactly as in the headers of `lam0.02.md`, `lam0.05.md`, `lam0.1.md` (one `apps/analyze.py --config configs/base.yaml --randomize-radius radius_marginal_lam<LAMBDA> --note radius_randomization/lam<LAMBDA>.md` per lambda, via `sbatch --wrap`). No code was run for this summary; every number below is copied from the generated tables of those notes.
- **Datasets:** STL-10 `train` / `test`, clean, center views ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10)); the canonical archive was used, HF is reference only.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); 9 checkpoints, lambda {0.02, 0.05, 0.1} x seeds {0, 1, 2}.
- **Scope:** h is not analyzed; LeJEPA's product is z.
- **Equivalence margin:** 0.5 accuracy points, fixed before looking at results.

## Question

Spec leg 2 (usage): does randomizing the radius of z hurt downstream readers? The claim being tested is that ||z|| is a nuisance parameter, i.e. carries no label-relevant information beyond the direction.

## Setup

See the per-lambda notes. Conditions: permute radius across images, residual-only shuffle (direction-predicted log-radius kept), constant radius. Readers: Euclidean k-NN, cosine k-NN (reference), linear probe, 1-hidden-layer MLP. Protocols: retrained on shuffled train features, and frozen (clean-trained, shuffled test features). Paired bootstrap over test images, averaged over 3 seeds; TOST-style verdict with margin 0.5.

## Results

Accuracy change versus baseline in points (all values from the generated tables; 90% CI in brackets; `E` = equivalent, `D` = different). Baseline accuracies: k-NN euclidean 77.7 / 78.7 / 78.9, linear 80.4 / 81.7 / 81.2, MLP 80.4 / 81.6 / 81.8 for lambda 0.02 / 0.05 / 0.1 (the baseline is the same for the retrained and frozen protocols).

**Retrained on shuffled train features**

| reader | condition | lambda 0.02 | lambda 0.05 | lambda 0.1 |
|---|---|---|---|---|
| k-NN euclidean | permute | -0.74 [-0.96, -0.53] D | -0.66 [-0.88, -0.43] D | -0.54 [-0.76, -0.32] D |
| k-NN euclidean | residual shuffle | -0.10 [-0.25, +0.04] E | -0.09 [-0.24, +0.05] E | -0.07 [-0.23, +0.07] E |
| k-NN euclidean | constant radius | -0.38 [-0.60, -0.15] D | -0.35 [-0.59, -0.10] D | -0.42 [-0.67, -0.16] D |
| linear probe | permute | -0.03 [-0.16, +0.09] E | -0.16 [-0.29, -0.03] E | -0.19 [-0.30, -0.08] E |
| linear probe | residual shuffle | -0.01 [-0.08, +0.07] E | +0.00 [-0.07, +0.08] E | -0.07 [-0.14, +0.01] E |
| linear probe | constant radius | +0.05 [-0.08, +0.19] E | +0.00 [-0.13, +0.14] E | -0.17 [-0.28, -0.06] E |
| MLP | permute | -0.23 [-0.36, -0.08] E | -0.10 [-0.25, +0.04] E | -0.02 [-0.15, +0.09] E |
| MLP | residual shuffle | -0.06 [-0.18, +0.05] E | -0.01 [-0.12, +0.10] E | -0.00 [-0.09, +0.09] E |
| MLP | constant radius | -0.08 [-0.25, +0.12] E | -0.11 [-0.28, +0.05] E | -0.03 [-0.19, +0.10] E |

**Frozen (trained clean, applied to shuffled test features)**

| reader | condition | lambda 0.02 | lambda 0.05 | lambda 0.1 |
|---|---|---|---|---|
| k-NN euclidean | permute | -0.17 [-0.33, -0.00] E | -0.07 [-0.23, +0.07] E | +0.10 [-0.03, +0.23] E |
| k-NN euclidean | residual shuffle | +0.02 [-0.08, +0.13] E | +0.10 [+0.01, +0.20] E | +0.01 [-0.07, +0.10] E |
| linear probe | permute | **-0.44 [-0.59, -0.30] D** | -0.34 [-0.46, -0.21] E | -0.25 [-0.36, -0.15] E |
| linear probe | residual shuffle | -0.02 [-0.09, +0.05] E | -0.05 [-0.11, +0.03] E | -0.08 [-0.13, -0.02] E |
| MLP | permute | -0.07 [-0.15, +0.01] E | -0.08 [-0.16, +0.00] E | -0.05 [-0.11, +0.01] E |
| MLP | residual shuffle | -0.07 [-0.11, -0.03] E | +0.00 [-0.04, +0.05] E | +0.02 [-0.01, +0.05] E |

**Frozen readers, constant radius (distribution-shift artifact)**

| reader | lambda 0.02 | lambda 0.05 | lambda 0.1 |
|---|---|---|---|
| k-NN euclidean accuracy (baseline about 78%) | 13.0% | 14.3% | 13.5% |
| linear probe accuracy (baseline about 81%) | 12.5% | 12.8% | 15.7% |
| MLP accuracy (baseline about 81%) | 28.4% | 40.3% | 48.2% |

Cosine k-NN: exactly 0.00 in all 18 cells per lambda (generated tables).

## Interpretation

What the data support:

1. **Linear probe and MLP do not need the radius, up to the margin.** Retrained on shuffled features, all 18 linear/MLP verdicts (3 conditions x 2 readers x 3 lambdas) are `equivalent`, largest point estimate -0.23. Residual-only shuffle, which keeps the direction-predictable part of the radius, is `equivalent` for every reader (including k-NN) with |difference| <= 0.10. Frozen readers under *permutation* are `equivalent` in 8 of 9 cells (linear, MLP, k-NN at three lambdas); the exception is the **frozen linear probe at lambda 0.02 (-0.44, `different`)**. So "equivalent under permute-radius at all lambdas" is not strictly true for the frozen linear probe at lambda 0.02. A caveat that deserves to be stated: the frozen linear drop is -0.44 / -0.34 / -0.25 for lambda 0.02 / 0.05 / 0.1, monotone in lambda like `kappa_z` (26.3 / 20.2 / 14.6) and always negative; this is a small, consistent dependence of the linear readout on the radius, not noise.
2. **Euclidean k-NN depends on the radius a little, by construction.** Retrained on shuffled train features it loses 0.54-0.74 points under permutation and 0.35-0.42 under constant radius (verdict `different`). The constant-radius case is exactly cosine k-NN, so that figure is the Euclidean-minus-cosine gap at baseline (Euclidean is slightly better, 78.7 vs 78.4 at lambda 0.05), i.e. a metric difference rather than a loss of label information. The permutation loss vanishes when the direction-predicted radius is preserved (residual shuffle, -0.07 to -0.10): what the metric uses is radius information that the direction already predicts (held-out R^2 0.93-0.95).
3. **Cosine k-NN is a sanity check, not evidence.** It is radius-invariant by construction, so its zeros only confirm the pipeline.
4. **Frozen readers under constant radius collapse (13-16% k-NN/linear, 28-48% MLP).** Unit-norm features are rescaled by about 1/16 relative to what the frozen readers saw (E||z||^2/d is about 1). This is distribution shift; the same features with retrained readers lose nothing (linear +0.05 to -0.17, MLP -0.03 to -0.11). It must not be read as label information in the radius.

Against the spec's claim and the margin: the data are consistent with "the radius carries no label information beyond what the direction predicts", with the radius effect on linear/MLP readers bounded to roughly 0.5 points. They do not show an exactly zero effect, and the frozen linear dependence at small lambda is at or just beyond the margin.

**Soundness of the `equivalent` verdicts near the margin.** The TOST rule itself is applied correctly: a 90% CI inside (-0.5, +0.5) is equivalence at the 5% level. For lambda 0.05, frozen linear, permute: -0.34 [-0.46, -0.21] is inside the margin, so the verdict follows from the stated rule. But the interval is the *image-resampling* interval only; training-seed variability is not in it. The per-run differences from the generated per-run table are -0.34 (seed 0), -0.21 (seed 1), -0.48 (seed 2) (0.8131 vs 0.8165, 0.8149 vs 0.8170, 0.8126 vs 0.8174). A t-interval over the three seeds (sd 0.14, standard error 0.08, t = 2.92 at 2 degrees of freedom) would be about [-0.58, -0.10] at 90%, which crosses the margin. I computed this by hand from the table values, not with the analysis app; it is an illustration, not a result. Verdicts with CI endpoints within about 0.1 of the margin (frozen linear permute at lambda 0.05 and 0.1 are the ones that matter) should therefore be called "equivalent under the image-bootstrap test, seed uncertainty not covered".

**Weaknesses of the test design.**
- Three training seeds per lambda; the bootstrap resamples test images only (8000 test images, paired), so the CI ignores seed-to-seed and shuffle-to-shuffle variation (5 repeats per randomized condition, shuffle sd about 0.1-0.25 points per repeat, averaged but not propagated).
- The weight decay of the MLP is selected once per run on a clean validation split and reused for all conditions; the MLP/linear epoch budget is fixed; one k (20) for k-NN. A reader tuned on the shuffled features could do better or worse.
- Residual shuffle: the held-out prediction of the radius still contains estimation error, which over-states the unpredictable part of the radius; and the shuffled log-radius has only approximately the original marginal. It is a conservative probe of the unpredictable part, but not an exact one.
- The margin of 0.5 points is the same order as the effects of interest (k-NN -0.35 to -0.74, frozen linear -0.25 to -0.44); the verdict classes therefore depend on the margin. Equivalence is also relative to STL-10 classification at 8000 test images; other tasks (OOD, corruptions) were not tested.
- Only the class-label task is used; "no label-relevant information" is not shown for other kinds of information (e.g. difficulty, OOD-ness), which earlier notes found in the radius.
- Figure bug (cosmetic): in each per-lambda note the generated Euclidean and cosine k-NN figures share one file name, so only the cosine figure survives, and with a single lambda each plot is one point per series. The tables are authoritative.

## Next steps

- Propagate seed variability into the verdicts (a hierarchical bootstrap over seeds and images, or more seeds), at least for the frozen linear probe, before calling the lambda 0.02 / 0.05 cases settled in either direction.
- Fix the figure file naming and plot the three lambdas in one figure (Plotly) in the analysis code, on its own feature branch.
- E3 (`sigreg-sphere`, training with a constant radius) is the necessity test; E1 only shows the trained readers' independence from the radius.
