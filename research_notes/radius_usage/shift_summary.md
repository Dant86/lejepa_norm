# radius_usage / shift: is the radius used under distribution shift? (follow-up to E1)

**Status: pre-registration committed before any result of this study was computed. Results, interpretation and the full header are added below this section after the runs.**

h is not analyzed; LeJEPA's product is z.

## Pre-registration (written and committed before the analysis jobs were submitted)

Claim under test: the embedding radius `||z||` is a nuisance parameter. E1 (`research_notes/radius_randomization/`) found how readers on *clean* STL-10 use it; here the radius is tested where it plausibly could matter: corrupted test images and out-of-distribution images. Only z, only the un-augmented center view, runs `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}` (`last.pt`), bank = clean STL-10 `train`.

### Part 1: classification of corrupted STL-10-C test images

- Test images: STL-10-C center views (10 vendored corruptions x 5 severities of the 8000 STL-10 `test` images). The 17-corruption flag is out of scope.
- Readers: Euclidean k-NN (k = 20), logistic linear probe, one-hidden-layer MLP (weight decay chosen once on a validation split of the clean bank, as in E1). Cosine k-NN is dropped (radius-invariant by construction).
- Conditions on the corrupted test embeddings, within each (corruption, severity) cell: `baseline`, `permute` (radius permuted across the cell's images), `residual` (log-radius = clean-fitted direction-predicted part + permuted residual), `constant` (unit norm).
- Protocols: `frozen` (reader fitted on the clean unmodified bank; primary) and `retrain` (reader fitted on the bank with the same type of modification; for `constant`, unit-norm on both bank and test).
- Statistic: accuracy change of each condition against the baseline on the *same* corrupted test images, paired bootstrap over test images within each seed (1000 resamples), mean over the three seeds; reported per cell, per corruption (mean over severities, same resample so paired) and over all corruptions.
- Equivalence margin: **0.5 accuracy points** (identical to E1, `eval.equivalence_margin`, `shift.EQUIVALENCE_MARGIN`): `equivalent` iff the 90% CI lies inside (-0.5, +0.5); `different` iff not equivalent and the 95% CI excludes 0; otherwise `inconclusive`.
- Descriptive link to E6 (`research_notes/radius_transfer/`): per corruption, the center-view log-radius shift, the mean residual of the clean-fitted direction -> log-radius map, and the Spearman correlation over the 50 cells of the radius shift with the accuracy change (no p-value: cells share images).
- Reading rule: if `constant` (frozen) gains accuracy on a corruption (95% CI above 0 and not equivalent), the shifted radius hurts the frozen reader; if it loses, the shifted radius helps; `equivalent` everywhere means the radius is not used even where it shifts.

### Part 2: OOD detection

- ID: STL-10 `test` (8000); OOD: DTD `test` and SVHN `test`. Scores, all oriented higher = more OOD: (a) raw radius `||z||`; (b) radius deviation `|log r - mu| / sd` (mu, sd from the clean bank; orientation-free, unsupervised); (c) k-NN distance (k = 20 to the bank) on normalized z (direction only); (d) k-NN distance on raw z; (e) combination: bank-standardized (c) + bank-standardized (b), equal weights, no OOD data used to build it; (f) optimistic upper bound: (c) + w x standardized signed log-radius with the best w of {-4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4} chosen on the evaluation data (inside each bootstrap replicate).
- Metrics: AUROC (OOD positive) and FPR@95TPR (ID positive: threshold keeps 95% of ID images, FPR = fraction of OOD kept), 95% bootstrap CIs resampling ID and OOD images separately (paired across scores), mean over the three seeds.
- Controls analogous to E1: radius permuted across the *pooled* query set (ID test + OOD set) with bank radii permuted within the bank (5 repeats): k-NN on raw z, radius deviation, and the combination, all with permuted radii. The constant-radius control is score (c) itself. No residual-shuffle control (the clean direction -> radius map is not defined for off-support OOD images).
- **Criterion for "radius adds information" (fixed now): the lower end of the 95% paired bootstrap CI of AUROC(e) - AUROC(c) exceeds +0.01 AUROC**, for a given OOD set and lambda. `does not add (equivalent)` iff the 90% CI of that difference lies inside (-0.01, +0.01) (margin 0.01 AUROC, `ood_radius.OOD_AUROC_MARGIN`); `worse` iff the upper end of the 95% CI is below -0.01; else `inconclusive`. Justification: 0.01 AUROC is the smallest difference regarded as practically meaningful for an OOD score (one AUROC point), far above the bootstrap noise at 8000 + 1880 / 26032 images, and the same order as the 0.5-point accuracy margin of E1. The same rule is applied to (f) - (c) as an upper-bound check.
- **Criterion for "the radius alone is informative" (fixed now): the 95% CI of its AUROC lies entirely outside [0.45, 0.55]** (`ood_radius.RADIUS_NULL_BAND` = 0.05); `not informative` iff entirely inside; else `inconclusive`.
- Reading rule, stated before looking: if the radius alone is informative or the combination adds information, the radius carries OOD information beyond the direction and this is a counterexample to the nuisance claim for OOD detection, and will be reported as such.

### Code

`lejepa_norm.eval.shift` (Part 1), `lejepa_norm.eval.ood_radius` (Part 2), `lejepa_norm.eval.usage_shift` (driver), `apps/analyze.py --shift-unit` / `--shift PREFIX`. Tests: `tests/test_usage_shift.py`.
