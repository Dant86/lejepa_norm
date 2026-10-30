# radius_confidence: does the radius track the readers' confidence and errors, beyond the direction?

**Status: pre-registration committed (commits `26196e9`, amendment `b5295d2`) before any result of this study was computed; results and interpretation are below the pre-registration. Places365 (d) is pending its embeddings; SVHN is excluded by design.**

- **Branch:** `radius-confidence` (pre-registration commit: see the git history of this file)
- **Pre-registration written (UTC):** 2026-10-26T15:55Z; amended 2026-10-26T16:40Z (before any statistic was computed) to replace SVHN by DTD + Places365 as OOD sets, on the coordinator's instruction
- **Results computed on branch / commit:** `radius-confidence` @ `8bf19c2` (statistics code); the notes were generated at 2026-10-26T17:57Z (UTC).
- **Compute:** CPU only (SLURM `general`, 8 CPUs, 48 GB for the LeJEPA units, 16 GB for the baselines; nodes g004, h001, m002 and h002; about 14-17 min per LeJEPA unit, 2-3 min per baseline unit, peak RSS 24 GB). Jobs: 2253167 (first unit, `radius_marginal_lam0.05_s0`), 2253180 / 2253203 (first pass of the other eight LeJEPA runs and the three baselines), **2253285 (array 0-8, all nine LeJEPA units, final, with the post hoc models M3/M4) and 2253286 (array 0-2, the three baselines, final)**; the first pass was recomputed because M3 and M4 were added, M0-M2 agree between the passes up to the fourth decimal (e.g. AUROC(-rho) of `probe_u` at lambda 0.05: 0.6956 vs 0.6954; the linear probe is not bitwise deterministic across nodes).
- **Scope:** h is not analyzed; LeJEPA's product is z. All LeJEPA analyses use z, un-augmented center views only. **Explicit exception, scoped to the E2 baselines (SimCLR, VICReg, supervised; user decision 2026-10-26):** their analyses use the backbone feature h, because their projector output is discarded after training; they are single-seed, clean STL-10 `test` only (the baselines' STL-10-C / DTD / SVHN dumps do not exist), and LeJEPA's h is not analyzed.
- **Datasets:** STL-10 ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10); the canonical archive via the prepared arrays was used, HF linked for reference), STL-10-C (the 10 vendored corruptions x 5 severities of the STL-10 `test` split, built from the canonical archive), DTD ([canonical](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [HF `tanganke/dtd`](https://huggingface.co/datasets/tanganke/dtd)) and Places365 `val` (downsampled to 96 x 96; embedded by branch `radius-usage-places-control`, whose note gives the source and links; used here only once its dumps exist). **SVHN is deliberately excluded** from this study: it is upsampled from 32 x 32 to 96 x 96, which confounds the low-level image statistics (see `research_notes/radius_usage/svhn_control.md` on branch `radius-usage-svhn-control`).
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), pretrained weights not used), 2048-2048-256 projector. Runs `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}` (`last.pt`), 200 epochs; baselines `baseline_{simclr,vicreg,supervised}_s0` (branch `radius-baselines-train`).

## Why this study (what the literature claims, and what we have not tested)

Draganov et al., [*On the Importance of Embedding Norms in Self-Supervised Learning*](https://arxiv.org/abs/2502.09252) (arXiv:2502.09252v2), claim, as read from the paper (HTML version; to be re-checked against the PDF by a human before this is cited): (i) for cosine-similarity-based SSL losses the gradient w.r.t. an embedding is orthogonal to it and scales as 1/||z||, so norms can only grow during training and large norms slow convergence (Propositions 3.1-3.2, Corollary 3.3, Theorem 3.4); (ii) "SSL embedding norms encode a model's confidence": smaller norms for novel / out-of-distribution samples (SimCLR / SimSiam trained on CIFAR-10, norms of CIFAR-10 train vs test vs CIFAR-100, Fig. 3), and higher norms go with higher kNN accuracy and with more accurate and lower-entropy *human* labels (CIFAR-10-N, CIFAR-10-H; Fig. 4). Their evidence is qualitative per-norm-bin trends of the k-NN accuracy; they report no error-detection AUROC, no control for the direction, and no per-image paired comparison. The mechanism (i) is specific to cosine-type losses; LeJEPA's invariance term is a squared distance between z's and SIGReg acts on the Euclidean z, so (i) does not transfer automatically.

What we already have for LeJEPA (z): OOD images have smaller radius (AUROC of the raw radius below 0.5), all ten STL-10-C corruptions shift the log-radius negatively, a low-radius view is a worse classifier input than a random view (`research_notes/radius_view_signal/`, branch `radius-view-signal`), yet randomizing the radius leaves linear / MLP readers unchanged (`research_notes/radius_randomization/`) and the oracle clean radius repairs nothing. What has **never** been tested: whether the radius tracks the readers' *own* confidence and correctness per image, and whether it does so **beyond what the direction already gives** (the radius is ~95% predictable from the direction, so a correlation of radius with confidence could be entirely inherited from the direction).

Claim under test (stated so it can fail): **the radius is a sample-level reliability signal (it tracks reader errors / confidence, more under shift than on clean images) that carries no class information beyond the direction.** The "no class information" half is E1 and is not re-tested here; this study tests the first half and whether the reliability information is *additional* to the direction.

## Definitions (fixed now)

- **Bank:** STL-10 `train` center-view embeddings (5000, labeled). **Queries:** STL-10 `test` (8000, clean), the 50 STL-10-C cells (8000 each), DTD `test`, Places365 `val` (no SVHN). One center view per image; the same readers are fitted on the clean bank only.
- **Radius signals.** `rho = log ||x||`; `resid = rho - f(u)` with `u = x/||x||` and `f` the kernel ridge (exp-dot kernel, nested-CV hyperparameters) fitted on the clean bank (`redundancy.fit_predict`; the same clean-fitted map as E6 and the shift study). On clean `test`, `f(u)` is held out. On corrupted images `resid` contains the systematic shift the clean map cannot explain (E6).
- **Readers and their confidence.** `probe_z`: logistic linear probe on x (the E1 reader, sees the radius); `probe_u`: the same probe on `u` (direction only); `knn_u`: cosine k-NN, k = 20 (direction only, radius-free by construction). Confidence = maximum class probability (vote share for k-NN) `p1`; margin = `p1 - p2`. Correct = argmax equals the label. The headline readers for "beyond the direction" are `probe_u` and `knn_u`; `probe_z` is reported as the standard reader.
- **Error-detection AUROC:** errors are the positive class; a score is oriented so that a *high* score predicts an error (`-rho`, `-resid`, `-p1`, `-margin`). 0.5 = uninformative.
- **Nested comparison (beyond the direction):** per query set, 5-fold cross-fitted logistic regression of "error" on standardized features: `M0 = [logit(p1), margin]` of the reader; `M1 = M0 + rho`; `M2 = M0 + resid`. The out-of-fold predicted error probabilities are the scores. Differences `AUROC(M1) - AUROC(M0)` and `AUROC(M2) - AUROC(M0)`, and the accuracy at 50% / 80% coverage when abstaining on the lowest-scored images, are compared with a paired bootstrap over images (1000 resamples within each training seed, the same resample for all scores; mean over the three seeds).
- **Margins (fixed now):** AUROC differences 0.01 (`ood_radius.OOD_AUROC_MARGIN`; justification as in the shift study: one AUROC point, the same order as the 0.5-point accuracy margin of E1); accuracy at coverage 0.5 points. Verdicts, as in the shift study: `adds` iff the lower end of the 95% CI of the difference exceeds +margin; `equivalent` iff the 90% CI lies inside (-margin, +margin); `worse` iff the upper 95% end is below -margin; otherwise `inconclusive`. A single signal is `informative` iff the lower 95% end of its AUROC exceeds 0.51, `uninformative` iff the 90% CI lies in (0.49, 0.51), otherwise `inconclusive`.
- **Within-class control:** `rho_cc = rho - mean(rho of the bank images of the TRUE class)`; AUROC of `-rho_cc` (class-dependent radius removed) and the macro average over classes of the within-class AUROC of `-rho` (classes with both errors and correct images). The relation is "not class-dependent radius" iff `(AUROC(-rho_cc) - 0.5) >= 0.5 (AUROC(-rho) - 0.5)`.
- **Corruptions:** per cell, the mean paired change of `rho` against the clean twin (the sign of the radius shift) and, within image, the Spearman correlation of `Delta rho` with `Delta p1` of `probe_u` (corrupted minus clean twin).
- **OOD (LeJEPA only):** ID = STL-10 `test`, OOD = DTD `test` and Places365 `val` (SVHN excluded, see above; Places365 is added when its dumps exist, DTD runs first); AUROC (OOD positive) of `-rho`, `-resid`, `-p1` (probe_u), and the nested comparison `M0' = [logit p1, margin]` vs `+rho` / `+resid` for ID-vs-OOD (same margin 0.01, bootstrap resampling ID and OOD images separately).
- **Statistics for tables:** Spearman correlations are descriptive with a bootstrap 95% CI; no p-values (cells share images).

## Predictions (written before any result; confidence is my own, low unless stated)

1. **Clean `test`, LeJEPA:** `-rho` is `informative` for the errors of `probe_u` and `knn_u` at all three lambdas, with AUROC between 0.55 and 0.70 and Spearman(rho, p1) > 0. Not informative at any lambda would contradict the claim.
2. **Beyond the direction, clean `test`:** `M1 - M0` and `M2 - M0` are `equivalent` (|difference| < 0.01) for `probe_u` and `knn_u`: on clean images the radius is a function of the direction and carries no error information not already in the reader's confidence (medium confidence). `adds` would say the radius carries direction-independent reliability on clean data.
3. **Beyond the direction, corrupted cells:** the residual adds: `M2 - M0` `adds` (>= +0.01) for `probe_u` and `knn_u` when pooled over the 50 cells at >= 2 of the 3 lambdas (it is the part of the radius the clean map cannot explain; low confidence, motivated by the view-signal result that low-residual views are worse).
4. **Sign of the shift:** the mean paired `Delta rho` is negative for all 10 corruptions (severity-averaged) and non-increasing with severity for >= 8 of 10 (E6 and the shift note already show the first part at lambda 0.05; it is a replication, not a blind prediction). Across the 10 corruption types the mean shift does **not** rank the types by accuracy loss (|Spearman| < 0.6; also already visible: JPEG has the largest shift and a mild accuracy loss).
5. **Within-class:** the relation is not class-dependent radius (the criterion above holds) at every lambda.
6. **OOD:** `-rho` AUROC (OOD positive) > 0.5 for DTD, i.e. OOD has the smaller radius (known from the shift study); radius is `equivalent` (adds nothing to `p1`) for DTD, as in the shift study. No prediction for Places365 (not seen before; reported descriptively, and not read as evidence until its own low-level-statistics validity gate, pre-registered in `radius-usage-places-control`, is passed).
7. **Baselines (h, 1 seed, clean test only):** SimCLR (cosine / InfoNCE, the setting of Draganov et al.): `-rho` `informative`. VICReg and supervised: no prediction, reported descriptively. If the h radius of any baseline is `uninformative`, that is a statement about that baseline's h on STL-10, not about the paper.

## Reading rules (stated before looking)

- **Claim supported** iff prediction 1 holds (the radius tracks reader errors) **and** the residual `adds` in the shifted settings (prediction 3) or on OOD, **and** prediction 5 holds. Then "sample-level reliability signal" is established for this setup and the radius is not a pure nuisance in the sense of carrying no reliability information.
- **Radius tracks confidence only through the direction** iff prediction 1 holds but `M1 - M0` and `M2 - M0` are `equivalent` everywhere (clean, corrupted, OOD): confidence is already in the direction and the radius adds nothing, so "confidence" cannot be read off the radius in a way the direction does not already provide. This will be reported as the nuisance-compatible outcome.
- **Claim contradicted** iff `-rho` is `uninformative` on clean and on corrupted cells for both direction-only readers.
- Mixed outcomes are reported as mixed. Seed variability beyond three seeds, and the dependence between the five severities of one corruption (they share images), are not in the intervals; the baselines are one seed each. The nested logistic regression is a two-feature-plus-one linear model; a stronger model could extract more from the radius, so `equivalent` means "not with this model".

## Results

Generated tables and figures, one note per group (seed means over three seeds for LeJEPA, one seed for the baselines): [`confidence_z_lam0.02.md`](confidence_z_lam0.02.md), [`confidence_z_lam0.05.md`](confidence_z_lam0.05.md), [`confidence_z_lam0.1.md`](confidence_z_lam0.1.md) (z), [`confidence_h_simclr.md`](confidence_h_simclr.md), [`confidence_h_vicreg.md`](confidence_h_vicreg.md), [`confidence_h_supervised.md`](confidence_h_supervised.md) (h, explicit exception). Below: the scorecard of the pre-registered predictions (hand-written from them; numbers are lambda 0.05 unless stated, the other lambdas are in the per-group notes).

The generated summary tables (z and h) are at the end of this note, after the commands.

### Scorecard of the predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | `-rho` is `informative` for the errors of `probe_u` and `knn_u` at all lambdas, AUROC 0.55-0.70, Spearman(rho, p1) > 0 | **Holds**, marginally above the range: AUROC(-rho) 0.692 / 0.695 / 0.700 (`probe_u`) and 0.708 / 0.697 / 0.690 (`knn_u`) at lambda 0.02 / 0.05 / 0.1; Spearman(rho, p1) 0.38 (`probe_u`), 0.45 (`knn_u`), 0.49 (`probe_z`) at 0.05. The confidence itself detects errors much better (AUROC(-p1) 0.86). |
| 2 | On clean test, `M1 - M0` and `M2 - M0` are `equivalent` | **Holds** in 5 of 6 reader-lambda cells for M1 and 6 of 6 for M2 (clean test); the exception is lambda 0.1 `probe_u`, M1: +0.0088, `inconclusive`. The M1 gains are small but not zero (+0.003 to +0.009 AUROC, CI lower ends above 0). |
| 3 | The residual adds on corrupted cells (`M2 - M0` `adds` for `probe_u` and `knn_u` at >= 2 of 3 lambdas) | **Holds for `knn_u` (2 of 3), fails for `probe_u` (1 of 3)**: `probe_u` +0.008 (`equivalent`) / +0.017 (`adds`) / +0.011 (`inconclusive`); `knn_u` +0.019 (`adds`) / +0.017 (`adds`) / +0.009 (`equivalent`). The radius itself (M1) adds more and more consistently: `probe_u` +0.025 / +0.022 / +0.015 (`adds` at all three), `knn_u` +0.035 / +0.020 / +0.009 (`adds`, `adds`, `inconclusive`); at 50% coverage this is +1.22 accuracy points for `probe_u` at lambda 0.05 (`adds`; at 80% coverage +0.37, `equivalent`). |
| 4 | All 10 corruptions shift the log-radius negatively; non-increasing with severity for >= 8 of 10; no rank link between shift and accuracy loss across types (\|Spearman\| < 0.6) | **Sign holds (10 of 10 at every lambda); monotonicity fails (1 of 10 at lambda 0.02 and 0.05, 2 of 10 at 0.1)**: for the noises, the blurs and contrast the shift is largest at a middle severity and shrinks (gaussian_blur -0.146 at severity 3, +0.047 at severity 5), brightness is monotone and JPEG is monotone up to a flat last step (-0.2243 to -0.2239). Spearman across types of shift vs accuracy loss of `probe_z`: -0.345 / -0.382 / -0.382 (holds). |
| 5 | Not class-dependent radius (class-centered radius keeps >= 50% of the excess AUROC) | **Holds** at every lambda and reader on clean test (retained fraction 0.58-0.70, e.g. `probe_u` lambda 0.05: AUROC(-rho_cc) 0.628 vs 0.694) and on the pooled corrupted cells (0.82 to above 1.0). |
| 6 | DTD: `-rho` AUROC > 0.5 (smaller OOD radius), radius adds nothing to `p1` | **First half holds** (AUROC 0.66 / 0.66 / 0.65). **Second half holds for the radius (M1 `equivalent` in all six cells) and fails for the residual:** M2 - M0 = +0.029 / +0.059 / +0.041 (`probe_u`) and +0.021 / +0.051 / +0.027 (`knn_u`), `adds` everywhere, and the residual alone detects DTD better than the radius at lambda 0.05 and 0.1 (AUROC 0.74 and 0.69 vs 0.66 and 0.65; 0.64 at lambda 0.02, below the radius). |
| 7 | Baselines (h, clean test, one seed): SimCLR `-rho` `informative` | **Holds** for all three: AUROC(-rho) SimCLR 0.64 (`probe_u`) / 0.64 (`knn_u`), VICReg 0.72 / 0.70, supervised 0.59 / 0.58. The residual alone is `inconclusive` for VICReg and supervised (AUROC 0.50) and weak for SimCLR (0.53, `informative` but barely); M1 and M2 `equivalent` on clean test everywhere except VICReg `probe_u`, M1 +0.0136 (`inconclusive`). |

### Other findings (not pre-registered as predictions)

- **The radius stops tracking errors under heavy corruption.** Per-cell AUROC(-rho) of `probe_u` at lambda 0.05 falls from 0.67-0.72 to 0.49-0.57 at the highest severities of contrast (4-5) and of defocus and gaussian blur (3-5) (and to about 0.5-0.58 for the three noises at every severity); it stays 0.69-0.76 for brightness and JPEG and 0.63-0.80 for pixelate. The pooled corrupted AUROC(-rho) shrinks with lambda: `probe_u` 0.644 / 0.618 / 0.573.
- **Within image, a drop in radius goes with a drop in confidence:** Spearman(Delta rho, Delta p1) of a corrupted image against its clean twin is +0.19 (`probe_u`), +0.28 (`knn_u`), +0.30 (`probe_z`) pooled over the 50 cells at lambda 0.05 (per corruption -0.01 to +0.46; about 0 for contrast).
- **Post hoc models (M3, M4).** M1 and M2 compare against the reader's own confidence scalars, not against every function of the direction, so "beyond the direction" needs more: `M3 = M0 + rho_hat` (`rho_hat = f(u)`, the direction-predicted part of the log-radius, a pure function of the direction) and `M4 = M0 + rho_hat + resid` were added after seeing the results and are labeled post hoc. On the pooled corrupted cells M3 - M0 is +0.023 / +0.015 / +0.020 (`probe_u`, `adds` at all three) and +0.026 / +0.013 / +0.008 (`knn_u`, `adds`, `adds`, `equivalent`), i.e. most of what M1 adds there is already a function of the direction (the direction-predicted radius carries error information that the reader's `p1` and margin do not); M4 - M0 is larger (+0.032 at lambda 0.05, `probe_u`), so the direction-independent residual adds something on top. On DTD it is the opposite: M3 - M0 is `equivalent` or small (0 to +0.013) and M4 - M0 is +0.02 to +0.06: the OOD information in the radius is in the part the direction does **not** predict. A direction-only confidence built from `rho_hat` (what M3 stands for) was not constructed from a reader, so "is a function of the direction" does not mean "the direction-only readers already compute it".

## Interpretation

Claim under test: the radius is a sample-level reliability signal that carries no class information beyond the direction.

- **The radius does track reader errors and confidence (supported, with a ceiling).** On clean images it detects the errors of the radius-free readers with AUROC about 0.70 and correlates with their confidence at Spearman 0.38-0.49, in all nine LeJEPA runs, and with AUROC 0.58-0.72 in the three baselines' h (SimCLR 0.64). It is a much weaker error detector than the readers' own confidence (0.86), and the radius is not specific to correctness: the class-centered radius keeps most of it (prediction 5), so this is not class-dependent radius.
- **On clean images that signal is already in the direction.** The residual alone is near chance (0.52) and M1 / M2 add nothing within the 0.01 margin (prediction 2), and the Spearman of the residual with `p1` is 0.04-0.08: a radius that is 95% predictable from the direction (E1, E2) is a direction function, and on clean data it adds no error information to the readers' confidence beyond +0.003 to +0.009 AUROC. This is the nuisance-compatible reading, and it is what the data say for clean classification and for the 0.5-point accuracy margin of E1.
- **Under shift the radius adds error detection beyond the readers' confidence (supported for the radius, partly for the residual).** On STL-10-C the radius adds +0.01 to +0.035 AUROC (prediction 3: `adds` in 5 of 6 reader-lambda cells for M1, in 3 of 6 for the pre-registered residual M2), and about 1 accuracy point at 50% coverage in selective classification (`probe_u`, lambda 0.05; 0.25-0.4 points at 80% coverage, `equivalent`). For the OOD set DTD the residual is the useful part (+0.02 to +0.06 AUROC; the radius itself adds nothing). So the radius is more than a function of the direction in exactly the settings where the clean direction -> radius map fails (E6), and the residual is a novelty signal that the readers' own `p1` does not contain.
- **It is a weak and non-monotone corruption-strength signal, not a corruption-strength meter.** The shift is negative for all ten corruptions (as in Draganov et al.: smaller norms for unexpected samples), but it is not monotone in severity for nine of ten (it shrinks at the highest severities of noise, blur and contrast), the AUROC of the radius for the errors falls to chance at severities 3-5 of contrast and blur, and across corruption types the shift ranks accuracy loss only weakly (Spearman about -0.4). That also matches the view-signal result that severity is predicted by the direction (Spearman 0.91) and not by the radius (0.27-0.46). "Embedding norm corresponds to confidence" holds as a correlation (clean, corrupted, OOD), not as a calibrated confidence: the readers' own `p1` is a better error detector on every set (corrupted: 0.69 vs 0.57-0.65).
- **Relation to Draganov et al.** Their qualitative trends (smaller norm for novel / OOD samples; higher norm with higher k-NN accuracy; lower label entropy) are reproduced for LeJEPA (z) and for the baselines (h) on STL-10, in a setup (SIGReg, squared-distance invariance) where their cosine-loss mechanism does not apply by construction. What is new is the paired comparison with the readers' own confidence: the radius's extra information is small on clean images and moderate under shift, and mostly a function of the direction.
- **Net for the nuisance story.** "No class information beyond the direction" stands (E1, prediction 5). "Pure nuisance" does not: the radius carries a reliability signal that is partly (clean, and most of the corrupted gain) a direction function and partly (OOD; the residual in some corrupted cells) independent of it, and it is not exploited by the readers, whose own confidence is the better error detector. A selective-prediction or OOD use of the residual would be a small gain over the readers' own `p1` (0.01-0.06 AUROC), not a replacement for it.

### Caveats

- The nested models are 3-feature logistic regressions with a single set of out-of-fold scores; the bootstrap does not refit them (stated in the module), and `equivalent` means "not with this model". A richer error model could extract more from the radius.
- Seed variability beyond three seeds is not in the intervals; the 50 corrupted cells share their 8000 images (the bootstrap resamples images, paired across cells, 200 replicates for the corrupted groups), and the severities of one corruption are not independent. The baselines are one seed each and clean test only (no corruption or OOD dumps for them), in h, an explicit exception.
- The linear probe and k-NN confidences are different things (a calibrated softmax vs a vote share with 21 possible values); the conclusions are the same for both direction-only readers, but "confidence" here is the readers', not the network's.
- Lambda dependence is not monotone for several outcomes (e.g. the DTD residual AUROC 0.64 / 0.74 / 0.69, corrupted `knn_u` M2); no explanation is claimed.
- **Places365 (d) is not yet included**: its embeddings were being produced by branch `radius-usage-places-control` and did not exist when these units ran. Once `places_test` dumps exist for the nine runs, `analyze --confidence-unit --confidence-part ood` per run (about 2 minutes each, CPU) adds it without recomputing the rest. **SVHN is deliberately excluded** (upsampled 32 -> 96, `radius_usage/svhn_control.md`).

## Next steps

- Add Places365 once its dumps exist (command above), and read it only if the validity gate of `radius-usage-places-control` passes.
- Embed STL-10-C, DTD and Places365 for the E2 baselines (h) to repeat the corrupted and OOD parts there; and for the E3 SPHERE-JEPA runs (z has a constant radius by construction, but the raw projector output has a radius whose behaviour is itself informative: does it still track confidence?).
- Test the severity non-monotonicity directly (it may be the clean-fitted map or the embedding saturating at high corruption), and a richer error model (gradient boosting on [direction logits, rho, resid]).
- Different embedding dimension and architecture (planned) to see whether the 0.7 clean AUROC and the shift sign survive.

## Commands

Per run (CPU, `configs/base.yaml`; LeJEPA in z, baselines in h), as SLURM arrays:

```bash
sbatch --export=ALL --output=/dev/null --array=0-8%3 slurm/confidence.sbatch configs/base.yaml z \
  radius_marginal_lam0.02_s0 radius_marginal_lam0.02_s1 radius_marginal_lam0.02_s2 \
  radius_marginal_lam0.05_s0 radius_marginal_lam0.05_s1 radius_marginal_lam0.05_s2 \
  radius_marginal_lam0.1_s0 radius_marginal_lam0.1_s1 radius_marginal_lam0.1_s2     # job 2253285
sbatch --export=ALL --output=/dev/null --array=0-2%1 --mem=16G slurm/confidence.sbatch configs/base.yaml h \
  baseline_simclr_s0 baseline_vicreg_s0 baseline_supervised_s0                       # job 2253286
# which runs: python apps/analyze.py --config configs/base.yaml --run-name RUN --confidence-unit --confidence-space {z,h}
```

Aggregation (login node, writes the per-group notes and figures next to this note):

```bash
python apps/analyze.py --config configs/base.yaml --confidence radius_marginal --confidence-space z --note radius_confidence/confidence.md
python apps/analyze.py --config configs/base.yaml --confidence baseline --confidence-space h --note radius_confidence/confidence.md
```

<!-- BEGIN GENERATED: analyze.confidence_summary_z -->
### Summary (z space, direction-only readers)

M3 is a post hoc addition (only the direction-predicted part of the log-radius); M1, M2 are pre-registered.

| group | seeds | reader | clean AUROC(-rho) | clean verdict | clean M1-M0 | clean M1 verdict | clean M2-M0 | clean M2 verdict | clean M3-M0 | clean M3 verdict | corrupted AUROC(-rho) | corrupted M1-M0 | corrupted M1 verdict | corrupted M2-M0 | corrupted M2 verdict | corrupted M3-M0 | corrupted M3 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lam0.02 | 3 | probe_u | 0.6916 | informative | 0.0034 | equivalent | 0.0002 | equivalent | 0.0029 | equivalent | 0.644 | 0.0245 | adds | 0.0081 | equivalent | 0.0225 | adds |
| lam0.02 | 3 | knn_u | 0.7084 | informative | 0.0064 | equivalent | 0.0015 | equivalent | 0.0053 | equivalent | 0.6522 | 0.0351 | adds | 0.0189 | adds | 0.0259 | adds |
| lam0.05 | 3 | probe_u | 0.6954 | informative | 0.0057 | equivalent | 0.0003 | equivalent | 0.0054 | equivalent | 0.6181 | 0.0224 | adds | 0.0171 | adds | 0.0154 | adds |
| lam0.05 | 3 | knn_u | 0.6971 | informative | 0.0052 | equivalent | 0.0008 | equivalent | 0.0047 | equivalent | 0.6142 | 0.0198 | adds | 0.0169 | adds | 0.0132 | adds |
| lam0.1 | 3 | probe_u | 0.7004 | informative | 0.0088 | inconclusive | 0.0017 | equivalent | 0.0075 | equivalent | 0.5726 | 0.0149 | adds | 0.0106 | inconclusive | 0.0198 | adds |
| lam0.1 | 3 | knn_u | 0.6904 | informative | 0.0041 | equivalent | 0.0014 | equivalent | 0.0033 | equivalent | 0.5808 | 0.0094 | inconclusive | 0.0092 | equivalent | 0.0084 | equivalent |
<!-- END GENERATED: analyze.confidence_summary_z -->

<!-- BEGIN GENERATED: analyze.confidence_summary_h -->
### Summary (h space, direction-only readers)

M3 is a post hoc addition (only the direction-predicted part of the log-radius); M1, M2 are pre-registered.

| group | seeds | reader | clean AUROC(-rho) | clean verdict | clean M1-M0 | clean M1 verdict | clean M2-M0 | clean M2 verdict | clean M3-M0 | clean M3 verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| simclr | 1 | probe_u | 0.6371 | informative | 0.0067 | equivalent | 0.0031 | equivalent | 0.0055 | equivalent |
| simclr | 1 | knn_u | 0.6407 | informative | 0.0009 | equivalent | 0.0009 | equivalent | 0.0003 | equivalent |
| supervised | 1 | probe_u | 0.5871 | informative | 0.002 | equivalent | 0.0013 | equivalent | 0.0022 | equivalent |
| supervised | 1 | knn_u | 0.5811 | informative | 0.0005 | equivalent | -0.0001 | equivalent | 0.0005 | equivalent |
| vicreg | 1 | probe_u | 0.7198 | informative | 0.0136 | inconclusive | 0.0013 | equivalent | 0.0143 | inconclusive |
| vicreg | 1 | knn_u | 0.6983 | informative | 0 | equivalent | -0 | equivalent | -0 | equivalent |
<!-- END GENERATED: analyze.confidence_summary_h -->
