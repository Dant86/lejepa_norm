# radius_usage / places_control: is the OOD radius signal just low-level image statistics? (Places365 instead of upsampled SVHN)

- **Branch / version:** radius-usage-places-control (pre-registration committed before any Places365 embedding, statistic or result was computed; the final header with commit and commands is in the results section below)
- **Pre-registration datetime (UTC):** 2026-10-26T02:10:53Z
- **Datasets:** STL-10 ([canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10)), DTD `test` ([canonical](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [HF `tanganke/dtd`](https://huggingface.co/datasets/tanganke/dtd)), Places365 validation, 36,500 images, small version with 256x256 JPEGs ([canonical](http://places2.csail.mit.edu/), downloaded with `torchvision.datasets.Places365(split="val", small=True)`; closest HF page [`rumenguin/places365-val`](https://huggingface.co/datasets/rumenguin/places365-val), not used). Canonical source used.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used); `last.pt` of `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}`.

h is not analyzed; LeJEPA's product is z.

## Question

`svhn_control.md` asked whether the OOD radius signal (OOD images have the smaller radius; adding the radius deviation to the direction score helped on SVHN) is explained by low-level image statistics. It could not answer: SVHN is built by upsampling 32x32 digits to 96x96, so blur and compression statistics separate it from STL-10 almost perfectly (statistics Mahalanobis AUROC 1.000), the matched control was badly unbalanced (one STL-10 image took up to 23,901 matches) and the "residual given statistics" scores extrapolated off the bank's support. Here the same analysis runs with an OOD set that is only ever *downsampled* to 96x96: Places365 validation scenes (native 256x256 or larger, bicubic downsampling by 2.67x). STL-10 itself is ImageNet photos downsampled to 96x96, so low-level statistics should overlap. Places365 scenes are far-OOD semantically (no STL-10 class is a scene category, though scenes contain objects).

**Resize.** The repo's builders (`data.prepare._resize`) use PIL bicubic for DTD and SVHN, and so does Places365 here (`convert("RGB").resize((96, 96), BICUBIC)`, no aspect-ratio crop, squash as in the 256x256 small version). STL-10 is read from the official binaries; its authors do not publish the downsampling filter, so "the filter that built STL-10" is unknown and cannot be matched. Whether the statistics of the downsampled Places365 images are comparable to STL-10 is exactly what the validity check below decides.

## Pre-registration (fixed before any Places365 embedding, statistic or result was computed)

Everything in `svhn_control.md` carries over unchanged (same statistics, models, scores, comparisons, bootstrap, margins, verdict rules, overall reading), with `svhn_test` replaced by `places_test` as the decisive OOD set; DTD is again reported but not decisive. Runs `radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}` (`last.pt`), center-view z, seeds averaged, 95% percentile intervals of 1000 image-bootstrap replicates, AUROC margin 0.01, R^2 margins 0.05 and 0.01, matched-AUROC null band [0.45, 0.55]. The Places365 score columns are computed with the shift study's `ood_radius.compute_scores` exactly as for DTD/SVHN (same bank, k and seed).

**Validity of the control (new, decisive).** The control is *valid* only if all of the following hold; if any fails, the note says so plainly and **the radius result is not interpreted** (the overall reading is reported as `invalid control`, the tables are still generated).

- V1 *Statistics do not saturate.* The seed-mean AUROC of the statistics Mahalanobis score (Places365 vs STL-10 test) is **at most 0.90** (point estimate; the 95% interval is reported). SVHN gave 1.000.
- V2 *The matched reference is balanced.* After matching, the absolute standardized mean difference (matched reference minus OOD set, divided by the OOD standard deviation, bank-standardized) is **below 0.25 for each** of mean luminance, contrast and log Laplacian variance (the blur statistic).
- V3 *No control image dominates.* Matching is **capped at 5 matches per STL-10 test image** (by construction the maximum multiplicity is at most 5; 36,500 images over 10,000 controls needs at least 4). Matches are assigned greedily in order of increasing distance to the nearest control, each OOD image taking its nearest control with spare capacity. The maximum multiplicity and the number of unique matches are reported.

V1 to V3 are properties of the data and the statistics, identical for the three lambdas.

**Modified matched control.** As in `svhn_control.md`, except that matching uses the cap above instead of unlimited replacement. Criteria for "still informative on matched images" are unchanged. The matching on DTD (1,880 images) is capped identically.

**Overall reading for Places365, per lambda (fixed in advance).** As for SVHN (`explained by low-level statistics`, `independent of low-level statistics`, otherwise `partly explained`), but reported only if V1 to V3 hold; otherwise `invalid control`.

## Code

`lejepa_norm.data.prepare` (Places365 stage), `lejepa_norm.eval.dumps` / `embedding` (`places_test` set), `lejepa_norm.eval.svhn_control` (generalized with a `Study`: SVHN or Places365, capped matching, validity rows), `apps/analyze.py --svhn-control PREFIX --control-study places`. Tests: `tests/test_svhn_control.py`, `tests/test_prepare_places.py`.
