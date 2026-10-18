# GPU augmentation vs the PIL reference pipeline
- **Branch / version:** `feat/gpu-augmentation` (branched from `origin/master` @
  `6506e11`); the script, the pipeline and this note are added in the branch's
  commit. The `HEAD when run` row below is the commit checked out while the script
  ran (the uncommitted working tree of this branch on top of it).
- **Run datetime (UTC):** 2026-10-18T22:57:10Z (also in the provenance table).
- **Command(s):**
  `uv run python research_notes/gpu_augmentation/compare_pipelines.py`
- **Datasets:** none from disk. 512 natural 96 px images built offline by the
  script from the photos bundled with scikit-image (`skimage.data`: astronaut,
  coffee, chelsea, rocket, immunohistochemistry, hubble_deep_field, retina):
  random square crops downsampled with PIL bicubic, imitating STL-10's downsampled
  photos ([STL-10 canonical](https://cs.stanford.edu/~acoates/stl10/),
  [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10); STL-10
  itself was not used).
- **Backbone:** none (input pipeline only).

## Question
Training is CPU-bound on PIL augmentation (A100 nodes with 4 CPUs per GPU feed
~650 images/s). `lejepa_norm.data.gpu_views` re-implements the augmentation as
batched tensor ops on the device. Does it produce the same views as the PIL
reference (`lejepa_norm.data.views`, kept unchanged), op by op and in
distribution, and which systematic differences remain?

## Setup (deviations from the base config)
- Base `DataConfig` (2 global 96 px views, scale 0.3–1.0, blur; 4 local 48 px
  views, scale 0.05–0.3, no blur; jitter 0.4/0.4/0.2/0.1 at p = 0.8; grayscale
  p = 0.2; blur p = 0.5, sigma 0.1–2.0, kernel 9). CPU, fp32, fixed seeds.
- **Op parity:** 64 images, per-sample parameters drawn once and given to both
  sides. Ground truth for the color ops, blur, flip and normalization is
  `torchvision.transforms.v2.functional` on float tensors, one call per sample.
  The "PIL backend" rows give the same parameters to torchvision's PIL code path,
  which is what the reference pipeline runs (uint8 between ops).
- **Crop + resize parity:** identical boxes (sampled by the GPU sampler) through
  ours, torchvision's antialiased bicubic tensor resize and the PIL path.
  `roi_align` (bilinear, `aligned=True`, `sampling_ratio=2`) is shown for context
  only; it is not used.
- **Distributions:** each pipeline makes 4 views of each of the 512 images per
  view type (2048 views); the PIL pipeline is run twice with different seeds to
  give a noise floor. W1 is the Wasserstein-1 distance (in normalized units) and
  KS the two-sample Kolmogorov–Smirnov statistic, over 200k random pixels per
  channel or over the per-view summaries.

## Results

Provenance:

<!-- BEGIN GENERATED: provenance -->
| item | value |
|---|---|
| HEAD when run | 6506e11 |
| run datetime (UTC) | 2026-10-18T22:57:10Z |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| Pillow | 13.0.0 |
| python | 3.13.0 |
| device | cpu (fp32) |
<!-- END GENERATED: provenance -->

**Per-op parity** (units of 1/255 of the [0, 1] pixel range):

<!-- BEGIN GENERATED: op_parity -->
| op | compared with | mean signed diff, ours - ref (/255) | max abs diff (/255) | mean abs diff (/255) | p99.9 abs diff (/255) |
|---|---|---|---|---|---|
| brightness | v2 tensor op | 0 | 0 | 0 | 0 |
| brightness | PIL backend (reference pipeline) | 0.465 | 1 | 0.465 | 0.997 |
| contrast | v2 tensor op | -6.55e-08 | 1.52e-05 | 8.73e-07 | 1.52e-05 |
| contrast | PIL backend (reference pipeline) | 0.475 | 1.16 | 0.477 | 1.05 |
| saturation | v2 tensor op | -1.71e-08 | 1.52e-05 | 4.36e-07 | 1.52e-05 |
| saturation | PIL backend (reference pipeline) | 0.455 | 1.1 | 0.457 | 1.04 |
| hue | v2 tensor op | 0 | 0 | 0 | 0 |
| hue | PIL backend (reference pipeline) | 0.272 | 8.5 | 0.548 | 6.78 |
| grayscale | v2 tensor op | 0 | 0 | 0 | 0 |
| grayscale | PIL backend (reference pipeline) | -0.011 | 0.525 | 0.234 | 0.517 |
| gaussian blur | v2 tensor op | 1.1e-06 | 0.000182 | 1.44e-05 | 0.000106 |
| gaussian blur | PIL backend (reference pipeline) | -1.41e-05 | 0.5 | 0.224 | 0.499 |
| horizontal flip | v2 tensor op | 0 | 0 | 0 | 0 |
| normalize (units: normalized values x 255) | v2 tensor op | 0 | 0 | 0 | 0 |
<!-- END GENERATED: op_parity -->

**Crop + resize parity** on identical boxes:

<!-- BEGIN GENERATED: crop_parity -->
| op | compared with | mean signed diff, ours - ref (/255) | max abs diff (/255) | mean abs diff (/255) | p99.9 abs diff (/255) |
|---|---|---|---|---|---|
| crop + resize, global (96 px) | v2 tensor bicubic, antialias (float) | -6.25e-07 | 0.000198 | 5.28e-06 | 6.08e-05 |
| crop + resize, global (96 px) | PIL bicubic (reference pipeline) | -0.00236 | 6.64 | 0.278 | 0.961 |
| roi_align bilinear, global (not used) | PIL bicubic (reference pipeline) | -0.00319 | 47.1 | 1.09 | 18.1 |
| crop + resize, local (48 px) | v2 tensor bicubic, antialias (float) | -3.74e-07 | 0.000152 | 6.08e-06 | 6.08e-05 |
| crop + resize, local (48 px) | PIL bicubic (reference pipeline) | -0.0011 | 3.9 | 0.275 | 0.956 |
| roi_align bilinear, local (not used) | PIL bicubic (reference pipeline) | 0.00851 | 54.2 | 1.13 | 18.1 |
<!-- END GENERATED: crop_parity -->

**Whole view** (crop, flip, gated jitter in per-sample order, grayscale, blur,
normalization) with the sampled parameters, against the same parameters pushed
through torchvision v2 ops one sample at a time:

<!-- BEGIN GENERATED: full_view_parity -->
| op | compared with | mean signed diff, ours - ref (/255) | max abs diff (/255) | mean abs diff (/255) | p99.9 abs diff (/255) |
|---|---|---|---|---|---|
| full global view, before normalization | v2 tensor ops, same parameters and order | -1.2e-06 | 0.000228 | 1.38e-05 | 0.000122 |
| full global view, before normalization | PIL code path, same parameters and order | 1.34 | 12.5 | 1.42 | 10 |
| full local view, before normalization | v2 tensor ops, same parameters and order | -7.16e-07 | 0.000304 | 7.8e-06 | 0.000149 |
| full local view, before normalization | PIL code path, same parameters and order | 1.33 | 16.3 | 1.42 | 11.1 |
<!-- END GENERATED: full_view_parity -->

**Sampled parameters** (GPU sampler, 40k draws; torchvision crop sampler, 10k
draws):

<!-- BEGIN GENERATED: sampled_parameters -->
| quantity | GPU sampler | torchvision | test |
|---|---|---|---|
| flip | 0.4991 | 0.5 | binomial sd 0.0025 |
| color jitter applied | 0.7985 | 0.8 | binomial sd 0.0020 |
| grayscale | 0.2003 | 0.2 | binomial sd 0.0020 |
| blur applied (global) | 0.5026 | 0.5 | binomial sd 0.0025 |
| jitter op order (24 permutations) | counts 1606..1726 | uniform (randperm) | chi2 p = 0.855 |
| global crop area fraction, mean | 0.5925 | 0.5934 | KS p = 0.370 |
| global crop log aspect ratio, sd | 0.1574 | 0.1583 | KS p = 0.276 |
| local crop area fraction, mean | 0.1747 | 0.1749 | KS p = 0.961 |
| local crop log aspect ratio, sd | 0.1667 | 0.1657 | KS p = 0.069 |
<!-- END GENERATED: sampled_parameters -->

**Output pixel distributions** (normalized values):

<!-- BEGIN GENERATED: pixel_distributions -->
| view | channel | PIL mean | GPU mean | PIL std | GPU std | W1 PIL vs GPU | W1 PIL vs PIL (seed 1) | KS PIL vs GPU | KS PIL vs PIL (seed 1) |
|---|---|---|---|---|---|---|---|---|---|
| global | R | 0.1367 | 0.1428 | 1.164 | 1.157 | 0.01085 | 0.01103 | 0.02432 | 0.00702 |
| global | G | -0.2574 | -0.2456 | 0.992 | 0.9926 | 0.01201 | 0.02271 | 0.02194 | 0.01235 |
| global | B | -0.2522 | -0.2279 | 0.9157 | 0.9188 | 0.02574 | 0.01743 | 0.02485 | 0.01488 |
| local | R | 0.1219 | 0.1483 | 1.168 | 1.166 | 0.02501 | 0.01047 | 0.02432 | 0.00852 |
| local | G | -0.2939 | -0.2364 | 0.978 | 1.007 | 0.06616 | 0.03172 | 0.03044 | 0.01663 |
| local | B | -0.2658 | -0.2169 | 0.9137 | 0.9367 | 0.05016 | 0.01619 | 0.02897 | 0.01144 |
<!-- END GENERATED: pixel_distributions -->

**Per-view statistics** (one number per view, normalized values):

<!-- BEGIN GENERATED: view_statistics -->
| view | statistic | PIL mean | GPU mean | W1 PIL vs GPU | W1 PIL vs PIL (seed 1) | KS PIL vs GPU | KS PIL vs PIL (seed 1) |
|---|---|---|---|---|---|---|---|
| global | per-view mean | -0.1243 | -0.1102 | 0.02399 | 0.02379 | 0.02197 | 0.0293 |
| global | per-view std | 0.6206 | 0.616 | 0.00621 | 0.007438 | 0.01855 | 0.02002 |
| global | per-view saturation (channel spread) | 0.7914 | 0.787 | 0.01047 | 0.01381 | 0.01221 | 0.01611 |
| global | per-view sharpness (mean abs horizontal gradient) | 0.05784 | 0.05736 | 0.00152 | 0.0008891 | 0.03076 | 0.01514 |
| local | per-view mean | -0.1459 | -0.1017 | 0.04425 | 0.01499 | 0.02441 | 0.01465 |
| local | per-view std | 0.5794 | 0.5737 | 0.009158 | 0.008788 | 0.02051 | 0.02002 |
| local | per-view saturation (channel spread) | 0.8045 | 0.7855 | 0.02432 | 0.02138 | 0.02539 | 0.02246 |
| local | per-view sharpness (mean abs horizontal gradient) | 0.0734 | 0.07295 | 0.001916 | 0.001432 | 0.02783 | 0.01953 |
<!-- END GENERATED: view_statistics -->

## Interpretation
**The GPU pipeline is torchvision's tensor semantics, exactly.** With identical
parameters, every op matches `torchvision.transforms.v2.functional` to float
rounding (≤ 2e-4/255 per op; the whole view, including the per-sample jitter gate
and op order, to ≤ 3e-4/255). Crop + resize reproduces torchvision's antialiased
bicubic resize (Pillow's algorithm: a = -0.5, kernel widened when downscaling) to
2e-4/255. The samplers match torchvision's: flip, jitter gate, grayscale and blur
rates are within 1.1 binomial standard deviations of 0.5 / 0.8 / 0.2 / 0.5, the 24
jitter orders are uniform (chi² p = 0.86), and crop area and aspect ratio
distributions are indistinguishable from `RandomResizedCrop.get_params` (KS
p ≥ 0.07).

**Bicubic was kept; there is no bicubic -> bilinear change.** The brief allowed
`roi_align` / `grid_sample`. `roi_align` is bilinear and, on the same boxes,
differs from the PIL reference by 1.1/255 on average and up to 47–54/255 at
edges (p99.9 = 18/255): bilinear upsampling (most crops are upsampled) is
visibly softer. The per-sample interpolation-matrix implementation costs two
small batched matmuls per view and removes that difference, so it replaced
`roi_align`.

**Systematic differences from the PIL reference** (which runs the same torchvision
transforms on uint8 PIL images):
1. **Brightness bias of +1.3/255 (GPU brighter).** Pillow's `ImageEnhance`
   (brightness, contrast, saturation) blends in float and **truncates** to uint8,
   so each applied op darkens the PIL output by ~0.46/255 on average (signed diffs
   0.455–0.475/255 in the op table); the PIL hue path (uint8 HSV) adds +0.27/255.
   With identical parameters the whole view is 1.34/255 brighter on the GPU (mean
   abs diff 1.42/255). This is the main reason the GPU channel means sit
   0.01–0.06 (normalized units) above the PIL ones in the distribution table. The
   GPU output is the unbiased one.
2. **Hue quantization.** PIL rotates hue in 8-bit HSV (1/256 turn steps,
   quantized S and V); differences up to 8.5/255 on saturated pixels (p99.9
   6.8/255). After a later contrast or saturation op this grows to ~10–16/255 on
   rare pixels in the whole-view comparison (p99.9 10–11/255, mean 1.4/255).
3. **No uint8 rounding between ops.** PIL rounds after every op and after each
   resize pass (and saturates the horizontal pass at 0/255 before the vertical
   one): ±0.5/255 per step, up to 4–7/255 on isolated edge pixels after the
   resize. The GPU keeps continuous values (the same as torchvision's float
   tensor path).
4. **Random streams differ.** The GPU draws from one device generator per run
   (per shard in the embedding dump), so individual views are not the PIL views;
   only their distribution is the same. CPU and CUDA generators give different
   streams for the same seed.

**Distributions.** Pixel and per-view distributions of the two pipelines differ by
about the PIL-vs-PIL (other seed) noise floor for global views (W1 0.011–0.026 vs
0.011–0.023; per-view std, saturation and sharpness at the floor). Local views
show the largest gap (channel means up to 0.06 higher, W1 up to 2–3x the single
floor draw), consistent with the +1.3/255 bias (≈ 0.02 normalized) plus
seed-to-seed spread of local-view means (≈ 0.01–0.02 across four seeds in a
side check, not tabulated). Output std, saturation and sharpness agree to within
1–2%.

**Verdict: acceptable for this research.** The differences are 8-bit
quantization artifacts of the PIL path, two orders of magnitude below the
augmentation strengths themselves (brightness alone moves pixels by up to
±0.4, i.e. ±100/255). The study compares runs that all use the same (GPU)
pipeline, so no comparison mixes the two. Runs trained with the PIL pipeline
before this change see inputs ~1.3/255 darker on average after jitter; treat
them as a slightly different augmentation, not as directly comparable seeds.
The dataset normalization statistics (`data.mean/std`, computed on raw images)
are unaffected.

## Next steps
- Benchmark on an A100 (`scripts/bench_data.py --skip-cpu --train-like`, command
  in the PR) to confirm that ~4 CPUs now feed the GPU, and record the numbers in a
  follow-up note.
- Repeat this comparison on CUDA (TF32 is disabled for the blur convolutions; the
  resize matmuls use full fp32 under torch's default matmul precision) to check
  that CUDA results match the CPU ones to float rounding.
- Optionally re-run one short PIL-pipeline training run against the GPU pipeline
  (same seed, same epochs) to confirm that the k-NN accuracy and radius
  statistics agree within seed noise.
