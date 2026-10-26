# E3: training without the radius (SIGReg on the sphere) vs the matching radius runs
- **Branch / version:** `sigreg-sphere` @ <commit> (see "Commands" for what each step used)
- **Run datetime (UTC):** pre-registration committed 2026-10-26T00:55Z (before any training result or evaluation existed); results: <filled in below>
- **Command(s):** <filled in below>
- **Datasets:** STL-10 unlabeled (100k, pretraining), STL-10 `train`/`test` (bank / queries), STL-10-C (generated from `test`), DTD `test`, SVHN `test`. STL-10: [canonical](https://cs.stanford.edu/~acoates/stl10/), [HF `tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10); the **canonical Stanford archive was used** (via the prepared arrays), the HF page is linked for reference. DTD: [canonical](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [HF `tanganke/dtd`](https://huggingface.co/datasets/tanganke/dtd). SVHN: [canonical](http://ufldl.stanford.edu/housenumbers/), [HF `ufldl-stanford/svhn`](https://huggingface.co/datasets/ufldl-stanford/svhn); the canonical archives (prepared arrays) were used.
- **Backbone:** ResNet-18 from scratch ([HF architecture reference](https://huggingface.co/timm/resnet18.a1_in1k), weights not used), projector 512 -> 2048 -> 2048 -> 256.
- **Scope:** h is not analyzed; LeJEPA's product is z.

## Question

Is the embedding radius needed *for training*? E1 and the follow-ups found the radius of z (nearly) unused by readers and highly predictable from the direction. Intervention: `loss.radius_free=true` replaces z by `z~ = sqrt(d) z/||z||` before **both** the invariance loss and SIGReg, so the radius of the loss-visible embedding is exactly constant (`sqrt(256) = 16`) and the loss gives the raw radius no gradient. Everything else is `configs/base.yaml` with lambda = 0.05, seeds 0, 1, 2; the comparison runs are `radius_marginal_lam0.05_s{0,1,2}` (same config, `radius_free=false`).

Caveat (documented in `lejepa_norm.losses.lejepa.project_to_sphere`): SIGReg targets N(0, I_d); the 1-D projections of `z~` (uniform on the sphere of radius sqrt(d)) are Gaussian only approximately, with a mismatch of order 1/d (d = 256; kurtosis 3 - 6/(d+2) = 2.977 vs 3). The sliced Epps-Pulley statistic therefore has a small non-zero minimum on the sphere; the direction-only target is "as Gaussian as a sphere can be". The backbone feature h keeps its radius (out of scope). The raw projector output z of a radius-free model has an unconstrained radius (it receives no gradient except through weight decay / optimizer dynamics): **all evaluations of radius-free runs use `z~` (direction only, norm sqrt(d))**; the online k-NN monitor is cosine k-NN and unaffected.

## Pre-registration (written before any result of these runs was computed or looked at)

Compared with the matching radius runs (same lambda and seed index; different trained models, so seed index pairing is only bookkeeping). Equivalence margin **0.5 accuracy points** (`eval.equivalence_margin`) for accuracy, **0.5 points** for retrieval metrics (`eval.retrieval_margin`), **0.01 AUROC** for OOD (`ood_radius.OOD_AUROC_MARGIN`), all from the existing configs and fixed before this experiment. Statistics: paired bootstrap over test images within each training seed (the same images are scored by both models), averaged over the 3 seeds, 1000 replicates, TOST-style verdict (`randomization.verdict`). **Caveat stated in advance: the intervals cover test-image sampling only; training-seed variance of two *different* trained models is not covered beyond averaging 3 seeds.**

Primary reference = the radius model as trained (raw z, with radius). Secondary reference = the radius model with its radius removed at test time (unit-norm z, E1's `constant` condition, retrained readers).

Predictions:

1. **Accuracy equivalent.** Euclidean k-NN, linear probe and MLP accuracy on clean STL-10 test (readers fitted on STL-10 train, E1 protocol) of the radius-free model are equivalent (90% CI within +-0.5 points) to the radius model's, for each of the three readers. Held iff all three verdicts are `equivalent`. Same comparison on STL-10-C (all 50 corruption x severity cells pooled, and per corruption) is reported with the same verdict rule but is not part of the prediction.
2. **Instance retrieval recovers through direction.** View-to-view instance retrieval R@1 and R@5 of the radius-free model are not worse than those of the radius model's *constant-radius* variant (the radius model's direction-only retrieval): held iff the lower end of the 90% CI of (radius-free minus radius-constant) exceeds -0.5 points for both R@1 and R@5 (one-sided non-inferiority at the retrieval margin). (Not predicted either way: the comparison with the radius model's as-trained retrieval.)
3. **OOD (direction-only) is lower than radius-using.** On SVHN (ID = STL-10 test, bank = STL-10 train), the AUROC of the radius-free model's direction-only score (k-NN distance of the normalized z) is lower than the best radius-using score of the radius model, where "best radius-using" = the larger seed-mean AUROC of the two pre-registered deployable scores `kNN raw z` and `direction + radius deviation` (the optimistic best-weight oracle and the post hoc score are reported but not used). Held iff the 95% paired bootstrap CI of the difference lies entirely below 0; "direction of effect only" if the point difference is negative but the CI contains 0. The same comparison on DTD is reported, not predicted.

Secondary expectation (descriptive, from the E1/retrieval results on the radius model, where unit-norm z improved class-level mAP by about 4 points): class-level retrieval mAP of the radius-free model is higher than the radius model's as-trained mAP and close to its constant-radius variant. Not a pre-registered verdict.

Training health (reported, not predicted): 200 epochs completed with finite losses, invariance loss and `knn_acc_h` / `knn_acc_z` curves compared with the radius runs; training is called healthy if all 3 runs complete, no value is non-finite, and the final k-NN monitor accuracy is within 2 points of the radius runs' (a looser, descriptive bar than the equivalence margin because the monitor is a 20-epoch-interval, single-pass estimate).

## Setup (deviations from the base config)

`loss.radius_free=true` only. Runs `sigreg_sphere_lam0.05_s{0,1,2}`, trained with the sweep template in parallel; GPU types and job ids below.

## Results

## Interpretation

## Next steps
