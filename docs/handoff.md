# Handoff: project state and how to continue on the cluster

Written for a new session (human or agent) that continues the work **on the DSI
cluster**. Read `CLAUDE.md` first (binding rules); this file is the current state
and the practical knowledge that is not in the code. It contains no real paths: the
locations come from `.env` (see `.env.example`); below, `<runs_dir>`, `<embed_dir>`,
`<data_dir>` and `<notes_dir>` mean `LEJEPA_NORM_RUNS_DIR`, `..._EMBED_DIR`,
`..._DATA_DIR` and `..._NOTES_DIR`.

## 1. Where the project stands

Research question: does LeJEPA's SIGReg make the embedding radius follow chi_d, and
is the radius an image-level signal (ICC, redundancy with direction, difficulty)?
Design lives in the Technical Design / Research Outline notes of the research vault
(not in this repo). **One deliberate deviation from that note:** SIGReg is computed
**per view** (N = per-view batch, shared directions, averaged over views), as in the
paper and the official code, not on all views pooled; the vault note still says
"concatenation" and should be corrected. See
`research_notes/sigreg_reference/reference_comparison.md` (its "Training layout" and
"Lambda combination" paragraphs still describe the earlier concatenated convention
and are historical; the code and training runs use the per-view form).

| Milestone | Status |
|---|---|
| M0 scaffold, M1 data, M2a models/losses, M2b training, M3a eval/analyze/embed | done, on `master` |
| GPU-side augmentation, in-RAM data loading, preemption-safe resume, sweep template | done |
| M2c `radius-marginal`: Variant A, lambda in {0.02, 0.05, 0.1} x seeds {0, 1, 2} | **done** (`research_notes/radius_marginal/lambda_sweep.md`) |
| M3b `radius-conditional` (ICC, redundancy R^2, difficulty, downstream) | **done** (`research_notes/radius_conditional/`); the go/no-go on variants B/C is **still not called** |
| Radius-nuisance program E1-E4, E6, E7 plus controls, confidence, view-signal | **done and merged** (table below) |
| E5 architecture generality (ResNet-50, ViT-S, ViT-B) | not started |
| M4 SUSReg / radial loss, variants B and C | not started (`build_loss` raises for B and C) |

### Radius-nuisance program (spec: `research_notes/radius_nuisance/spec.md`)
Claim under test: the LeJEPA radius `||z||` is a nuisance parameter. All analyses are
z only unless a note says otherwise. Everything below is merged to `master`.

| Experiment | Notes | Code entry point |
|---|---|---|
| E1 usage (radius randomization, retrained and frozen readers) | `radius_randomization/`, `radius_usage/` (low labels, retrieval, frozen linear, shift) | `analyze --randomize-radius`, `--lowlabel`, `--retrieval`, `--frozen-linear`, `--shift` |
| E2 is the predictability LeJEPA-specific (random init, SimCLR, VICReg, supervised) | `radius_baselines/` (train, controls, h extension) | `train` with the baseline losses, `embed --random-init`, `analyze --baseline-controls` |
| E3 training without the radius (`loss.radius_free`) | `sigreg_sphere/sphere_vs_radius.md` | `train --set loss.radius_free=true`, `analyze --compare-prefix` |
| E4 SIGReg power at the observed kappa (synthetic) | `sigreg_power/power.md` | `analyze --sigreg-power` |
| E6 clean-to-corrupted transfer | `radius_transfer/` | `analyze --transfer` |
| E7 dynamics over epochs | `radius_dynamics/epoch_sweep.md` | `analyze --epoch-sweep` |
| Controls: SVHN low-level statistics, Places365 OOD | `radius_usage/svhn_control*.md`, `radius_usage/places_control.md` | `analyze --svhn-control` |
| Confidence: radius vs reader error / risk-coverage | `radius_confidence/` | `analyze --confidence-unit`, `--confidence` |
| View signal: radial vs tangential corruption displacement, per-view reliability | `radius_view_signal/` (with `preregistration.md`) | `analyze --view-signal` |

Read the notes for numbers; do not trust this table for them. The current framing:
the radius is a stable, image-level, largely direction-predictable property with no
class information for the readers tested, but it is not noise (it identifies an image
across views and carries some OOD and per-view-reliability signal). Interpretation
rule: the high same-view R^2 of log-radius given direction is **not** specific to
LeJEPA (random init and the baselines reach similar values), so it is not evidence
that SIGReg shapes the radius; weight cross-view stability, the usage experiments and
view reliability instead. Everything is STL-10, ResNet-18 and at most 3 seeds, and the
SVHN comparisons are confounded by upsampling.

### Results so far (details in the note)
- `kappa_z` (= Var(||z||^2)/(2d), 1 for N(0, I)) at epoch 200: **26.3, 20.2, 14.6** for
  lambda 0.02, 0.05, 0.1; monotone in lambda, seed spread <= 0.1. It rises during
  training after a minimum at epochs 3-4. k-NN accuracy (backbone) is about 0.79 for
  all lambda: no accuracy cost in this range; the invariance loss rises with lambda.
- Test-time global-view `kappa_z` matches the pooled training value within ~3%
  (pooling artifact ruled out). `E||z||^2/d` is ~1 (scale matched) but the relative
  spread of `||z||` is 0.17-0.23 against 0.044 for a Gaussian. Hypothesis (not
  measured): the sliced Epps-Pulley test has little power against this spread.
- **Open observation, deliberately not pursued:** at lambda = 0.02 the backbone feature
  `h` has a heavy-tailed norm on test images in all three seeds (`kappa_h` 15-66 on
  center views vs ~0.4 in the training log).

## 2. What exists where

Code: `src/lejepa_norm/` (see `docs/architecture.md` for the diagrams), apps in
`apps/` (`prepare_data`, `train`, `embed`, `analyze`; keep it to these four),
SLURM templates in `slurm/` (`slurm/README.md` explains preemption/resume),
configs in `configs/base.yaml`. Notes: `research_notes/`.

On the cluster, artifacts are under the `.env` locations:
- `<data_dir>`: prepared STL-10 / DTD / SVHN arrays and STL-10-C
  (`stl10c/<corruption>/<severity>/images.npy`), `stl10/channel_stats.json`.
  Verify with the data check in section 3 before new embedding jobs.
- `<runs_dir>/radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}/`: `metrics.jsonl`,
  `config.resolved.yaml`, `last.pt`, `epoch_XXXX.pt` (every 10 epochs), `resume.pt`
  (a small "complete" marker after a finished run), job logs.
- `<embed_dir>/<run>/<set>/`: embedding dumps (K = 8 augmented views + the center
  view per image) of the sweep, baseline and `sigreg_sphere` runs; which sets exist
  per run varies, check with `ls` (analyses skip missing sets).
- `<notes_dir>/radius_marginal/diag/<run>.md`: the per-run marginal-radius notes.

## 3. Open items (current)

1. **E5 architecture generality**: ResNet-50, ViT-S, ViT-B (spec: E5). Needs new
   backbones (a ViT backbone is not implemented yet), size the compute from a short
   ViT-S pilot first, then rerun the E1/E2 readers on the new embeddings.
2. **Wider sweeps**: lambda beyond {0.02, 0.05, 0.1} (kappa_z is monotone in lambda
   over this range; the target kappa of 1 is not reached) and embedding-dimension
   sweeps (everything uses d = 256).
3. **Places365 in the confidence analysis**: the `places_test` set exists in the data
   and embedding layout (`dumps.PLACES_TEST`), but the radius-confidence notes were
   produced before it; recompute with `analyze --confidence-unit --confidence-part
   ood` and re-aggregate.
4. **Go/no-go for variants B and C (SUSReg / radial loss)** is still not called. The
   M3b and nuisance results are the input; thresholds in the Technical Design are
   proposals. `build_loss` raises for B and C until then.
5. Smaller loose ends: radius histograms against chi_256 for the sweep checkpoints;
   the heavy-tailed backbone-feature norm at lambda = 0.02 (see section 1); a
   cleaner OOD control with STL-10-like low-level statistics to replace SVHN.

Before running any analysis that needs all five dumps on the cluster, verify the data
(`ls <data_dir>/stl10c | wc -l` is 10, 50 `images.npy` under it, and
`<data_dir>/{dtd,svhn,places365}` splits exist; `sbatch slurm/prepare_data_build.sbatch`
is idempotent). The SLURM templates for each analysis are in `slurm/`
(`embed`, `analyze`, `frozen_linear`, `lowlabel`, `baseline_controls`, `confidence`,
`view_signal`, `sigreg_power`, `embed_epochs`, `analyze_epochs`, `embed_random_init`).
STL-10-C view-signal units need about 64 GB of RAM.

## 4. Cluster facts learned (DSI cluster; confirm against the docs if in doubt)

- Partition `general`, QoS `general` (preemptable, 12 h limit, 24 running jobs/user).
  Preemption sends `SIGUSR1` (5-minute grace) and requeues; the templates implement
  the docs' contract (`--signal=B:USR1@300`, `--requeue`, exit code 99), training
  resumes from `resume.pt` (`--resume auto`), embed with `--resume`.
- GPU: pin the model, `--gres=gpu:a100:1` (nodes mix A40/A100/L40S/H100/H200; the
  `dev` partition is one A40 node with a **10-minute** limit, good for smoke tests).
- **CPUs are the scarce resource**: A100 nodes have 32 CPUs for 8 GPUs (4 per GPU). Do
  not request more than ~8 CPUs per GPU job. Augmentation now runs on the GPU, so 4
  CPUs suffice for training. `billing` charges 1 per CPU and 11 per GPU.
- Measured: training ~2,200 images/s at 4 CPUs (~48 s/epoch, ~2.45 h for 200
  epochs); MaxRSS 13-17 GB at 8-16 DataLoader workers (before the GPU pipeline).
- Home has a 50 GB quota; scratch files unused for 60 days are purged: copy final
  results somewhere backed up before then. Compute nodes may have no network: all
  downloads (`prepare_data --stage download`, Chrome, `uv sync`) run on the login
  node.
- Data loading: random reads of memory-mapped arrays on the cluster filesystem cost
  ~100 ms/image; arrays are therefore read into RAM at start-up (`data.datasets`).
- `sbatch` command-line options override the script's `#SBATCH` lines. Job logs go to
  `<runs_dir>/<run>/slurm-<app>-<jobid>.log` (they record host, commit, restarts).

## 5. Working conventions that bit us

- **Stage files by name.** `git add -A` once swept untracked local files (notebooks,
  metrics dumps) into a PR. Pre-commit runs ruff, the import-module rule and pyright
  on commit; a failed hook means the commit did not happen: fix and commit again.
- One research question = one branch (`radius-marginal`, `radius-conditional`, ...)
  with notes under `research_notes/<idea>/`; result tables are generated by code
  (`apps/analyze.py`, `reporting.update_section`), the exact command and UTC time go
  at the top of each note. PRs use regular merges (not squash).
- CI (ruff, import rule, pyright strict, pytest ~1 min) must pass before merging.
  `make check` is the local equivalent; `make test-fast` skips the `slow` tests.
- Local `.env` files and run outputs never go into the repo; `.env.example` has
  placeholders only.
- The official LeJEPA code is CC BY-NC 4.0: never copy it; it was only used as a
  numerical reference, loaded by path from outside the repo.

## 6. Known gaps and caveats

- Vendored STL-10-C corruptions were not compared numerically against upstream
  `imagecorruptions`; `frost` and `spatter` are not vendored (need OpenCV and
  textures; the full-set flag covers the other 17).
- Global-view blur probability (0.5) and sigma range are assumptions; the GPU
  augmentation is ~1.3/255 brighter than the PIL pipeline (documented in
  `research_notes/gpu_augmentation/pipeline_comparison.md`); runs trained with the
  PIL pipeline are not seed-comparable with later ones.
- `reporting.py` writes notes with a plain `write_text`; a preemption during `analyze`
  could truncate a note (follow-up: atomic write).
- Per-view-type (global vs. local) diagnostics (`kappa_z_global`, `kappa_z_local`,
  `z_norm_*_global/local`, and the same for `h`) exist in the training logs of runs
  started after they were added; the nine sweep runs do not have them.
- Sweep reproducibility: GPU augmentation is seeded from a single generator, so
  results do not depend on the worker count; CUDA non-determinism can still cause
  float-level differences on resume.
