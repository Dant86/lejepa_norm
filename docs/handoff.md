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
`research_notes/sigreg_reference/reference_comparison.md`.

| Milestone | Status |
|---|---|
| M0 scaffold, M1 data, M2a models/losses, M2b training, M3a eval/analyze/embed | done, on `master` |
| GPU-side augmentation, in-RAM data loading, preemption-safe resume, sweep template | done |
| M2c `radius-marginal`: Variant A, lambda in {0.02, 0.05, 0.1} x seeds {0, 1, 2} | **done** (note: `research_notes/radius_marginal/lambda_sweep.md`) |
| Test-time diagnostic of the nine final checkpoints (STL-10 test embeddings) | **done** |
| M3b `radius-conditional`: ICC, redundancy R^2, difficulty, downstream, then **go/no-go** | **next** |
| M4 SUSReg / radial loss, variants B and C | not started (only after the go/no-go; `build_loss` raises for B and C) |

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
- Larger-lambda sweep was judged unnecessary for now.

## 2. What exists where

Code: `src/lejepa_norm/` (see `docs/architecture.md` for the diagrams), apps in
`apps/` (`prepare_data`, `train`, `embed`, `analyze`; keep it to these four),
SLURM templates in `slurm/` (`slurm/README.md` explains preemption/resume),
configs in `configs/base.yaml`. Notes: `research_notes/`.

On the cluster, artifacts are under the `.env` locations:
- `<data_dir>`: prepared STL-10 / DTD / SVHN arrays and STL-10-C
  (`stl10c/<corruption>/<severity>/images.npy`), `stl10/channel_stats.json`.
  **Not verified in this session:** that the DTD/SVHN splits and all 50 STL-10-C
  arrays (10 corruptions x 5 severities) were built; check before M3b (step 0 below).
- `<runs_dir>/radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}/`: `metrics.jsonl`,
  `config.resolved.yaml`, `last.pt`, `epoch_XXXX.pt` (every 10 epochs), `resume.pt`
  (a small "complete" marker after a finished run), job logs.
- `<embed_dir>/<run>/stl10_test/`: embedding dumps of the nine final checkpoints,
  **only the `stl10_test` set** (K = 8 augmented views + the center view).
- `<notes_dir>/radius_marginal/diag/<run>.md`: the per-run marginal-radius notes.

## 3. Next task: M3b (suggested plan)

Subjects: Variant A at **lambda = 0.05 (seed 0) as the main run** (the design default),
and **lambda = 0.1 (seed 0)** as the comparison closest to the target. The go/no-go is
on Variant A only; thresholds in the Technical Design are proposals.

0. **Verify the prepared data** (cheap, on the login node):
   `ls <data_dir>/stl10c | wc -l` (10), `find <data_dir>/stl10c -name images.npy | wc -l`
   (50), and that `<data_dir>/{dtd,svhn}/test_images.npy` exist. If not, run
   `sbatch slurm/prepare_data_build.sbatch` (idempotent, skips finished arrays).
1. **Embed all five evaluation sets** (the `stl10_test` shards already exist and are
   reused because the template passes `--resume`):
   ```bash
   sbatch --cpus-per-task=4 slurm/embed.sbatch configs/base.yaml \
     radius_marginal_lam0.05_s0 last.pt
   ```
   Default `--sets` is all five (`stl10_train`, `stl10_test`, `stl10c`, `dtd_test`,
   `svhn_test`). STL-10-C is ~400k images x 9 views: about 11-12 GB of shards per
   checkpoint and the dominant cost (runtime **unmeasured**; the template allows 12 h
   and is preemption-safe, resubmitting the same command continues).
2. **Analyze** (`apps/analyze.py` per-run mode): needs memory, because the STL-10-C
   embeddings are loaded whole (h alone is ~7.5 GB). Request more than the template's
   32 GB to be safe:
   ```bash
   sbatch --mem=64G slurm/analyze.sbatch configs/base.yaml radius_marginal_lam0.05_s0 \
     radius_conditional/variant_a_lam0.05.md
   ```
   Figures need Chrome for Kaleido (`LEJEPA_NORM_KALEIDO_CHROME`, installed once from
   the login node); pass `--no-figures` to skip them. Use `--set eval.num_bootstrap=...`
   and `--set eval.max_kernel_images=...` to trade accuracy for speed.
3. Write the note under `research_notes/radius_conditional/` on a branch named
   `radius-conditional` (template in the Technical Design, command + UTC time +
   commit in the header), then evaluate the go/no-go.
4. Also open: radius histograms against chi_256 for the sweep's final checkpoints
   (generated from the existing `stl10_test` dumps via `lejepa_norm.plotting`).

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
