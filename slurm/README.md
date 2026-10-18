# SLURM templates

Batch templates for the UChicago DSI cluster
([cluster policy](https://cluster-policy.ds.uchicago.edu/), batch jobs and
preemption: [batch-jobs guide](https://cluster-policy.ds.uchicago.edu/using-the-cluster/batch-jobs/)).

| Template | App | Resources | Resume behavior |
|---|---|---|---|
| `prepare_data_build.sbatch` | `apps/prepare_data.py --stage build` | CPU | skips arrays already built |
| `train.sbatch` | `apps/train.py --resume auto` | 1 GPU | continues from `resume.pt` |
| `embed.sbatch` | `apps/embed.py --resume` | 1 GPU | writes only missing shards |
| `analyze.sbatch` | `apps/analyze.py` | CPU | recomputes (idempotent) |

All templates request `--partition=general --qos=general --time=12:00:00`
(the general QoS maximum) with `--signal=B:USR1@300 --requeue`, and launch the app
through `run_preemptable` from `common.sh`.

## One-time setup

1. On the login node, in the repo checkout: `uv sync --locked` (creates
   `.venv/`; override the interpreter with `LEJEPA_NORM_PYTHON`).
2. Copy `.env.example` to `.env` and point every variable at the real locations
   for this cluster. The templates read every path from `.env` (override the file
   with `LEJEPA_NORM_ENV_FILE`) and write only there: checkpoints, resume files,
   resolved configs, `metrics.jsonl` and job logs all go under
   `LEJEPA_NORM_RUNS_DIR`, embedding dumps under `LEJEPA_NORM_EMBED_DIR`. The run
   directories are created on demand, so no `mkdir` is needed as long as `.env`
   points at the real locations. Never commit real paths.
3. `python apps/prepare_data.py --stage download` on the login node (needs
   network; compute nodes are assumed offline), then
   `sbatch slurm/prepare_data_build.sbatch`.

## Usage

Submit from the repo root (the templates `cd` to `$SLURM_SUBMIT_DIR`):

```bash
sbatch slurm/prepare_data_build.sbatch
sbatch slurm/train.sbatch configs/base.yaml a_l0.05_s0 --set loss.lambda=0.05
sbatch slurm/embed.sbatch configs/base.yaml a_l0.05_s0 last.pt
sbatch slurm/analyze.sbatch configs/base.yaml a_l0.05_s0 radius-conditional/variant_a.md
```

Each job appends its log to `<LEJEPA_NORM_RUNS_DIR>/<run_name>/slurm-<app>-<jobid>.log`
(`_prepare_data` is the run name of the data build). A requeued job keeps its job
ID, so all attempts land in the same log; each attempt starts with a header
showing `restart #N` (`SLURM_RESTART_COUNT`). The `slurm-%x-%j.out` file in the
submit directory only catches errors before the run directory is known.

GPU type: the GPU templates request `--gres=gpu:a100:1` so that run times are
comparable across runs. The general partition also has A40, L40S, H100 and H200
nodes; change the type in the template (see `sinfo -o "%P %G"`) to use another.

## Preemption and resume

The general QoS is preemptable (after 5 minutes of runtime). On preemption, and
also 300 s before the wall-time limit, Slurm sends `SIGUSR1` to the batch shell,
kills the job after a 5-minute grace period and requeues it under the same job
ID when it exits with code 99.

`run_preemptable CMD ...` (in `common.sh`, adapted from the guide's wrapper):

- starts `CMD` with `setsid` in its own process group and waits for it;
- on `USR1`: forwards it to that process group, waits for `CMD`, then exits
  **99** (requeue), whatever `CMD` exited with;
- on `TERM` (e.g. `scancel --batch`): forwards it, waits, exits **143** (no
  requeue);
- otherwise exits with `CMD`'s exit code (a `CMD` exiting 99 on its own also
  requeues).

What each app does when it receives `USR1`:

- **train**: a handler sets a flag; after the current optimizer step, training
  writes `<run_dir>/resume.pt` atomically and exits 99. `resume.pt` (separate from
  the `epoch_XXXX.pt` / `last.pt` model checkpoints, whose format is unchanged)
  holds the model, projector and optimizer state, global step, epoch and step
  within the epoch, the partial epoch statistics and the RNG state. It is also
  rewritten at the end of every epoch. The requeued job runs
  `apps/train.py --resume auto`, which truncates `metrics.jsonl` to the resumed
  step (no duplicate lines), continues the interrupted epoch with the same data
  order (per-epoch seeded shuffle, skipped forward) and the same LR schedule
  (a function of the global step). When the run finishes, `resume.pt` is replaced
  by a small "complete" marker, so resubmitting a finished run exits 0
  immediately. Augmentation runs on the GPU from one seeded generator whose
  state is saved in `resume.pt`, so the resumed run sees exactly the same
  augmented views as an uninterrupted one, whatever `data.num_workers` is
  (training and embedding ignore it; only `prepare_data build` uses it). On CPU
  the resumed run is bit-identical; on CUDA, non-deterministic kernels (cuDNN
  benchmark mode) can still cause float-level differences.
- **embed**, **prepare_data build**, **analyze**: no handler; the process dies
  and the requeued job re-runs it. Embedding shards and prepared arrays are
  written atomically (temp file + rename), so the rerun skips completed work
  (`embed --resume` checks that each kept shard holds the images it expects).
  `analyze` recomputes everything.

`train --resume auto` refuses a run directory that has model checkpoints but no
`resume.pt` (a run from before resume support, or a different run) and one whose
`config.resolved.yaml` differs from the requested config (ignoring
`data.num_workers`).

## Testing the signal path

Send the preemption signal to the batch shell of a running job, as Slurm would:

```bash
scancel --signal=USR1 --batch <jobid>
```

The job log should show `USR1: preemption or time-limit warning`, then (for
training) `preemption requested: saved .../resume.pt at step N` and
`exiting 99 to request a requeue`. Check the requeue with
`sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed` and
`scontrol show job <jobid>` (`Restarts=`); the next attempt logs
`resuming after step N`.

## Sweeps (job arrays)

`sweep.sbatch` runs a lambda x seed grid as one job array: every task is its own
independent, preemptable training run with its own run directory
(`<RUN_PREFIX>_lam<lambda>_s<seed>`), and a requeued task resumes like a plain
`train.sbatch` job.

```bash
LAMBDAS="0.02 0.05 0.1" SEEDS="0 1 2" \
  sbatch --array=0-8 slurm/sweep.sbatch configs/base.yaml radius_marginal
```

- The array range must be `0-(n_lambdas * n_seeds - 1)`; a task outside the grid
  fails immediately with the correct range. The task index runs over lambdas
  first, then seeds.
- Add `%N` to cap concurrency (`--array=0-8%3`). The DSI guide suggests `%5`-`%10`
  for GPU-heavy arrays; `general` allows 24 running jobs per user.
- Override resources at submit time, e.g. `--cpus-per-task=4`.
- Resubmit a single failed task with `--array=<index>`.
- Everything after `RUN_PREFIX` is passed to `apps/train.py` as extra arguments.
