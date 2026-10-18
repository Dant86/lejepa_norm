"""Pretrains one model from a config; outputs go to `<runs_dir>/<run-name>/`.

Writes `config.resolved.yaml`, `metrics.jsonl`, checkpoints (`epoch_XXXX.pt`,
`last.pt`) and the resume state `resume.pt` into the run directory under
`LEJEPA_NORM_RUNS_DIR`. Data is read from `LEJEPA_NORM_DATA_DIR`. No network
access is needed.

Preemption: SIGUSR1 makes training save `resume.pt` at the next step boundary and
exit with code 99 (requeue). `--resume auto` continues from `<run_dir>/resume.pt`
if it exists (exiting 0 at once if the run already finished) and starts fresh
otherwise; `--resume PATH` continues from a specific resume file. Without
`--resume`, a run directory that was already used is refused.

Usage:
  python apps/train.py --run-name a_l0.05_s0 --set loss.lambda=0.05 --set seed=0
  python apps/train.py --run-name a_l0.05_s0 --resume auto --set ...
"""

import argparse
import collections.abc
import pathlib
import sys

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.train import loop
from lejepa_norm.train import resume

_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
_RESOLVED = "config.resolved.yaml"
AUTO = "auto"


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    """Parses arguments, saves the resolved config and runs training.

    Returns:
        Process exit code: 0 when the run is complete, `resume.REQUEUE_EXIT_CODE`
        when it stopped on a preemption request.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=pathlib.Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config override, e.g. train.lr=1e-3 (repeatable)",
    )
    parser.add_argument("--env-file", type=pathlib.Path, default=None)
    parser.add_argument(
        "--run-name", required=True, help="run directory name under the runs dir"
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH|auto",
        help="resume file to continue from; 'auto' uses <run_dir>/resume.pt if "
        "present and starts fresh otherwise",
    )
    args = parser.parse_args(argv)

    cfg = config.load_config(args.config, args.overrides)
    env = paths.load_paths(args.env_file)
    run_dir = env.runs_dir / args.run_name
    used = (run_dir / _RESOLVED).exists()
    if args.resume is None and used:
        parser.error(f"{run_dir} was already used by a run; pick a new run name")
    if used and not resume.same_run(config.load_config(run_dir / _RESOLVED), cfg):
        parser.error(f"{run_dir} holds a run with a different config")

    state: resume.ResumeState | None = None
    if args.resume is not None:
        auto = args.resume == AUTO
        path = run_dir / resume.RESUME_FILE if auto else pathlib.Path(args.resume)
        if path.exists():
            state = resume.load(path)
        elif not auto:
            parser.error(f"resume file {path} does not exist")
        elif (run_dir / loop.LAST_CHECKPOINT).exists():
            # Checkpoints without a resume file: not a run this app can continue.
            parser.error(f"{run_dir} has checkpoints but no {resume.RESUME_FILE}")
    if state is not None and state.complete:
        print(f"{run_dir} already finished ({state.global_step} steps); nothing to do")
        return 0

    config.save_resolved(cfg, run_dir)
    flag = resume.PreemptionFlag()
    flag.install()
    try:
        result = loop.train(
            cfg,
            env.data_dir,
            run_dir,
            resume=state,
            reuse_run_dir=args.resume is not None,
            preempt=flag,
        )
    finally:
        flag.uninstall()
    if result.preempted:
        print(f"stopped at step {result.steps} for requeue -> {result.run_dir}")
        return resume.REQUEUE_EXIT_CODE
    print(f"done: {result.steps} steps, {result.epochs} epochs -> {result.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
