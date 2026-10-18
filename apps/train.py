"""Pretrains one model from a config; outputs go to `<runs_dir>/<run-name>/`.

Writes `config.resolved.yaml`, `metrics.jsonl` and checkpoints (`epoch_XXXX.pt`,
`last.pt`) into the run directory under `LEJEPA_NORM_RUNS_DIR`. Data is read from
`LEJEPA_NORM_DATA_DIR`. No network access is needed.

Usage:
  python apps/train.py --run-name a_l0.05_s0 --set loss.lambda=0.05 --set seed=0
"""

import argparse
import collections.abc
import pathlib

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.train import loop

_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def main(argv: collections.abc.Sequence[str] | None = None) -> None:
    """Parses arguments, saves the resolved config and runs training."""
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
    args = parser.parse_args(argv)

    cfg = config.load_config(args.config, args.overrides)
    env = paths.load_paths(args.env_file)
    run_dir = env.runs_dir / args.run_name
    if (run_dir / "config.resolved.yaml").exists():
        parser.error(f"{run_dir} was already used by a run; pick a new run name")
    config.save_resolved(cfg, run_dir)
    result = loop.train(cfg, env.data_dir, run_dir)
    print(f"done: {result.steps} steps, {result.epochs} epochs -> {result.run_dir}")


if __name__ == "__main__":
    main()
