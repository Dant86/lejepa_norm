"""Prepares datasets under `LEJEPA_NORM_DATA_DIR`.

Stages:
  download  Fetch canonical STL-10, DTD, SVHN and Places365 archives (needs
            network; run on the login node).
  build     Convert to the prepared layout and generate STL-10-C (offline,
            CPU-heavy; run under SLURM).
  stats     Compute per-channel mean/std of the pretraining images (STL-10
            unlabeled + train) from the built arrays; light, offline. Copy the
            printed values into DataConfig / configs/base.yaml.

Usage:
  python apps/prepare_data.py --stage download
  python apps/prepare_data.py --stage build --set data.num_workers=16
  python apps/prepare_data.py --stage stats
"""

import argparse
import pathlib

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.data import layout
from lejepa_norm.data import prepare
from lejepa_norm.data import stats

_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def main() -> None:
    """Parses arguments and runs the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("download", "build", "stats"), required=True
    )
    parser.add_argument("--config", type=pathlib.Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config override, e.g. data.full_corruption_set=true",
    )
    parser.add_argument("--env-file", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cfg = config.load_config(args.config, args.overrides)
    data_dir = paths.load_paths(args.env_file).data_dir
    if args.stage == "download":
        prepare.download(data_dir)
    elif args.stage == "build":
        prepare.build(data_dir, cfg.data)
        config.save_resolved(cfg, data_dir / layout.STL10_C_DIR)
    else:
        mean, std = stats.channel_stats(data_dir)
        out = stats.write_channel_stats(data_dir, mean, std)
        print(f"mean={mean}\nstd={std}\nwrote {out}")
        print(f"override: {stats.override_args(mean, std)}")


if __name__ == "__main__":
    main()
