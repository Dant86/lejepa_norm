"""Dumps multi-view embeddings of a checkpoint on the evaluation sets.

For each evaluation set, every image is embedded un-augmented (view 0) and under
`eval.num_views` random global-view augmentations (views 1..K); projector outputs
`z` and backbone features `h` are written as `.npz` shards to
`LEJEPA_NORM_EMBED_DIR/<run-name>/<eval_set>/` (layout in `lejepa_norm.eval.dumps`).

Existing shards are an error unless `--overwrite` (replace them) or `--resume`
(keep them and write only the missing ones, e.g. after preemption) is given.

Example:
    python apps/embed.py --checkpoint runs/a_l0.05_s0/last.pt \
        --run-name a_l0.05_s0 --set eval.num_views=8
"""

import argparse
import collections.abc
import pathlib
import sys

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import dumps
from lejepa_norm.eval import embedding


def _parse(argv: collections.abc.Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dumps multi-view embeddings of a checkpoint."
    )
    parser.add_argument(
        "--config", type=pathlib.Path, default=pathlib.Path("configs/base.yaml")
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--env-file", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--sets",
        default=",".join(dumps.EVAL_SETS),
        help="Comma-separated subset of: " + ", ".join(dumps.EVAL_SETS),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Debug cap on images per image array (per corruption/severity).",
    )
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing shards instead of failing.",
    )
    existing.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing shards and write only the missing ones (continue an "
        "interrupted job with the same checkpoint and arguments).",
    )
    return parser.parse_args(argv)


def main(argv: collections.abc.Sequence[str]) -> int:
    """Runs the embedding app; returns a process exit code."""
    args = _parse(argv)
    cfg = config.load_config(args.config, args.overrides)
    locations = paths.load_paths(args.env_file)
    counts = embedding.embed_run(
        args.checkpoint,
        cfg,
        locations.data_dir,
        locations.embed_dir,
        args.run_name,
        eval_sets=[s.strip() for s in args.sets.split(",") if s.strip()],
        max_images=args.max_images,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    for name, count in counts.items():
        print(f"{name}: {count} images")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
