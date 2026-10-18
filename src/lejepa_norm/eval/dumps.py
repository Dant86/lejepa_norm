"""On-disk layout of embedding dumps under `LEJEPA_NORM_EMBED_DIR`.

`apps/embed.py` writes, and `apps/analyze.py` reads, one directory per run and
evaluation set; each directory holds any number of `.npz` shards
(`shards.save_shard`), concatenated in sorted filename order on load:

    <embed_dir>/<run_name>/<eval_set>/*.npz

Evaluation sets (see `EVAL_SETS`): `stl10_train` (labeled bank for probes and
k-NN), `stl10_test` (clean test images), `stl10c` (STL-10 `test` under
corruptions; `image_id` indexes `stl10_test`, `corruption`/`severity` set per
image), `dtd_test` and `svhn_test` (OOD sets). Sets that are missing on disk
are skipped by the analyses that need them.
"""

import os
import pathlib
import typing

import numpy
import numpy.typing

from lejepa_norm.eval import shards

STL10_TRAIN = "stl10_train"
STL10_TEST = "stl10_test"
STL10_C = "stl10c"
DTD_TEST = "dtd_test"
SVHN_TEST = "svhn_test"
EVAL_SETS = (STL10_TRAIN, STL10_TEST, STL10_C, DTD_TEST, SVHN_TEST)
OOD_SETS = (DTD_TEST, SVHN_TEST)


def set_dir(
    embed_dir: str | os.PathLike[str], run_name: str, eval_set: str
) -> pathlib.Path:
    """Directory holding the shards of `eval_set` for `run_name`."""
    return pathlib.Path(embed_dir) / run_name / eval_set


def concatenate(parts: list[shards.EmbeddingShard]) -> shards.EmbeddingShard:
    """Concatenates shards along the image axis.

    Raises:
        ValueError: `parts` is empty.
    """
    if not parts:
        raise ValueError("nothing to concatenate")
    return shards.EmbeddingShard(
        z=numpy.concatenate([p.z for p in parts]),
        h=numpy.concatenate([p.h for p in parts]),
        image_id=numpy.concatenate([p.image_id for p in parts]),
        label=numpy.concatenate([p.label for p in parts]),
        corruption=numpy.concatenate([p.corruption for p in parts]),
        severity=numpy.concatenate([p.severity for p in parts]),
    )


def take(
    shard: shards.EmbeddingShard, index: numpy.typing.NDArray[numpy.integer[typing.Any]]
) -> shards.EmbeddingShard:
    """Selects the images at `index` (copies)."""
    return shards.EmbeddingShard(
        z=shard.z[index],
        h=shard.h[index],
        image_id=shard.image_id[index],
        label=shard.label[index],
        corruption=shard.corruption[index],
        severity=shard.severity[index],
    )


def load_set(directory: str | os.PathLike[str]) -> shards.EmbeddingShard | None:
    """Loads and concatenates all shards in `directory`; None if there are none."""
    files = sorted(pathlib.Path(directory).glob("*.npz"))
    if not files:
        return None
    return concatenate([shards.load_shard(f) for f in files])


def load_run(
    embed_dir: str | os.PathLike[str], run_name: str
) -> dict[str, shards.EmbeddingShard]:
    """Loads every evaluation set present for `run_name`, keyed by set name."""
    loaded: dict[str, shards.EmbeddingShard] = {}
    for name in EVAL_SETS:
        shard = load_set(set_dir(embed_dir, run_name, name))
        if shard is not None:
            loaded[name] = shard
    return loaded
