import pathlib

import numpy
import pytest
import torch

from lejepa_norm.eval import shards
from lejepa_norm.losses import batch


def test_view_batch_splits_global_and_local() -> None:
    z = torch.randn(6, 4, 8)
    is_global = torch.tensor([True, True, False, False, False, False])
    vb = batch.ViewBatch(z=z, is_global=is_global)
    assert vb.global_z.shape == (2, 4, 8)
    assert vb.local_z.shape == (4, 4, 8)
    with pytest.raises(ValueError):
        batch.ViewBatch(z=z, is_global=torch.zeros(6, dtype=torch.bool))


def test_shard_round_trip(tmp_path: pathlib.Path) -> None:
    n, views = 5, 9
    shard = shards.EmbeddingShard(
        z=numpy.random.rand(n, views, 4).astype(numpy.float32),
        h=numpy.random.rand(n, views, 7).astype(numpy.float32),
        image_id=numpy.arange(n, dtype=numpy.int64),
        label=numpy.arange(n, dtype=numpy.int64) % 3,
        corruption=numpy.array(["", "contrast", "", "pixelate", ""]),
        severity=numpy.array([0, 3, 0, 5, 0], dtype=numpy.int8),
    )
    path = tmp_path / "shard.npz"
    shards.save_shard(path, shard)
    loaded = shards.load_shard(path)
    numpy.testing.assert_array_equal(loaded.z, shard.z)
    numpy.testing.assert_array_equal(loaded.corruption, shard.corruption)
    assert loaded.severity.dtype == numpy.int8
