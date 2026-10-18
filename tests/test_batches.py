"""uint8 batch gathering for device-side augmentation."""

import numpy as np
import pytest
import torch

from lejepa_norm.data import batches


def _arrays() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(n, 4, 4, 3), dtype=np.uint8) for n in (5, 3, 4)]


def test_array_concat_gathers_like_a_concatenation_without_copying() -> None:
    arrays = _arrays()
    concat = batches.ArrayConcat(arrays)
    joined = np.concatenate(arrays)
    assert len(concat) == 12 and concat.image_shape == (4, 4, 3)
    assert all(a is b for a, b in zip(concat.arrays, arrays, strict=True))
    indices = np.array([11, 0, 6, 5, 4, 8, 2])
    np.testing.assert_array_equal(concat.gather(indices), joined[indices])
    with pytest.raises(IndexError):
        concat.gather(np.array([12]))
    with pytest.raises(ValueError):
        batches.ArrayConcat([arrays[0], arrays[1][:, :2]])


def test_prefetcher_yields_batches_in_order_and_stops_early() -> None:
    arrays = _arrays()
    concat = batches.ArrayConcat(arrays)
    joined = np.concatenate(arrays)
    order = np.random.default_rng(1).permutation(12)
    index_batches = [order[i : i + 4] for i in range(0, 12, 4)]
    with batches.Prefetcher(concat, index_batches, depth=1) as prefetcher:
        got = list(prefetcher)
    assert len(got) == 3
    for batch, indices in zip(got, index_batches, strict=True):
        assert batch.dtype == torch.uint8
        np.testing.assert_array_equal(batch.numpy(), joined[indices])

    # Leaving the loop early stops the thread even though batches remain.
    endless = (order[:4] for _ in iter(int, 1))
    with batches.Prefetcher(concat, endless, depth=1) as prefetcher:
        next(iter(prefetcher))
    assert not prefetcher._thread.is_alive()  # pyright: ignore[reportPrivateUsage]

    # Errors in the gather thread surface in the consumer.
    with batches.Prefetcher(concat, [np.array([99])]) as prefetcher:
        with pytest.raises(IndexError):
            list(prefetcher)


def test_contiguous_batches_cover_the_range() -> None:
    images = np.concatenate(_arrays())
    got = list(batches.contiguous_batches(images, 5, torch.device("cpu"), 1, 12))
    assert [len(b) for b in got] == [5, 5, 1]
    np.testing.assert_array_equal(torch.cat(got).numpy(), images[1:12])
