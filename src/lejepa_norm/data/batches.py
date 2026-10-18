"""uint8 image batches for the device-side augmentation (`data.gpu_views`).

The CPU only gathers raw uint8 images from the in-RAM arrays (numpy fancy
indexing) into a pinned buffer; the device copies them asynchronously and does
all augmentation. No DataLoader workers are involved, so nothing about the data
depends on the CPU count.

* `ArrayConcat` indexes several image arrays as one without concatenating them
  (STL-10 `unlabeled` + `train` is 2.9 GB; a copy is not acceptable).
* `Prefetcher` gathers the next batches in a background thread while the device
  works on the current one.
* `contiguous_batches` streams consecutive slices of one array (evaluation).
"""

import collections.abc
import queue
import threading
import types

import numpy as np
import torch


class ArrayConcat:
    """Several uint8 `(N_k, H, W, 3)` arrays indexed as one, without copying."""

    def __init__(self, arrays: collections.abc.Sequence[np.ndarray]) -> None:
        """Wraps `arrays` (all with the same image shape and dtype)."""
        if not arrays:
            raise ValueError("no arrays")
        shapes = {(a.shape[1:], a.dtype) for a in arrays}
        if len(shapes) != 1:
            raise ValueError(f"arrays differ in image shape or dtype: {shapes}")
        self.arrays = list(arrays)
        self._offsets = np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self) -> int:
        """Total number of images."""
        return int(self._offsets[-1])

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Shape of one image, e.g. `(96, 96, 3)`."""
        return tuple(self.arrays[0].shape[1:])

    def gather(self, indices: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Copies the images at global `indices` (in that order) into `out`.

        Args:
            indices: int array of global indices in `[0, len(self))`.
            out: Destination `(len(indices), *image_shape)`; allocated if None.

        Returns:
            `out`.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= len(self)):
            raise IndexError(f"indices out of range [0, {len(self)})")
        if out is None:
            out = np.empty(
                (len(indices), *self.image_shape), dtype=self.arrays[0].dtype
            )
        for array, start in zip(self.arrays, self._offsets[:-1], strict=True):
            mask = (indices >= start) & (indices < start + len(array))
            if mask.any():
                out[mask] = array[indices[mask] - start]
        return out


def _gather_tensor(
    source: ArrayConcat, indices: np.ndarray, pin_memory: bool
) -> torch.Tensor:
    batch = torch.empty(
        (len(indices), *source.image_shape), dtype=torch.uint8, pin_memory=pin_memory
    )
    source.gather(indices, out=batch.numpy())
    return batch


_DONE = object()


class Prefetcher:
    """Iterates over uint8 batches gathered by a background thread.

    Use as a context manager so the thread stops when the loop exits early:

        with Prefetcher(images, index_batches, pin_memory=True) as batches:
            for batch in batches:  # torch.uint8 (B, H, W, 3), pinned
                batch = batch.to(device, non_blocking=True)
    """

    def __init__(
        self,
        source: ArrayConcat,
        index_batches: collections.abc.Iterable[np.ndarray],
        pin_memory: bool = False,
        depth: int = 2,
    ) -> None:
        """Starts gathering.

        Args:
            source: Images to gather from.
            index_batches: Global indices of each batch, in order.
            pin_memory: Gather into page-locked memory (for CUDA transfers).
            depth: Batches gathered ahead of the consumer.
        """
        self._queue: queue.Queue[object] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._produce,
            args=(source, iter(index_batches), pin_memory),
            daemon=True,
        )
        self._thread.start()

    def _put(self, item: object) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(
        self,
        source: ArrayConcat,
        index_batches: collections.abc.Iterator[np.ndarray],
        pin_memory: bool,
    ) -> None:
        try:
            for indices in index_batches:
                if not self._put(_gather_tensor(source, indices, pin_memory)):
                    return
        except BaseException as err:  # re-raised in the consumer
            self._put(err)
            return
        self._put(_DONE)

    def __iter__(self) -> collections.abc.Iterator[torch.Tensor]:
        """Yields the batches in order; re-raises errors of the gather thread."""
        while True:
            item = self._queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, torch.Tensor)
            yield item

    def close(self) -> None:
        """Stops the background thread."""
        self._stop.set()
        self._thread.join()

    def __enter__(self) -> "Prefetcher":
        """Returns self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Stops the background thread."""
        self.close()


def contiguous_batches(
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
    start: int = 0,
    stop: int | None = None,
) -> collections.abc.Iterator[torch.Tensor]:
    """Yields `images[start:stop]` as uint8 device tensors of `batch_size` images.

    Args:
        images: uint8 `(N, H, W, 3)` array (in RAM or memory-mapped).
        batch_size: Images per batch (the last batch may be smaller).
        device: Target device; CUDA transfers go through pinned memory.
        start: First image.
        stop: One past the last image (default: the end).
    """
    stop = len(images) if stop is None else stop
    pin = device.type == "cuda"
    for begin in range(start, stop, batch_size):
        chunk = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
            np.ascontiguousarray(images[begin : min(begin + batch_size, stop)])
        )
        if pin:
            chunk = chunk.pin_memory()
        yield chunk.to(device, non_blocking=pin)
