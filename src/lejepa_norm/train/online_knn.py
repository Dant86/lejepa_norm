"""Online k-NN monitor: cosine k-NN accuracy on STL-10 `test` during training.

The bank is STL-10 `train` and the queries are STL-10 `test`, both un-augmented
(`data.gpu_views.center_view`, computed on the device from uint8 batches).
Features are the backbone output `h` and the projector output `z`;
classification uses `lejepa_norm.eval.downstream.knn_votes`.
"""

import os

import numpy
import numpy.typing
import torch
from torch import nn

from lejepa_norm import config
from lejepa_norm.data import batches
from lejepa_norm.data import datasets
from lejepa_norm.data import gpu_views
from lejepa_norm.eval import downstream

FloatArray = numpy.typing.NDArray[numpy.float32]
IntArray = numpy.typing.NDArray[numpy.int64]


@torch.no_grad()
def extract(
    backbone: nn.Module,
    projector: nn.Module,
    images: numpy.ndarray,
    cfg: config.ExperimentConfig,
    device: torch.device,
) -> tuple[FloatArray, FloatArray]:
    """Runs the models in eval mode over the center views of uint8 `images`.

    The modules are put back into their previous (train/eval) mode afterwards.

    Args:
        backbone: Backbone on `device`.
        projector: Projector on `device`.
        images: uint8 `(N, H, W, 3)`.
        cfg: `data` (normalization), `eval.batch_size` and `train.precision` are
            used.
        device: Device the models live on.

    Returns:
        `(h, z)` as float32 arrays.
    """
    autocast = cfg.train.precision == "bf16"
    modes = (backbone.training, projector.training)
    backbone.eval()
    projector.eval()
    hs: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    try:
        for batch in batches.contiguous_batches(images, cfg.eval.batch_size, device):
            view = gpu_views.center_view(batch, cfg.data)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                h = backbone(view)
                z = projector(h)
            hs.append(h.float().cpu())
            zs.append(z.float().cpu())
    finally:
        backbone.train(modes[0])
        projector.train(modes[1])
    return torch.cat(hs).numpy(), torch.cat(zs).numpy()


class OnlineKnn:
    """Cosine k-NN accuracy of `h` and `z` on STL-10 `test` (bank: `train`)."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        cfg: config.ExperimentConfig,
        device: torch.device,
    ) -> None:
        """Opens the bank and query splits.

        Args:
            data_dir: Prepared data root (`LEJEPA_NORM_DATA_DIR`).
            cfg: Experiment config; `data` (normalization), `eval.batch_size`,
                `eval.knn_k` and `train.precision` are used.
            device: Device the models live on.
        """
        self._cfg = cfg
        self._device = device
        self._k = cfg.eval.knn_k
        self._splits: dict[str, datasets.ImageArrayDataset[object]] = {
            split: datasets.load_split(data_dir, "stl10", split)
            for split in ("train", "test")
        }

    def __call__(self, backbone: nn.Module, projector: nn.Module) -> dict[str, float]:
        """Returns `{"knn_acc_h": ..., "knn_acc_z": ...}` on STL-10 `test`."""
        bank_set, query_set = self._splits["train"], self._splits["test"]
        bank_h, bank_z = extract(
            backbone, projector, bank_set.images, self._cfg, self._device
        )
        query_h, query_z = extract(
            backbone, projector, query_set.images, self._cfg, self._device
        )
        bank_y = bank_set.labels.astype(numpy.int64)
        query_y = query_set.labels.astype(numpy.int64)
        num_classes = int(max(bank_y.max(), query_y.max())) + 1
        out: dict[str, float] = {}
        for name, bank, query in (("h", bank_h, query_h), ("z", bank_z, query_z)):
            votes = downstream.knn_votes(bank, bank_y, query, self._k, num_classes)
            out[f"knn_acc_{name}"] = downstream.accuracy(votes.argmax(axis=1), query_y)
        return out
