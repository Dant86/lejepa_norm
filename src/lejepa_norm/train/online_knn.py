"""Online k-NN monitor: cosine k-NN accuracy on STL-10 `test` during training.

The bank is STL-10 `train` and the queries are STL-10 `test`, both un-augmented
(`CenterTransform`). Features are the backbone output `h` and the projector
output `z`; classification uses `lejepa_norm.eval.downstream.knn_votes`.
"""

import os

import numpy
import numpy.typing
import torch
from torch import nn
import torch.utils.data

from lejepa_norm import config
from lejepa_norm.data import datasets
from lejepa_norm.data import views
from lejepa_norm.eval import downstream

FloatArray = numpy.typing.NDArray[numpy.float32]
IntArray = numpy.typing.NDArray[numpy.int64]


@torch.no_grad()
def extract(
    backbone: nn.Module,
    projector: nn.Module,
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    autocast: bool,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Runs the models in eval mode over `loader`.

    The modules are put back into their previous (train/eval) mode afterwards.

    Returns:
        `(h, z, labels)` as float32 / int64 arrays.
    """
    modes = (backbone.training, projector.training)
    backbone.eval()
    projector.eval()
    hs: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    try:
        for images, label in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                h = backbone(images)
                z = projector(h)
            hs.append(h.float().cpu())
            zs.append(z.float().cpu())
            labels.append(label)
    finally:
        backbone.train(modes[0])
        projector.train(modes[1])
    return (
        torch.cat(hs).numpy(),
        torch.cat(zs).numpy(),
        torch.cat(labels).numpy().astype(numpy.int64),
    )


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
            cfg: Experiment config; `data` (normalization, workers),
                `eval.batch_size`, `eval.knn_k` and `train.precision` are used.
            device: Device the models live on.
        """
        transform = views.CenterTransform(cfg.data)
        self._device = device
        self._autocast = cfg.train.precision == "bf16"
        self._k = cfg.eval.knn_k
        self._loaders = {
            split: torch.utils.data.DataLoader(
                datasets.load_split(data_dir, "stl10", split, transform),
                batch_size=cfg.eval.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=device.type == "cuda",
            )
            for split in ("train", "test")
        }

    def __call__(self, backbone: nn.Module, projector: nn.Module) -> dict[str, float]:
        """Returns `{"knn_acc_h": ..., "knn_acc_z": ...}` on STL-10 `test`."""
        bank_h, bank_z, bank_y = extract(
            backbone, projector, self._loaders["train"], self._device, self._autocast
        )
        query_h, query_z, query_y = extract(
            backbone, projector, self._loaders["test"], self._device, self._autocast
        )
        num_classes = int(max(bank_y.max(), query_y.max())) + 1
        out: dict[str, float] = {}
        for name, bank, query in (("h", bank_h, query_h), ("z", bank_z, query_z)):
            votes = downstream.knn_votes(bank, bank_y, query, self._k, num_classes)
            out[f"knn_acc_{name}"] = downstream.accuracy(votes.argmax(axis=1), query_y)
        return out
