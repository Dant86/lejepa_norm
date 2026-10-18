"""Single-device pretraining loop (Variant A; variant-agnostic via `ViewBatch`).

Per step: the global views (all the same resolution) go through the backbone in
one forward pass and the local views in another; the backbone features `h` of all
views are concatenated and passed through the projector together, and the
projector outputs are reshaped to a `ViewBatch` (`z`: views x batch x d, global
views first). The model runs under bf16 autocast when `train.precision == "bf16"`;
the loss is computed in fp32 outside the autocast region.

Optimization: AdamW with peak learning rate `train.lr * batch_size / 256`, linear
warmup over `warmup_epochs`, then cosine decay to 0 (updated every step). Weight
decay applies to weight matrices and conv kernels only, not to biases or
normalization parameters.

Outputs in `run_dir`: `metrics.jsonl` (see `metrics_log`) and checkpoints in the
`models.checkpoint` format: `epoch_XXXX.pt` every `checkpoint_every_epochs` and at
the end, with `last.pt` always holding the most recent one.
"""

import dataclasses
import math
import os
import pathlib
import random
import time
import typing

import numpy
import torch
from torch import nn
import torch.utils.data

from lejepa_norm import config
from lejepa_norm.data import datasets
from lejepa_norm.data import views
from lejepa_norm.losses import batch as batch_lib
from lejepa_norm.losses import diagnostics
from lejepa_norm.losses import lejepa
from lejepa_norm.models import backbones
from lejepa_norm.models import checkpoint
from lejepa_norm.models import projector as projector_lib
from lejepa_norm.train import metrics_log
from lejepa_norm.train import online_knn
from lejepa_norm.train import schedule

LAST_CHECKPOINT = "last.pt"
# Scalars tracked every step, in this order (see `_step_stats`).
STAT_KEYS = (
    "loss_total",
    "loss_inv",
    "loss_dist",
    "kappa_z",
    "kappa_h",
    "z_norm_mean",
    "z_norm_std",
    "h_norm_mean",
    "h_norm_std",
)


@dataclasses.dataclass(frozen=True)
class TrainResult:
    """Summary of a finished run.

    Attributes:
        run_dir: Directory holding the run's outputs.
        last_checkpoint: Path of `last.pt`.
        steps: Optimizer steps taken.
        epochs: Epochs completed.
    """

    run_dir: pathlib.Path
    last_checkpoint: pathlib.Path
    steps: int
    epochs: int


def checkpoint_name(epoch: int) -> str:
    """File name of the checkpoint written after `epoch` completed epochs."""
    return f"epoch_{epoch:04d}.pt"


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and torch (all devices)."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


def _seed_worker(worker_id: int) -> None:
    # torch seeds each worker from the loader generator; derive the others from it.
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def _param_groups(
    modules: list[nn.Module], weight_decay: float
) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                (decay if param.ndim > 1 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _norm_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norms = x.float().square().sum(dim=-1).sqrt()
    return norms.mean(), norms.std()


def forward_views(
    backbone: nn.Module,
    projector: nn.Module,
    view_list: list[torch.Tensor],
    num_global: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embeds all views of a batch: one backbone pass per resolution.

    Args:
        backbone: Maps `(B, 3, H, W)` to `(B, feat_dim)`.
        projector: Maps `(B, feat_dim)` to `(B, d)`.
        view_list: One `(batch, 3, H, W)` tensor per view, global views first.
        num_global: Number of leading global views (all of the same size).

    Returns:
        `h` of shape `(views * batch, feat_dim)` (view-major) and `z` of shape
        `(views, batch, d)`.
    """
    batch_size = view_list[0].shape[0]
    groups = [view_list[:num_global], view_list[num_global:]]
    h = torch.cat([backbone(torch.cat(group)) for group in groups if group])
    z = projector(h)
    return h, z.reshape(len(view_list), batch_size, -1)


def _step_stats(
    h: torch.Tensor, z: torch.Tensor, loss_stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Stacks the `STAT_KEYS` scalars of one step into a detached fp32 vector."""
    flat_z = z.detach().float().reshape(-1, z.shape[-1])
    flat_h = h.detach().float()
    z_mean, z_std = _norm_stats(flat_z)
    h_mean, h_std = _norm_stats(flat_h)
    return torch.stack(
        [
            loss_stats["total"].float(),
            loss_stats["inv"].float(),
            loss_stats["dist"].float(),
            diagnostics.kappa(flat_z),
            diagnostics.kappa(flat_h),
            z_mean,
            z_std,
            h_mean,
            h_std,
        ]
    )


def _as_record(values: torch.Tensor, suffix: str = "") -> dict[str, float]:
    floats = typing.cast(list[float], values.tolist())  # pyright: ignore[reportUnknownMemberType]
    items = zip(STAT_KEYS, floats, strict=True)
    return {f"{key}{suffix}": float(value) for key, value in items}


def train(
    cfg: config.ExperimentConfig,
    data_dir: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    device: torch.device | None = None,
) -> TrainResult:
    """Pretrains a backbone + projector on STL-10 `unlabeled` + `train`.

    Args:
        cfg: Fully resolved experiment config.
        data_dir: Prepared data root (`LEJEPA_NORM_DATA_DIR`).
        run_dir: Output directory for `metrics.jsonl` and checkpoints.
        device: Defaults to CUDA if available, else CPU.

    Returns:
        Summary with the path of the last checkpoint.

    Raises:
        FileExistsError: `run_dir` already holds a `metrics.jsonl` (resuming is
            not supported; pick a new run name).
        ValueError: `train.epochs < 1` or the dataset is smaller than one batch.
        FloatingPointError: The loss became non-finite.
    """
    if cfg.train.epochs < 1:
        raise ValueError(f"train.epochs must be >= 1, got {cfg.train.epochs}")
    run_path = pathlib.Path(run_dir)
    log = metrics_log.MetricsLog(run_path)
    if log.path.exists():
        raise FileExistsError(f"{log.path} exists; use a new run name")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(cfg.seed)
    tc = cfg.train
    autocast = tc.precision == "bf16"

    dataset = datasets.load_pretraining(data_dir, views.MultiViewTransform(cfg.data))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=tc.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.data.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError(
            f"{len(dataset)} pretraining images < batch size {tc.batch_size}"
        )
    total_steps = tc.epochs * steps_per_epoch
    warmup_steps = min(tc.warmup_epochs * steps_per_epoch, total_steps)
    peak_lr = schedule.scaled_lr(tc.lr, tc.batch_size)

    backbone, feat_dim = backbones.build_backbone(cfg.model)
    projector = projector_lib.build_projector(feat_dim, cfg.model)
    backbone.to(device).train()
    projector.to(device).train()
    optimizer = torch.optim.AdamW(
        _param_groups([backbone, projector], tc.weight_decay), lr=peak_lr
    )
    loss_fn = lejepa.build_loss(cfg.loss)
    knn = online_knn.OnlineKnn(data_dir, cfg, device)
    num_global = cfg.data.num_global_views
    num_views = num_global + cfg.data.num_local_views
    is_global = torch.arange(num_views, device=device) < num_global

    step = 0
    epoch = 0
    last = run_path / LAST_CHECKPOINT
    for epoch in range(1, tc.epochs + 1):
        epoch_start = interval_start = time.perf_counter()
        interval_steps = 0
        epoch_sum = torch.zeros(len(STAT_KEYS), device=device)
        for view_list, _ in loader:
            lr = schedule.lr_at(step, peak_lr, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            view_list = [v.to(device, non_blocking=True) for v in view_list]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                h, z = forward_views(backbone, projector, view_list, num_global)
            loss, loss_stats = loss_fn(batch_lib.ViewBatch(z.float(), is_global))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            step += 1
            interval_steps += 1
            stats = _step_stats(h, z, loss_stats)
            epoch_sum += stats
            if step == 1 or step % tc.log_every_steps == 0:
                record = _as_record(stats)  # syncs with the device
                if not math.isfinite(record["loss_total"]):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                now = time.perf_counter()
                log.write(
                    {
                        "kind": "step",
                        "step": step,
                        "epoch": epoch,
                        "lr": lr,
                        **record,
                        "images_per_s": interval_steps
                        * tc.batch_size
                        / (now - interval_start),
                    }
                )
                interval_start, interval_steps = now, 0
        epoch_time = time.perf_counter() - epoch_start
        is_last = epoch == tc.epochs
        epoch_record: dict[str, object] = {
            "kind": "epoch",
            "step": step,
            "epoch": epoch,
            **_as_record(epoch_sum / steps_per_epoch, "_epoch_mean"),
            "epoch_time_s": epoch_time,
            "images_per_s": steps_per_epoch * tc.batch_size / epoch_time,
        }
        if epoch % tc.knn_every_epochs == 0 or is_last:
            epoch_record.update(knn(backbone, projector))
        log.write(epoch_record)
        if epoch % tc.checkpoint_every_epochs == 0 or is_last:
            path = run_path / checkpoint_name(epoch)
            checkpoint.save(path, backbone, projector, cfg, epoch)
            checkpoint.save(last, backbone, projector, cfg, epoch)
    return TrainResult(run_path, last, step, epoch)
