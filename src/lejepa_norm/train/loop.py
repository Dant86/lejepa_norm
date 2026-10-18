"""Single-device pretraining loop (Variant A; variant-agnostic via `ViewBatch`).

Data: the STL-10 `unlabeled` + `train` uint8 arrays stay in RAM; each step's
images are gathered on the CPU by a background thread (`data.batches`), copied
to the device (pinned, non-blocking) and augmented there by
`data.gpu_views.MultiViewAugment` with a dedicated device `torch.Generator`
seeded from `seed`. No DataLoader workers are used, so `data.num_workers` does not
affect training (results do not depend on the CPU count).

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

Preemption and resume (`train.resume`): `resume.pt` is written at the end of every
epoch and, when the `PreemptionFlag` is set (SIGUSR1), at the next step boundary,
after which `train` returns with `preempted=True`. A resumed run continues at the
saved global step: the LR schedule is a function of the global step, the data
order of each epoch is a permutation seeded by `(seed, epoch)` that is skipped
forward to the interrupted step, and the states of the augmentation generator and
of torch's global RNG (SIGReg directions) are restored, so a resumed run sees the
same views and directions as an uninterrupted one (bit-identical results on CPU;
on CUDA, non-deterministic kernels can still differ at float precision).
"""

import dataclasses
import math
import os
import pathlib
import random
import time
import typing
import zlib

import numpy
import torch
from torch import nn

from lejepa_norm import config
from lejepa_norm.data import batches
from lejepa_norm.data import datasets
from lejepa_norm.data import gpu_views
from lejepa_norm.losses import batch as batch_lib
from lejepa_norm.losses import diagnostics
from lejepa_norm.losses import lejepa
from lejepa_norm.models import backbones
from lejepa_norm.models import checkpoint
from lejepa_norm.models import projector as projector_lib
from lejepa_norm.train import metrics_log
from lejepa_norm.train import online_knn
from lejepa_norm.train import resume as resume_lib
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
        preempted: Training stopped early on a preemption request after saving
            `resume.pt`; rerun with the resume state to continue.
    """

    run_dir: pathlib.Path
    last_checkpoint: pathlib.Path
    steps: int
    epochs: int
    preempted: bool = False


def epoch_permutation(seed: int, epoch: int, size: int) -> numpy.ndarray:
    """The image order of `epoch`: a permutation seeded by `(seed, epoch)`.

    It does not depend on how many epochs ran in this process, so a resumed run
    sees the same order.
    """
    key = zlib.crc32(f"{seed}/epoch/{epoch}".encode())
    generator = torch.Generator().manual_seed(key)
    return torch.randperm(size, generator=generator).numpy()


def augmentation_generator(
    seed: int, device: torch.device, global_step: int = 0
) -> torch.Generator:
    """The device generator that drives all training augmentation.

    `global_step` > 0 gives an independent stream for resuming from a state that
    has no saved generator state.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(zlib.crc32(f"{seed}/augment/{global_step}".encode()))
    return generator


def checkpoint_name(epoch: int) -> str:
    """File name of the checkpoint written after `epoch` completed epochs."""
    return f"epoch_{epoch:04d}.pt"


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and torch (all devices)."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


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
    *,
    resume: resume_lib.ResumeState | None = None,
    reuse_run_dir: bool = False,
    preempt: resume_lib.PreemptionFlag | None = None,
) -> TrainResult:
    """Pretrains a backbone + projector on STL-10 `unlabeled` + `train`.

    Args:
        cfg: Fully resolved experiment config.
        data_dir: Prepared data root (`LEJEPA_NORM_DATA_DIR`).
        run_dir: Output directory for `metrics.jsonl` and checkpoints.
        device: Defaults to CUDA if available, else CPU.
        resume: Continue from this state (`metrics.jsonl` is truncated to its
            global step). Must not be a completed state.
        reuse_run_dir: Without `resume`, start from scratch even if `run_dir`
            holds a log from an attempt that never saved a resume state (the log
            is cleared).
        preempt: When set (by SIGUSR1), save `resume.pt` at the next step
            boundary and return with `preempted=True`.

    Returns:
        Summary with the path of the last checkpoint.

    Raises:
        FileExistsError: `run_dir` already holds a `metrics.jsonl` and neither
            `resume` nor `reuse_run_dir` is given.
        ValueError: `train.epochs < 1`, the dataset is smaller than one batch, or
            `resume` is complete, has a different config or dataset size.
        FloatingPointError: The loss became non-finite.
    """
    if cfg.train.epochs < 1:
        raise ValueError(f"train.epochs must be >= 1, got {cfg.train.epochs}")
    if resume is not None:
        if resume.complete:
            raise ValueError("the resume state marks the run as complete")
        if not resume_lib.same_run(config.from_dict(resume.config), cfg):
            raise ValueError("the resume state was saved with a different config")
    run_path = pathlib.Path(run_dir)
    log = metrics_log.MetricsLog(run_path)
    if resume is None and not reuse_run_dir and log.path.exists():
        raise FileExistsError(f"{log.path} exists; use a new run name")
    preempt = preempt or resume_lib.PreemptionFlag()
    resume_path = run_path / resume_lib.RESUME_FILE
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(cfg.seed)
    tc = cfg.train
    autocast = tc.precision == "bf16"

    images = datasets.load_pretraining_images(data_dir)
    augment = gpu_views.MultiViewAugment(cfg.data)
    aug_generator = augmentation_generator(cfg.seed, device)
    steps_per_epoch = len(images) // tc.batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            f"{len(images)} pretraining images < batch size {tc.batch_size}"
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
    start_epoch, skip = 1, 0
    if resume is not None:
        expected = resume.epoch * steps_per_epoch + resume.step_in_epoch
        if resume.global_step != expected or resume.step_in_epoch >= steps_per_epoch:
            raise ValueError("the resume state does not match this dataset size")
        assert resume.backbone is not None and resume.projector is not None
        assert resume.optimizer is not None
        backbone.load_state_dict(resume.backbone)
        projector.load_state_dict(resume.projector)
        optimizer.load_state_dict(resume.optimizer)
        step, epoch = resume.global_step, resume.epoch
        start_epoch, skip = resume.epoch + 1, resume.step_in_epoch
        log.truncate(step)
        if resume.rng_cpu is not None:
            torch.set_rng_state(resume.rng_cpu)
        if resume.rng_cuda and device.type == "cuda":
            torch.cuda.set_rng_state_all(resume.rng_cuda)
        if resume.rng_augment is not None:
            aug_generator.set_state(resume.rng_augment)
        else:  # resume state written before GPU augmentation existed
            aug_generator = augmentation_generator(cfg.seed, device, step)
            print("resume state has no augmentation RNG state; reseeding it")
        print(f"resuming after step {step} (epoch {start_epoch}, step {skip} of it)")
    elif reuse_run_dir:
        log.truncate(0)

    def snapshot(
        epoch_done: int, in_epoch: int, epoch_sum: torch.Tensor | None, elapsed: float
    ) -> resume_lib.ResumeState:
        return resume_lib.ResumeState(
            config=config.to_dict(cfg),
            global_step=step,
            epoch=epoch_done,
            step_in_epoch=in_epoch,
            backbone=backbone.state_dict(),
            projector=projector.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch_sum=None if epoch_sum is None else epoch_sum.cpu(),
            epoch_elapsed_s=elapsed,
            rng_cpu=torch.get_rng_state(),
            rng_cuda=_cuda_rng_states(),
            rng_augment=aug_generator.get_state(),
        )

    last = run_path / LAST_CHECKPOINT
    for epoch in range(start_epoch, tc.epochs + 1):
        epoch_start = interval_start = time.perf_counter()
        interval_steps = 0
        epoch_sum = torch.zeros(len(STAT_KEYS), device=device)
        elapsed_before = 0.0
        epoch_step = 0
        if resume is not None and epoch == start_epoch and skip > 0:
            epoch_step = skip
            elapsed_before = resume.epoch_elapsed_s
            if resume.epoch_sum is not None:
                epoch_sum += resume.epoch_sum.to(device)
        order = epoch_permutation(cfg.seed, epoch, len(images))
        index_batches = (
            order[s * tc.batch_size : (s + 1) * tc.batch_size]
            for s in range(epoch_step, steps_per_epoch)
        )
        prefetcher = batches.Prefetcher(
            images, index_batches, pin_memory=device.type == "cuda"
        )
        with prefetcher:
            for cpu_batch in prefetcher:
                lr = schedule.lr_at(step, peak_lr, warmup_steps, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                uint8_batch = cpu_batch.to(device, non_blocking=True)
                view_list = augment(uint8_batch, aug_generator)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=autocast
                ):
                    h, z = forward_views(backbone, projector, view_list, num_global)
                loss, loss_stats = loss_fn(batch_lib.ViewBatch(z.float(), is_global))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # pyright: ignore[reportUnknownMemberType]
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                step += 1
                epoch_step += 1
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
                if preempt.requested and epoch_step < steps_per_epoch:
                    elapsed = elapsed_before + time.perf_counter() - epoch_start
                    state = snapshot(epoch - 1, epoch_step, epoch_sum, elapsed)
                    resume_lib.save(resume_path, state)
                    print(f"preemption requested: saved {resume_path} at step {step}")
                    return TrainResult(run_path, last, step, epoch - 1, preempted=True)
        epoch_time = elapsed_before + time.perf_counter() - epoch_start
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
        if is_last:
            break
        resume_lib.save(resume_path, snapshot(epoch, 0, None, 0.0))
        if preempt.requested:
            print(f"preemption requested: saved {resume_path} at step {step}")
            return TrainResult(run_path, last, step, epoch, preempted=True)
    # The weights live in last.pt; the marker only records that the run is done.
    done = resume_lib.ResumeState(
        config.to_dict(cfg), step, epoch, step_in_epoch=0, complete=True
    )
    resume_lib.save(resume_path, done)
    return TrainResult(run_path, last, step, epoch)


def _cuda_rng_states() -> list[torch.Tensor]:
    if not torch.cuda.is_available():
        return []
    return list(torch.cuda.get_rng_state_all())
