"""Preemption-safe training: the resume file and the SIGUSR1 flag.

On the cluster's preemptable QoS, Slurm sends SIGUSR1 to the job (forwarded to
Python by `run_preemptable` in `slurm/common.sh`), waits a grace period, kills the
job and requeues it. Training reacts as follows:

* `PreemptionFlag.install` registers a SIGUSR1 handler that only sets a flag.
* The loop checks the flag after every optimizer step; when it is set, the loop
  writes `<run_dir>/resume.pt` atomically and the app exits with
  `REQUEUE_EXIT_CODE` (99), which the cluster treats as "requeue me".
* `resume.pt` is also rewritten at the end of every epoch, and replaced by a
  small "complete" marker (no weights) when the run finishes.

`resume.pt` is separate from the model checkpoints (`models.checkpoint`), whose
format is unchanged. It is a `torch.save` dict loadable with `weights_only=True`
(see `ResumeState` for the fields).
"""

import dataclasses
import os
import pathlib
import signal
import types
import typing

import torch

from lejepa_norm import config

RESUME_FILE = "resume.pt"
REQUEUE_EXIT_CODE = 99
FORMAT_VERSION = 1
# Fields that may differ between the original launch and a resumed one
# (`data.num_workers` does not affect training at all).
_RUNTIME_ONLY = (("data", "num_workers"),)


@dataclasses.dataclass
class ResumeState:
    """Everything needed to continue a training run where it stopped.

    Attributes:
        config: `config.to_dict` of the run's config.
        global_step: Optimizer steps completed.
        epoch: Epochs fully completed.
        step_in_epoch: Steps completed within epoch `epoch + 1` (0 at an epoch
            boundary).
        complete: The run finished; nothing else needs to be restored.
        backbone: Backbone state dict (None once complete).
        projector: Projector state dict (None once complete).
        optimizer: Optimizer state dict (None once complete).
        epoch_sum: Sum of the per-step statistics of the interrupted epoch so far,
            so the epoch means stay correct (None at an epoch boundary).
        epoch_elapsed_s: Wall time already spent in the interrupted epoch.
        rng_cpu: `torch.get_rng_state()` at the time of saving.
        rng_cuda: `torch.cuda.get_rng_state_all()`, empty without CUDA.
        rng_augment: State of the augmentation generator
            (`torch.Generator.get_state()` of the training device's generator);
            None in files written before device-side augmentation.
        loss_module: State dict of the loss's own learnable module (the supervised
            baseline's linear head); None for losses without one.
    """

    config: dict[str, typing.Any]
    global_step: int
    epoch: int
    step_in_epoch: int
    complete: bool = False
    backbone: dict[str, torch.Tensor] | None = None
    projector: dict[str, torch.Tensor] | None = None
    optimizer: dict[str, typing.Any] | None = None
    epoch_sum: torch.Tensor | None = None
    epoch_elapsed_s: float = 0.0
    rng_cpu: torch.Tensor | None = None
    rng_cuda: list[torch.Tensor] = dataclasses.field(default_factory=list[torch.Tensor])
    rng_augment: torch.Tensor | None = None
    loss_module: dict[str, torch.Tensor] | None = None


def save(path: str | os.PathLike[str], state: ResumeState) -> None:
    """Writes `state` atomically (temp file, then `os.replace`)."""
    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, object] = {"format": FORMAT_VERSION}
    # Not `dataclasses.asdict`: it would deep-copy every tensor.
    blob.update({f.name: getattr(state, f.name) for f in dataclasses.fields(state)})
    tmp = dest.with_name(dest.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, dest)


def load(path: str | os.PathLike[str]) -> ResumeState:
    """Reads a file written by `save` onto the CPU.

    Raises:
        ValueError: The file has an unknown format version.
    """
    blob = typing.cast(
        dict[str, typing.Any],
        torch.load(path, map_location="cpu", weights_only=True),
    )
    version = blob.pop("format", None)
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: resume format {version!r} != {FORMAT_VERSION}")
    return ResumeState(**blob)


def _comparable(cfg: config.ExperimentConfig) -> dict[str, typing.Any]:
    raw = config.to_dict(cfg)
    for section, key in _RUNTIME_ONLY:
        raw[section].pop(key, None)
    return raw


def same_run(a: config.ExperimentConfig, b: config.ExperimentConfig) -> bool:
    """Whether two configs describe the same run (ignoring `data.num_workers`)."""
    return _comparable(a) == _comparable(b)


class PreemptionFlag:
    """Set by SIGUSR1; polled by the training loop at step boundaries."""

    def __init__(self) -> None:
        """Creates an unset flag (no handler is installed yet)."""
        self._requested = False
        self._previous: list[tuple[int, typing.Any]] = []

    @property
    def requested(self) -> bool:
        """Whether a checkpoint-and-exit was requested."""
        return self._requested

    def request(self) -> None:
        """Sets the flag (what the signal handler does)."""
        self._requested = True

    def install(self, signum: int = signal.SIGUSR1) -> None:
        """Makes `signum` set this flag instead of killing the process.

        The handler only sets the flag (no I/O), so it is safe whenever the
        signal arrives.
        """
        self._previous.append((signum, signal.signal(signum, self._handle)))

    def uninstall(self) -> None:
        """Restores the handlers that `install` replaced."""
        while self._previous:
            signum, handler = self._previous.pop()
            signal.signal(signum, handler)

    def _handle(self, signum: int, frame: types.FrameType | None) -> None:
        del signum, frame
        self._requested = True


def ignore_preemption_signal() -> None:
    """Ignores SIGUSR1 in this process (DataLoader workers).

    `run_preemptable` signals the whole process group, which includes DataLoader
    workers; only the main process coordinates checkpointing.
    """
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
