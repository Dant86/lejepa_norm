"""Training workflow: apps/train.py on a tiny fake dataset (CPU), LR schedule."""

import importlib.util
import itertools
import math
import os
import pathlib
import subprocess
import sys
import types

import numpy
import pytest
import torch

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.data import layout
from lejepa_norm.eval import runlog
from lejepa_norm.losses import batch as batch_lib
from lejepa_norm.losses import lejepa
from lejepa_norm.models import checkpoint
from lejepa_norm.train import loop
from lejepa_norm.train import resume
from lejepa_norm.train import schedule

REPO = pathlib.Path(__file__).parent.parent
RUN = "smoke"
# 10 pretraining images at batch size 4 (drop_last) -> 2 steps in one epoch.
TINY = (
    "model.projector_hidden=[32]",
    "model.embed_dim=16",
    "loss.num_slices=16",
    "data.num_workers=0",
    "train.batch_size=4",
    "train.epochs=1",
    "train.warmup_epochs=1",
    "train.log_every_steps=1",
    "train.knn_every_epochs=1",
    "train.checkpoint_every_epochs=1",
    "eval.batch_size=4",
    "eval.knn_k=3",
)


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "train_app", REPO / "apps" / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_stl10(data_dir: pathlib.Path) -> None:
    rng = numpy.random.default_rng(0)
    for split, n in (("unlabeled", 6), ("train", 4), ("test", 4)):
        images = rng.integers(0, 256, size=(n, 96, 96, 3), dtype=numpy.uint8)
        labels = numpy.arange(n, dtype=numpy.int64) % 2
        if split == "unlabeled":
            labels[:] = -1
        path = layout.images_path(data_dir, "stl10", split)
        path.parent.mkdir(parents=True, exist_ok=True)
        numpy.save(path, images)
        numpy.save(layout.labels_path(data_dir, "stl10", split), labels)


@pytest.fixture
def env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    for var, name in (
        (paths.DATA_DIR_VAR, "data"),
        (paths.RUNS_DIR_VAR, "runs"),
        (paths.EMBED_DIR_VAR, "embed"),
        (paths.NOTES_DIR_VAR, "notes"),
    ):
        monkeypatch.setenv(var, str(tmp_path / name))
    _write_fake_stl10(tmp_path / "data")
    return tmp_path


def _train_argv(*extra: str) -> list[str]:
    argv = ["--run-name", RUN, "--env-file", os.devnull]
    for item in (*TINY, *extra):
        argv += ["--set", item]
    return argv


def test_train_app_writes_metrics_config_and_checkpoints(env: pathlib.Path) -> None:
    app = _load_app()
    app.main(_train_argv("seed=3"))
    run_dir = env / "runs" / RUN

    resolved = config.load_config(run_dir / "config.resolved.yaml")
    assert resolved.seed == 3 and resolved.train.batch_size == 4

    records = runlog.read_metrics(run_dir / runlog.METRICS_FILE)
    steps = [r for r in records if r["kind"] == "step"]
    epochs = [r for r in records if r["kind"] == "epoch"]
    assert [r["step"] for r in steps] == [1, 2]
    assert [(r["step"], r["epoch"]) for r in epochs] == [(2, 1)]
    for record in steps:
        assert set(loop.STAT_KEYS) | {"lr", "images_per_s", "epoch"} <= set(record)
    assert {"knn_acc_h", "knn_acc_z", "kappa_z_epoch_mean"} <= set(epochs[0])
    for record in records:
        for key, value in record.items():
            if key != "kind":
                assert isinstance(value, (int, float)) and math.isfinite(value), key
    assert 0.0 <= epochs[0]["knn_acc_h"] <= 1.0
    # Warmup over the whole (2-step) run: lr rises to the batch-scaled peak.
    peak = schedule.scaled_lr(resolved.train.lr, 4)
    assert [r["lr"] for r in steps] == pytest.approx([peak / 2, peak])
    # apps/analyze.py plots every numeric key containing "kappa" against step.
    assert {"kappa_z", "kappa_h"} <= set(runlog.numeric_keys(records, "kappa"))

    for name in (loop.checkpoint_name(1), loop.LAST_CHECKPOINT):
        ckpt = checkpoint.load(run_dir / name)
        assert ckpt.epoch == 1 and ckpt.cfg == resolved

    # The trained model still gives a finite, differentiable loss in bf16 autocast.
    ckpt = checkpoint.load(run_dir / loop.LAST_CHECKPOINT)
    views = [torch.randn(4, 3, 96, 96) for _ in range(2)]
    views += [torch.randn(4, 3, 48, 48) for _ in range(4)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        h, z = loop.forward_views(ckpt.backbone, ckpt.projector, views, 2)
    is_global = torch.arange(6) < 2
    loss, _ = lejepa.build_loss(resolved.loss)(
        batch_lib.ViewBatch(z.float(), is_global)
    )
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert h.shape == (24, 512) and z.shape == (6, 4, 16)
    assert torch.isfinite(loss)
    grads = [p.grad for p in ckpt.backbone.parameters()]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert any(g is not None and bool(g.abs().sum() > 0) for g in grads)

    # Reusing a run name is refused rather than mixing two runs' logs.
    with pytest.raises(SystemExit):
        app.main(_train_argv())


# Runs apps/train.py in this process after patching `schedule.lr_at` so that, just
# before optimizer step STOP_AT + 1, the process either receives a real SIGUSR1
# (preemption warning) or dies abruptly (os._exit, as if killed). The loop calls
# `schedule.lr_at(step, ...)` with the number of steps already taken.
_DRIVER = """
import os, runpy, signal, sys
from lejepa_norm.train import schedule

app, mode, stop_at = sys.argv[1], sys.argv[2], int(sys.argv[3])
original = schedule.lr_at

def lr_at(step, *args):
    if step == stop_at:
        if mode == "signal":
            os.kill(os.getpid(), signal.SIGUSR1)
        else:
            os._exit(137)
    return original(step, *args)

schedule.lr_at = lr_at
sys.argv = [app, *sys.argv[4:]]
runpy.run_path(app, run_name="__main__")
"""
# 10 pretraining images at batch size 2 -> 5 steps per epoch, 10 steps in total.
RESUME_RUN = (
    "train.batch_size=2",
    "train.epochs=2",
    "train.warmup_epochs=1",
)


def _run_driver(
    tmp_path: pathlib.Path, mode: str, stop_at: int, run: str, *extra: str
) -> subprocess.CompletedProcess[str]:
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    argv = ["--run-name", run, "--env-file", os.devnull]
    for item in (*TINY, *RESUME_RUN):
        argv += ["--set", item]
    return subprocess.run(
        [sys.executable, str(driver), str(REPO / "apps" / "train.py"), mode]
        + [str(stop_at), *argv, *extra],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


@pytest.mark.parametrize(
    ("mode", "stop_at", "exit_code", "resumed_step"),
    [
        # SIGUSR1 before step 7: step 7 finishes, then resume.pt is saved at once.
        ("signal", 6, resume.REQUEUE_EXIT_CODE, 7),
        # Killed before step 8: only the end-of-epoch-1 resume.pt (step 5) exists,
        # and the log holds orphan records for steps 6-7.
        ("kill", 7, 137, 5),
    ],
)
def test_interrupted_training_resumes_like_an_uninterrupted_run(
    env: pathlib.Path, mode: str, stop_at: int, exit_code: int, resumed_step: int
) -> None:
    app = _load_app()
    ref_argv = ["--run-name", "ref", "--env-file", os.devnull]
    for item in (*TINY, *RESUME_RUN):
        ref_argv += ["--set", item]
    assert app.main(ref_argv) == 0
    ref_dir = env / "runs" / "ref"
    run_dir = env / "runs" / RUN

    first = _run_driver(env, mode, stop_at, RUN)
    assert first.returncode == exit_code, first.stderr
    state = resume.load(run_dir / resume.RESUME_FILE)
    assert not state.complete
    assert state.global_step == resumed_step
    assert (state.epoch, state.step_in_epoch) == divmod(resumed_step, 5)

    # Without --resume a used run directory is still refused.
    refused = _run_driver(env, "none", -1, RUN)
    assert refused.returncode == 2 and "already used" in refused.stderr

    second = _run_driver(env, "none", -1, RUN, "--resume", "auto")
    assert second.returncode == 0, second.stderr
    assert f"resuming after step {resumed_step}" in second.stdout

    records = runlog.read_metrics(run_dir / runlog.METRICS_FILE)
    ref_records = runlog.read_metrics(ref_dir / runlog.METRICS_FILE)
    steps = [r for r in records if r["kind"] == "step"]
    ref_steps = [r for r in ref_records if r["kind"] == "step"]
    assert [r["step"] for r in steps] == list(range(1, 11))  # contiguous, no dups
    epochs = [(r["step"], r["epoch"]) for r in records if r["kind"] == "epoch"]
    assert epochs == [(5, 1), (10, 2)]
    # The LR schedule continues across the restart.
    peak = schedule.scaled_lr(config.ExperimentConfig().train.lr, 2)
    expected_lr = [schedule.lr_at(s, peak, 5, 10) for s in range(10)]
    assert [r["lr"] for r in steps] == pytest.approx(expected_lr)
    assert [r["lr"] for r in ref_steps] == pytest.approx(expected_lr)
    # Same data order, optimizer state and (num_workers=0) RNG: the resumed run
    # reproduces the uninterrupted one.
    for key in ("loss_total", "z_norm_mean"):
        got = [r[key] for r in steps]
        assert got == pytest.approx([r[key] for r in ref_steps], rel=1e-4), key

    final = checkpoint.load(run_dir / loop.LAST_CHECKPOINT)
    ref_final = checkpoint.load(ref_dir / loop.LAST_CHECKPOINT)
    assert final.epoch == 2
    for got, want in zip(
        final.backbone.state_dict().values(),
        ref_final.backbone.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4)
    assert resume.load(run_dir / resume.RESUME_FILE).complete

    # Resubmitting a finished run is a no-op.
    again = _run_driver(env, "none", -1, RUN, "--resume", "auto")
    assert again.returncode == 0, again.stderr
    assert "nothing to do" in again.stdout
    assert runlog.read_metrics(run_dir / runlog.METRICS_FILE) == records


def test_resume_auto_refuses_a_run_without_resume_state(env: pathlib.Path) -> None:
    app = _load_app()
    assert app.main(_train_argv()) == 0
    run_dir = env / "runs" / RUN
    (run_dir / resume.RESUME_FILE).unlink()  # e.g. a run from before resume.pt
    with pytest.raises(SystemExit):
        app.main([*_train_argv(), "--resume", "auto"])
    # A different config in the same run directory is refused too.
    (run_dir / loop.LAST_CHECKPOINT).unlink()
    with pytest.raises(SystemExit):
        app.main([*_train_argv("seed=5"), "--resume", "auto"])
    assert not (run_dir / resume.RESUME_FILE).exists()


def test_lr_schedule_endpoints_and_batch_scaling() -> None:
    assert schedule.scaled_lr(5e-4, 256) == pytest.approx(5e-4)
    assert schedule.scaled_lr(5e-4, 512) == pytest.approx(1e-3)
    assert schedule.scaled_lr(5e-4, 64) == pytest.approx(1.25e-4)

    peak, warmup, total = 1e-3, 10, 110
    lrs = [schedule.lr_at(s, peak, warmup, total) for s in range(total + 1)]
    assert lrs[0] == pytest.approx(peak / warmup)
    assert lrs[warmup - 1] == pytest.approx(peak)
    assert lrs[warmup] == pytest.approx(peak)
    assert lrs[warmup + 50] == pytest.approx(peak / 2)  # cosine midpoint
    assert lrs[total] == pytest.approx(0.0, abs=1e-12)
    assert all(a < b for a, b in itertools.pairwise(lrs[:warmup]))
    assert all(a > b for a, b in itertools.pairwise(lrs[warmup:]))

    # No warmup: cosine from the peak at step 0.
    assert schedule.lr_at(0, peak, 0, total) == pytest.approx(peak)
    with pytest.raises(ValueError):
        schedule.lr_at(0, peak, total + 1, total)
