"""Training workflow: apps/train.py on a tiny fake dataset (CPU), LR schedule."""

import importlib.util
import itertools
import math
import os
import pathlib
import subprocess
import sys
import types

import pytest
import torch

import helpers
from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import runlog
from lejepa_norm.losses import batch as batch_lib
from lejepa_norm.losses import lejepa
from lejepa_norm.models import checkpoint
from lejepa_norm.train import loop
from lejepa_norm.train import resume
from lejepa_norm.train import schedule
import tiny_backbone

REPO = pathlib.Path(__file__).parent.parent
RUN = "smoke"
# 10 pretraining images at batch size 4 (drop_last) -> 2 steps in one epoch.
TINY = (
    *helpers.TINY_MODEL,
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


@pytest.fixture
def env(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_stl10_dir: pathlib.Path,
) -> pathlib.Path:
    """Per-test runs/embed/notes dirs; the (read-only) fake data is shared."""
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(fake_stl10_dir))
    for var, name in (
        (paths.RUNS_DIR_VAR, "runs"),
        (paths.EMBED_DIR_VAR, "embed"),
        (paths.NOTES_DIR_VAR, "notes"),
    ):
        monkeypatch.setenv(var, str(tmp_path / name))
    return tmp_path


_VIEW_TYPE_KEYS = tuple(
    f"{stat}_{view}"
    for stat in (
        "kappa_z",
        "z_norm_mean",
        "z_norm_std",
        "kappa_h",
        "h_norm_mean",
        "h_norm_std",
    )
    for view in ("global", "local")
)


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
    for key in _VIEW_TYPE_KEYS:  # finite values are checked for all keys below
        assert key in steps[0] and f"{key}_epoch_mean" in epochs[0], key
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
    views = [torch.randn(4, 3, 32, 32) for _ in range(2)]
    views += [torch.randn(4, 3, 16, 16) for _ in range(4)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        h, z = loop.forward_views(ckpt.backbone, ckpt.projector, views, 2)
    is_global = torch.arange(6) < 2
    loss, _ = lejepa.build_loss(resolved.loss)(
        batch_lib.ViewBatch(z.float(), is_global)
    )
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert h.shape == (24, tiny_backbone.FEAT_DIM) and z.shape == (6, 4, 16)
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
import tiny_backbone  # noqa: F401  (registers the test-only backbone; on PYTHONPATH)
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
        env={**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).parent)},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


@pytest.fixture(scope="module")
def reference_dir(
    tmp_path_factory: pytest.TempPathFactory, fake_stl10_dir: pathlib.Path
) -> pathlib.Path:
    """Run dir of the uninterrupted reference run, shared by the resume cases."""
    root = tmp_path_factory.mktemp("reference")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(fake_stl10_dir))
        for var, name in (
            (paths.RUNS_DIR_VAR, "runs"),
            (paths.EMBED_DIR_VAR, "embed"),
            (paths.NOTES_DIR_VAR, "notes"),
        ):
            mp.setenv(var, str(root / name))
        argv = ["--run-name", "ref", "--env-file", os.devnull]
        for item in (*TINY, *RESUME_RUN):
            argv += ["--set", item]
        assert _load_app().main(argv) == 0
    return root / "runs" / "ref"


@pytest.mark.slow
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
    env: pathlib.Path,
    reference_dir: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    stop_at: int,
    exit_code: int,
    resumed_step: int,
) -> None:
    ref_dir = reference_dir
    run_dir = env / "runs" / RUN

    first = _run_driver(env, mode, stop_at, RUN)
    assert first.returncode == exit_code, first.stderr
    state = resume.load(run_dir / resume.RESUME_FILE)
    assert not state.complete
    assert state.global_step == resumed_step
    assert (state.epoch, state.step_in_epoch) == divmod(resumed_step, 5)

    # Without --resume a used run directory is still refused.
    # (In-process: `parser.error` raises SystemExit(2) and prints to stderr.)
    app = _load_app()
    with pytest.raises(SystemExit) as refused:
        app.main(_train_argv(*RESUME_RUN))
    assert refused.value.code == 2
    assert "already used" in capsys.readouterr().err

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
    # Same data order, optimizer state, global RNG and augmentation generator
    # state: the resumed run reproduces the uninterrupted one exactly.
    for key in ("loss_total", "z_norm_mean", *_VIEW_TYPE_KEYS):
        assert [r[key] for r in steps] == [r[key] for r in ref_steps], key
    # The full records (step and epoch means, incl. the per-view-type epoch means of
    # the epoch that spans the interruption) match, except wall-clock throughput.
    timing = ("images_per_s", "epoch_time_s")
    assert [{k: v for k, v in r.items() if k not in timing} for r in records] == [
        {k: v for k, v in r.items() if k not in timing} for r in ref_records
    ]

    final = checkpoint.load(run_dir / loop.LAST_CHECKPOINT)
    ref_final = checkpoint.load(ref_dir / loop.LAST_CHECKPOINT)
    assert final.epoch == 2
    for got, want in zip(
        final.backbone.state_dict().values(),
        ref_final.backbone.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert resume.load(run_dir / resume.RESUME_FILE).complete

    # Resubmitting a finished run is a no-op.
    # (`apps/train.py` ends with `sys.exit(main())`, so 0 here is exit code 0.)
    assert app.main([*_train_argv(*RESUME_RUN), "--resume", "auto"]) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert runlog.read_metrics(run_dir / runlog.METRICS_FILE) == records


@pytest.mark.slow
def test_resume_state_from_before_view_type_stats_still_loads(
    env: pathlib.Path, reference_dir: pathlib.Path
) -> None:
    """An old-format `resume.pt` (9-entry `epoch_sum`) continues the run."""
    run_dir = env / "runs" / RUN
    first = _run_driver(env, "signal", 6, RUN)
    assert first.returncode == resume.REQUEUE_EXIT_CODE, first.stderr
    state = resume.load(run_dir / resume.RESUME_FILE)
    assert state.epoch_sum is not None  # saved mid-epoch (step 7 of epoch 2)
    assert state.epoch_sum.numel() == len(loop.STAT_KEYS)
    state.epoch_sum = state.epoch_sum[: loop.LEGACY_NUM_STATS].clone()
    resume.save(run_dir / resume.RESUME_FILE, state)

    second = _run_driver(env, "none", -1, RUN, "--resume", "auto")
    assert second.returncode == 0, second.stderr
    records = runlog.read_metrics(run_dir / runlog.METRICS_FILE)
    ref_records = runlog.read_metrics(reference_dir / runlog.METRICS_FILE)
    old_keys = set(loop.STAT_KEYS[: loop.LEGACY_NUM_STATS])
    epoch_two = next(r for r in records if r["kind"] == "epoch" and r["epoch"] == 2)
    ref_two = next(r for r in ref_records if r["kind"] == "epoch" and r["epoch"] == 2)
    for key in old_keys:  # the pooled statistics are unaffected
        assert epoch_two[f"{key}_epoch_mean"] == ref_two[f"{key}_epoch_mean"], key
    # The spanning epoch has no (partial, wrong) per-view-type means; steps do.
    assert all(f"{key}_epoch_mean" not in epoch_two for key in _VIEW_TYPE_KEYS)
    epoch_one = next(r for r in records if r["kind"] == "epoch" and r["epoch"] == 1)
    assert all(f"{key}_epoch_mean" in epoch_one for key in _VIEW_TYPE_KEYS)
    steps = [r for r in records if r["kind"] == "step"]
    assert [r["step"] for r in steps] == list(range(1, 11))
    assert all(key in r for r in steps for key in _VIEW_TYPE_KEYS)


def test_training_does_not_depend_on_num_workers(env: pathlib.Path) -> None:
    """Augmentation runs on the device from one seeded generator, not in workers."""
    app = _load_app()
    results: list[tuple[list[dict[str, float]], dict[str, torch.Tensor]]] = []
    for workers in (0, 3):
        argv = ["--run-name", f"w{workers}", "--env-file", os.devnull]
        for item in (*TINY, "train.epochs=2", f"data.num_workers={workers}"):
            argv += ["--set", item]
        assert app.main(argv) == 0
        run_dir = env / "runs" / f"w{workers}"
        records = [
            {k: v for k, v in r.items() if k not in ("images_per_s", "epoch_time_s")}
            for r in runlog.read_metrics(run_dir / runlog.METRICS_FILE)
        ]
        weights = checkpoint.load(run_dir / loop.LAST_CHECKPOINT).backbone.state_dict()
        results.append((records, weights))
    (records_a, weights_a), (records_b, weights_b) = results
    assert records_a == records_b
    for name, value in weights_a.items():
        torch.testing.assert_close(value, weights_b[name], rtol=0, atol=0)


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
