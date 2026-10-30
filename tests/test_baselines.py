"""E2 baseline losses (SimCLR, VICReg, supervised) and their training workflow."""

import math
import os
import pathlib

import pytest
import torch

import helpers
from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import runlog
from lejepa_norm.losses import baselines
from lejepa_norm.losses import batch
from lejepa_norm.losses import lejepa
from lejepa_norm.models import checkpoint
from lejepa_norm.train import loop
from lejepa_norm.train import resume

N, D = 32, 16


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _batch(
    z: torch.Tensor,
    labels: torch.Tensor | None = None,
    num_global: int = 2,
) -> batch.ViewBatch:
    return batch.ViewBatch(z, torch.arange(z.shape[0]) < num_global, labels)


def _cfg(variant: str, **kwargs: float) -> config.LossConfig:
    return config.LossConfig(variant=variant, **kwargs)  # pyright: ignore[reportArgumentType]


# ---- SimCLR


def test_nt_xent_known_values() -> None:
    # Orthonormal embeddings, identical views: s_pos = 1, all 2N - 2 others 0.
    n, tau = 4, 0.2
    z = torch.eye(n)
    pos, lse = baselines.nt_xent(z, z, tau)
    assert float(pos) == pytest.approx(-1 / tau)
    expected_lse = math.log(math.exp(1 / tau) + (2 * n - 2))
    assert float(lse) == pytest.approx(expected_lse)


def test_simclr_ignores_the_radius_and_the_local_views() -> None:
    z = torch.randn(2, N, D, generator=_gen(1))
    scale = torch.exp(torch.randn(2, N, 1, generator=_gen(2)))  # per-sample radius
    loss_fn = baselines.SimCLRLoss(_cfg("simclr"))
    base, stats = loss_fn(_batch(z))
    rescaled, _ = loss_fn(_batch(z * scale))
    assert float(rescaled) == pytest.approx(float(base), rel=1e-5)
    assert float(stats["inv"] + stats["dist"]) == pytest.approx(float(base), rel=1e-6)
    extra = torch.cat([z, torch.randn(3, N, D, generator=_gen(3))])
    with_locals, _ = loss_fn(_batch(extra))
    assert float(with_locals) == pytest.approx(float(base), rel=1e-6)


def test_simclr_prefers_aligned_views_and_has_gradients() -> None:
    loss_fn = baselines.SimCLRLoss(_cfg("simclr"))
    z = torch.randn(1, N, D, generator=_gen(4))
    noise = torch.randn(2, N, D, generator=_gen(5))
    aligned, _ = loss_fn(_batch(z.expand(2, -1, -1) + 0.01 * noise))
    unaligned, _ = loss_fn(_batch(noise))
    assert float(aligned) < float(unaligned)
    x = noise.clone().requires_grad_(True)
    loss, _ = loss_fn(_batch(x))
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert x.grad is not None and bool(x.grad.abs().sum() > 0)
    # The gradient is tangent to the sphere: it has no component along z.
    radial = (x.grad * x).sum(dim=-1)
    assert float(radial.detach().abs().max()) < 1e-5


# ---- VICReg


def test_vicreg_known_values() -> None:
    cfg = _cfg("vicreg")
    # Identical, constant views: invariance 0, std = 0 -> variance term 1 (with
    # the 1e-4 epsilon), covariance 0.
    z = torch.zeros(2, N, D)
    total, stats = baselines.VICRegLoss(cfg)(_batch(z))
    eps_std = math.sqrt(1e-4)
    assert float(stats["inv"]) == 0.0 and float(stats["cov"]) == 0.0
    assert float(stats["var"]) == pytest.approx(1 - eps_std)
    assert float(total) == pytest.approx(25 * (1 - eps_std))


def test_vicreg_terms_and_gradients() -> None:
    gen = _gen(6)
    x = torch.randn(2, 512, D, generator=gen).requires_grad_(True)
    total, stats = baselines.VICRegLoss(_cfg("vicreg"))(_batch(x))
    # Independent unit Gaussians: MSE ~ 2, variance hinge ~ 0, covariance ~ small.
    assert float(stats["inv"]) == pytest.approx(2.0, rel=0.1)
    assert float(stats["var"]) < 0.02
    assert float(total) == pytest.approx(
        25 * float(stats["inv"]) + float(stats["dist"]), rel=1e-5
    )
    total.backward()  # pyright: ignore[reportUnknownMemberType]
    assert x.grad is not None and bool(x.grad.abs().sum() > 0)


def test_vicreg_has_no_radial_constraint_but_a_variance_floor() -> None:
    gen = _gen(7)
    z = torch.randn(2, 512, D, generator=gen)
    loss_fn = baselines.VICRegLoss(_cfg("vicreg", sim_coeff=0.0, cov_coeff=0.0))
    _, small = loss_fn(_batch(0.1 * z))
    _, unit = loss_fn(_batch(z))
    _, large = loss_fn(_batch(10 * z))
    assert float(small["total"]) > 20  # collapse is penalized
    assert float(unit["total"]) < 1.0
    assert float(large["total"]) == pytest.approx(0.0, abs=1e-6)  # growth is free


# ---- Supervised


def test_supervised_cross_entropy_known_value_and_gradient_flow() -> None:
    loss_fn = baselines.SupervisedLoss(_cfg("supervised", num_classes=4), D)
    head = loss_fn.module
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    z = torch.randn(3, N, D, generator=_gen(8)).requires_grad_(True)
    labels = torch.arange(N) % 4
    loss, stats = loss_fn(_batch(z, labels))
    assert float(loss) == pytest.approx(math.log(4))  # uniform logits
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert head.weight.grad is not None and bool(head.weight.grad.abs().sum() > 0)
    assert 0.0 <= float(stats["acc"]) <= 1.0
    assert batch.trainable_module(loss_fn) is head
    assert batch.trainable_module(baselines.SimCLRLoss(_cfg("simclr"))) is None
    with pytest.raises(ValueError, match="labels"):
        loss_fn(_batch(z))


def test_build_loss_registry() -> None:
    assert isinstance(lejepa.build_loss(_cfg("simclr")), baselines.SimCLRLoss)
    assert isinstance(lejepa.build_loss(_cfg("vicreg")), baselines.VICRegLoss)
    sup = lejepa.build_loss(_cfg("supervised"), 8)
    assert isinstance(sup, baselines.SupervisedLoss)
    with pytest.raises(ValueError):
        lejepa.build_loss(_cfg("supervised"))
    assert isinstance(lejepa.build_loss(_cfg("A")), lejepa.LeJEPALoss)


# ---- Training workflow (tiny_cnn, CPU)

TINY = (
    *helpers.TINY_MODEL,
    "train.batch_size=2",
    "train.epochs=2",
    "train.warmup_epochs=1",
    "train.log_every_steps=1",
    "train.knn_every_epochs=1",
    "train.checkpoint_every_epochs=1",
    "eval.batch_size=4",
    "eval.knn_k=3",
)


@pytest.fixture
def env(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_stl10_dir: pathlib.Path,
) -> pathlib.Path:
    """Per-test runs dir; the read-only fake data is shared."""
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(fake_stl10_dir))
    for var, name in (
        (paths.RUNS_DIR_VAR, "runs"),
        (paths.EMBED_DIR_VAR, "embed"),
        (paths.NOTES_DIR_VAR, "notes"),
    ):
        monkeypatch.setenv(var, str(tmp_path / name))
    return tmp_path


def _cfg_for(variant: str, *extra: str) -> config.ExperimentConfig:
    cfg_path = pathlib.Path(__file__).parent.parent / "configs" / "base.yaml"
    return config.load_config(cfg_path, [*TINY, f"loss.variant={variant}", *extra])


@pytest.mark.parametrize("variant", ["simclr", "vicreg"])
def test_two_view_baselines_train_end_to_end(env: pathlib.Path, variant: str) -> None:
    cfg = _cfg_for(variant, "data.num_local_views=0")
    run_dir = env / "runs" / variant
    result = loop.train(cfg, os.environ[paths.DATA_DIR_VAR], run_dir)
    assert result.epochs == 2 and not result.preempted
    records = runlog.read_metrics(run_dir / runlog.METRICS_FILE)
    for record in records:
        for key, value in record.items():
            if key == "kind":
                continue
            if value is None:  # NaN is logged as null: only the absent local views
                assert "local" in key, key
            else:
                assert isinstance(value, (int, float)) and math.isfinite(value), key
    assert "knn_acc_z" in [r for r in records if r["kind"] == "epoch"][0]
    ckpt = checkpoint.load(run_dir / loop.LAST_CHECKPOINT)
    assert ckpt.cfg.loss.variant == variant


def test_supervised_trains_on_labeled_split_and_resumes_the_head(
    env: pathlib.Path,
) -> None:
    cfg = _cfg_for("supervised", "loss.num_classes=2")
    data_dir = os.environ[paths.DATA_DIR_VAR]
    run_dir = env / "runs" / "sup"
    flag = resume.PreemptionFlag()
    flag.request()  # preempt after the first step
    first = loop.train(cfg, data_dir, run_dir, preempt=flag)
    assert first.preempted and first.steps == 1
    state = resume.load(run_dir / resume.RESUME_FILE)
    assert state.loss_module is not None
    assert set(state.loss_module) == {"weight", "bias"}
    done = loop.train(cfg, data_dir, run_dir, resume=state)
    # The labeled train split has 4 images -> 2 steps per epoch at batch size 2.
    assert done.steps == 4 and not done.preempted
    records = runlog.read_metrics(run_dir / runlog.METRICS_FILE)
    steps = [r for r in records if r["kind"] == "step"]
    assert [r["step"] for r in steps] == [1, 2, 3, 4]
    assert all(math.isfinite(r["loss_total"]) for r in steps)
