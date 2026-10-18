import math

import pytest
import torch

from lejepa_norm import config
from lejepa_norm.losses import batch
from lejepa_norm.losses import diagnostics
from lejepa_norm.losses import invariance
from lejepa_norm.losses import lejepa
from lejepa_norm.losses import sigreg

DIM = 256


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _sigreg(seed: int = 0) -> sigreg.SIGReg:
    return sigreg.SIGReg(num_slices=1024, num_points=17, generator=_gen(seed))


def _views(z: torch.Tensor, num_global: int = 2) -> batch.ViewBatch:
    is_global = torch.arange(z.shape[0]) < num_global
    return batch.ViewBatch(z=z, is_global=is_global)


def test_noise_floor_matches_closed_form() -> None:
    # Over the real line, E[EP] = sqrt(2 pi) - sqrt(2 pi / 3); truncating at
    # t_max = 3 and using 17 trapezoid nodes only changes it slightly.
    closed_form = math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3)
    assert _sigreg().noise_floor == pytest.approx(closed_form, abs=0.02)


def test_sigreg_near_noise_floor_on_standard_gaussian() -> None:
    reg, gen = _sigreg(), _gen(1)
    values = [float(reg(torch.randn(1536, DIM, generator=gen))) for _ in range(8)]
    mean = sum(values) / len(values)
    assert mean == pytest.approx(reg.noise_floor, rel=0.1)
    assert max(values) < 1.5 * reg.noise_floor


def test_sigreg_large_on_collapsed_and_anisotropic_samples() -> None:
    reg, gen = _sigreg(), _gen(2)
    floor = reg.noise_floor
    point = torch.randn(1, DIM, generator=gen)
    collapsed = point + 0.01 * torch.randn(1536, DIM, generator=gen)
    assert float(reg(collapsed)) > 500 * floor

    scale = torch.ones(DIM)
    scale[: DIM // 2], scale[DIM // 2 :] = 2.0, 0.5
    anisotropic = torch.randn(1536, DIM, generator=gen) * scale
    assert float(reg(anisotropic)) > 30 * floor


def test_sigreg_detects_radius_mismatch() -> None:
    # Directions are uniform but every radius is 10% too large: the statistic
    # must see this, since the radius is what this project studies.
    reg, gen = _sigreg(), _gen(3)
    assert float(reg(1.1 * torch.randn(1536, DIM, generator=gen))) > 4 * (
        reg.noise_floor
    )


@pytest.mark.parametrize("initial_scale", [3.0, 0.3])
def test_sigreg_gradient_recovers_unit_scale(initial_scale: float) -> None:
    # Gradient descent on a global log-scale of Gaussian samples must drive the
    # radius back to that of N(0, I) from either side.
    gen = _gen(4)
    reg = sigreg.SIGReg(num_slices=256, generator=gen)
    z = torch.randn(512, 64, generator=gen)
    log_scale = torch.tensor(initial_scale).log()
    for _ in range(150):
        log_scale = log_scale.detach().requires_grad_()
        (grad,) = torch.autograd.grad(reg(log_scale.exp() * z), log_scale)
        log_scale = log_scale - 1e-3 * grad
    assert float(log_scale.detach().exp()) == pytest.approx(1.0, abs=0.05)


def test_invariance_zero_for_identical_views_and_positive_otherwise() -> None:
    base = torch.randn(1, 8, DIM, generator=_gen(5))
    identical = _views(base.expand(6, 8, DIM).clone())
    assert float(invariance.invariance_loss(identical)) == pytest.approx(0.0)

    # Global views at -1/+1 around the center 0, local views at the center:
    # squared error 1 on 2 of 4 views, averaged over every coordinate.
    ones = torch.ones(8, DIM)
    z = torch.stack([-ones, ones, 0 * ones, 0 * ones])
    assert float(invariance.invariance_loss(_views(z))) == pytest.approx(0.5)


def test_kappa_on_gaussian_and_spherical_samples() -> None:
    gauss = torch.randn(16384, DIM, generator=_gen(6))
    assert float(diagnostics.kappa(gauss)) == pytest.approx(1.0, abs=0.05)
    sphere = math.sqrt(DIM) * torch.nn.functional.normalize(gauss, dim=-1)
    assert float(diagnostics.kappa(sphere)) == pytest.approx(0.0, abs=1e-6)


def test_lejepa_loss_combines_terms_and_reports_diagnostics() -> None:
    cfg = config.LossConfig(lambda_=0.25, num_slices=256)
    loss_fn = lejepa.build_loss(cfg)
    views = _views(torch.randn(6, 128, DIM, generator=_gen(7)))
    total, stats = loss_fn(views)
    assert set(stats) == {"inv", "dist", "total", "kappa"}
    assert all(not v.requires_grad and v.ndim == 0 for v in stats.values())
    expected = 0.75 * stats["inv"] + 0.25 * stats["dist"]
    assert float(total) == pytest.approx(float(expected), rel=1e-6)
    # Independent Gaussian views: per-coordinate MSE to the mean of two
    # independent globals is 1 + 1/2 on locals and 1/2 on globals.
    assert float(stats["inv"]) == pytest.approx((4 * 1.5 + 2 * 0.5) / 6, rel=0.05)


def test_build_loss_rejects_unimplemented_variants() -> None:
    with pytest.raises(NotImplementedError):
        lejepa.build_loss(config.LossConfig(variant="B"))


def test_losses_finite_and_differentiable_under_bf16_autocast() -> None:
    gen = _gen(8)
    layer = torch.nn.Linear(64, DIM)
    x = torch.randn(6 * 32, 64, generator=gen)
    loss_fn = lejepa.LeJEPALoss(config.LossConfig(num_slices=128), generator=gen)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = layer(x)
        assert z.dtype == torch.bfloat16
        total, stats = loss_fn(_views(z.reshape(6, 32, DIM)))
    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    assert all(torch.isfinite(v) for v in stats.values())
    (grad,) = torch.autograd.grad(total, layer.weight)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0
