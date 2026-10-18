# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Device-side augmentation vs. torchvision: op parity, distributions, seeding.

Color ops, blur, flip and normalization are compared with
`torchvision.transforms.v2.functional` on float tensors (the ground truth);
crop + resize with torchvision's antialiased bicubic tensor resize and with the
PIL reference pipeline (`data.views`).
"""

import itertools

import numpy as np
import PIL.Image
import pytest
import scipy.stats
import skimage.data
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as tv_functional_v1
import torchvision.transforms.v2.functional as tv_functional

from lejepa_norm import config
from lejepa_norm.data import gpu_views
from lejepa_norm.data import view_spec
from lejepa_norm.data import views

CFG = config.DataConfig()
GLOBAL = view_spec.global_spec(CFG)
LOCAL = view_spec.local_spec(CFG)
BICUBIC = tv_functional.InterpolationMode.BICUBIC


def _natural_batch(n: int, side: int = 96) -> torch.Tensor:
    """uint8 `(n, side, side, 3)` patches of a natural photo (bundled, offline)."""
    photo = skimage.data.astronaut()
    rng = np.random.default_rng(0)
    tops = rng.integers(0, photo.shape[0] - side, n)
    lefts = rng.integers(0, photo.shape[1] - side, n)
    patches = [
        photo[t : t + side, s : s + side] for t, s in zip(tops, lefts, strict=True)
    ]
    return torch.from_numpy(np.stack(patches))


def _params(n: int, seed: int = 0, spec: view_spec.ViewSpec = GLOBAL):
    gen = torch.Generator().manual_seed(seed)
    return gpu_views.sample_params(spec, n, 96, 96, gen)


def _pil_resized_crop(
    image: np.ndarray, box: tuple[int, int, int, int], size: int
) -> torch.Tensor:
    """The PIL reference: torchvision's resized crop on a PIL image, bicubic."""
    pil = tv_functional_v1.resized_crop(
        PIL.Image.fromarray(image),  # pyright: ignore[reportArgumentType]
        *box,
        [size, size],
        tv_transforms.InterpolationMode.BICUBIC,
    )
    return tv_functional_v1.to_tensor(pil)  # pyright: ignore[reportArgumentType]


def _ks_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    return float(scipy.stats.ks_2samp(a, b).pvalue)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.fixture(scope="module")
def images() -> torch.Tensor:
    """Float `(8, 3, 96, 96)` natural images in [0, 1]."""
    return gpu_views.to_float(_natural_batch(8))


def test_color_ops_match_torchvision_per_sample(images: torch.Tensor) -> None:
    gen = torch.Generator().manual_seed(1)
    factors = 0.6 + 0.8 * torch.rand(len(images), generator=gen)
    shifts = -0.1 + 0.2 * torch.rand(len(images), generator=gen)
    cases = [
        (gpu_views.adjust_brightness, tv_functional.adjust_brightness, factors),
        (gpu_views.adjust_contrast, tv_functional.adjust_contrast, factors),
        (gpu_views.adjust_saturation, tv_functional.adjust_saturation, factors),
        (gpu_views.adjust_hue, tv_functional.adjust_hue, shifts),
    ]
    for ours, theirs, values in cases:
        got = ours(images, values)
        want = torch.stack(
            [theirs(img, float(v)) for img, v in zip(images, values, strict=True)]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-5, msg=ours.__name__)

    gray = gpu_views.grayscale(images)
    torch.testing.assert_close(gray, tv_functional.rgb_to_grayscale(images))


def test_blur_flip_and_normalize_match_torchvision(images: torch.Tensor) -> None:
    sigma = torch.tensor([0.1, 0.5, 1.0, 2.0, 0.3, 1.5, 0.7, 1.9])
    kernel = GLOBAL.blur_kernel_size
    got = gpu_views.gaussian_blur(images, sigma, kernel)
    want = torch.stack(
        [
            tv_functional.gaussian_blur(img, [kernel, kernel], [float(s), float(s)])
            for img, s in zip(images, sigma, strict=True)
        ]
    )
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)

    torch.testing.assert_close(
        gpu_views.normalize(images, CFG),
        tv_functional.normalize(images, list(CFG.mean), list(CFG.std)),
    )
    # Center view == the PIL reference's CenterTransform.
    raw = _natural_batch(3)
    center = views.CenterTransform(CFG)
    want = torch.stack([center(PIL.Image.fromarray(img.numpy())) for img in raw])
    torch.testing.assert_close(gpu_views.center_view(raw, CFG), want, rtol=0, atol=1e-6)


def test_crop_resize_matches_torchvision_bicubic_and_pil() -> None:
    raw = _natural_batch(64)
    x = gpu_views.to_float(raw)
    for spec in (GLOBAL, LOCAL):
        p = _params(len(x), seed=2, spec=spec)
        got = gpu_views.resized_crop(x, p.top, p.left, p.height, p.width, spec.size)
        boxes = list(
            zip(
                p.top.tolist(),
                p.left.tolist(),
                p.height.tolist(),
                p.width.tolist(),
                strict=True,
            )
        )
        # Antialiased bicubic on float tensors: same algorithm, float rounding only.
        want = torch.stack(
            [
                tv_functional.resized_crop(
                    img, *box, [spec.size, spec.size], BICUBIC, antialias=True
                )
                for img, box in zip(x, boxes, strict=True)
            ]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
        # PIL works in uint8: fixed-point weights, and the horizontal pass is
        # rounded and saturated to 0..255 before the vertical one, which moves
        # isolated pixels at strong edges by a few levels.
        pil = torch.stack(
            [
                _pil_resized_crop(img.numpy(), box, spec.size)
                for img, box in zip(raw, boxes, strict=True)
            ]
        )
        diff = (got.clamp(0, 1) - pil).abs() * 255
        assert float(diff.mean()) <= 0.5, spec
        assert float(torch.quantile(diff.flatten()[:500_000], 0.999)) <= 1.5, spec
        assert float(diff.max()) <= 8.0, spec


def _reference_view(
    image: torch.Tensor, p: gpu_views.ViewParams, spec: view_spec.ViewSpec
) -> torch.Tensor:
    """One sample through torchvision v2 ops, in torchvision's order."""
    out = tv_functional.resized_crop(
        image,
        int(p.top),
        int(p.left),
        int(p.height),
        int(p.width),
        [spec.size, spec.size],
        BICUBIC,
        antialias=True,
    ).clamp(0, 1)
    if bool(p.flip):
        out = tv_functional.horizontal_flip(out)
    if bool(p.jitter):
        ops = {
            gpu_views.BRIGHTNESS: (tv_functional.adjust_brightness, p.brightness),
            gpu_views.CONTRAST: (tv_functional.adjust_contrast, p.contrast),
            gpu_views.SATURATION: (tv_functional.adjust_saturation, p.saturation),
            gpu_views.HUE: (tv_functional.adjust_hue, p.hue),
        }
        for op in p.order.tolist():
            fn, value = ops[op]
            out = fn(out, float(value))
    if bool(p.grayscale):
        out = tv_functional.rgb_to_grayscale(out, num_output_channels=3)
    if bool(p.blur):
        k = spec.blur_kernel_size
        out = tv_functional.gaussian_blur(out, [k, k], [float(p.sigma)] * 2)
    return tv_functional.normalize(out, list(CFG.mean), list(CFG.std))


def test_full_view_matches_torchvision_ops_with_the_same_parameters() -> None:
    """Whole pipeline, incl. the per-sample jitter gate and op order."""
    x = gpu_views.to_float(_natural_batch(48))
    for spec in (GLOBAL, LOCAL):
        p = _params(len(x), seed=3, spec=spec)
        assert p.jitter.any() and (~p.jitter).any() and p.grayscale.any()
        assert len({tuple(o) for o in p.order.tolist()}) > 10
        got = gpu_views.apply_view(x, p, spec, CFG)
        want = torch.stack(
            [_reference_view(x[i], p.select(i), spec) for i in range(48)]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-4)


def test_sampled_crops_follow_torchvision_distribution() -> None:
    n = 20000
    for spec, side in ((GLOBAL, 96), (LOCAL, 96), (GLOBAL, 40)):
        gen = torch.Generator().manual_seed(4)
        top, left, h, w = gpu_views.sample_crops(n, side, side, spec.scale, gen)
        assert bool((top >= 0).all() and (left >= 0).all())
        assert bool((top + h <= side).all() and (left + w <= side).all())
        assert bool((h > 0).all() and (w > 0).all())
        # Every box is the rounding of some (w', h') with area and aspect ratio
        # inside the configured ranges (|w - w'|, |h - h'| <= 0.5).
        hd, wd = h.double(), w.double()
        lo_ratio, hi_ratio = view_spec.CROP_RATIO
        assert bool(((wd + 0.5) / (hd - 0.5) >= lo_ratio).all())
        assert bool(((wd - 0.5) / (hd + 0.5) <= hi_ratio).all())
        assert bool(((wd + 0.5) * (hd + 0.5) >= spec.scale[0] * side**2).all())
        assert bool(((wd - 0.5) * (hd - 0.5) <= spec.scale[1] * side**2).all())
        area = (hd * wd) / side**2
        aspect = wd / hd
        # Same distribution as torchvision's sampler (KS on area and aspect).
        torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
        image = torch.zeros(3, side, side)
        ref = np.array(
            [
                tv_transforms.RandomResizedCrop.get_params(
                    image, list(spec.scale), list(view_spec.CROP_RATIO)
                )
                for _ in range(4000)
            ]
        )
        ref_area = ref[:, 2] * ref[:, 3] / side**2
        ref_aspect = ref[:, 3] / ref[:, 2]
        assert _ks_pvalue(area.numpy(), ref_area) > 1e-3
        assert _ks_pvalue(aspect.numpy(), ref_aspect) > 1e-3
        # Offsets: uniform over the valid range (centered on average).
        rel = (top.double() + h / 2) / side
        assert abs(float(rel.mean()) - 0.5) < 0.01


def test_sampled_flags_and_factors_follow_torchvision_distribution() -> None:
    n = 40000
    p = _params(n, seed=5)
    rates = {
        "flip": (p.flip, view_spec.FLIP_P),
        "jitter": (p.jitter, view_spec.COLOR_JITTER_P),
        "grayscale": (p.grayscale, view_spec.GRAYSCALE_P),
        "blur": (p.blur, view_spec.BLUR_P),
    }
    for name, (flags, rate) in rates.items():
        # 5 standard errors of a Bernoulli mean.
        assert (
            abs(float(flags.float().mean()) - rate) < 5 * (rate * (1 - rate) / n) ** 0.5
        ), name
    assert not _params(1000, spec=LOCAL).blur.any()

    b, c, s, h = view_spec.COLOR_JITTER
    for values, lo, hi in (
        (p.brightness, 1 - b, 1 + b),
        (p.contrast, 1 - c, 1 + c),
        (p.saturation, 1 - s, 1 + s),
        (p.hue, -h, h),
        (p.sigma, *view_spec.BLUR_SIGMA),
    ):
        assert float(values.min()) >= lo and float(values.max()) <= hi
        # Uniform: KS against U(lo, hi).
        u = ((values - lo) / (hi - lo)).numpy()
        assert scipy.stats.kstest(u, "uniform").pvalue > 1e-3

    # Every sample's order is a permutation, and all 24 are equally likely.
    assert bool((p.order.sort(dim=1).values == torch.arange(4)).all())
    perms = {perm: i for i, perm in enumerate(itertools.permutations(range(4)))}
    counts = np.bincount([perms[tuple(o)] for o in p.order.tolist()], minlength=24)
    assert scipy.stats.chisquare(counts).pvalue > 1e-3


def test_same_seed_same_views_different_seed_different_views() -> None:
    raw = _natural_batch(6)
    augment = gpu_views.MultiViewAugment(CFG)

    def run(seed: int) -> list[torch.Tensor]:
        return augment(raw, torch.Generator().manual_seed(seed))

    first, again, other = run(7), run(7), run(8)
    assert [tuple(v.shape) for v in first] == [(6, 3, 96, 96)] * 2 + [
        (6, 3, 48, 48)
    ] * 4
    assert all(v.dtype == torch.float32 for v in first)
    for a, b in zip(first, again, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(first, other, strict=True):
        assert not torch.allclose(a, b)
    # Views of one call differ from each other.
    assert not torch.allclose(first[0], first[1])
    # Channels-first input gives the same result.
    chw = augment(raw.permute(0, 3, 1, 2), torch.Generator().manual_seed(7))
    torch.testing.assert_close(chw[0], first[0], rtol=0, atol=0)


def test_samples_get_independent_parameters() -> None:
    # The same image repeated: every copy must get its own augmentation.
    raw = _natural_batch(1).expand(16, -1, -1, -1).contiguous()
    view = gpu_views.augmented_view(raw, CFG, torch.Generator().manual_seed(9))
    flat = view.flatten(1)
    for i, j in itertools.combinations(range(16), 2):
        assert not torch.allclose(flat[i], flat[j]), (i, j)
    p = _params(16)
    assert len(set(p.brightness.tolist())) == 16
    assert (
        len(set(zip(p.top.tolist(), p.left.tolist(), p.height.tolist(), strict=True)))
        > 12
    )


def test_black_image_stays_black_under_every_augmentation() -> None:
    cfg = config.DataConfig(mean=(0.5, 0.4, 0.3), std=(0.25, 0.2, 0.1))
    black = torch.zeros(5, 96, 96, 3, dtype=torch.uint8)
    expected = torch.tensor([-2.0, -2.0, -3.0]).view(1, 3, 1, 1)
    outputs = gpu_views.MultiViewAugment(cfg)(black, torch.Generator().manual_seed(0))
    for out in [gpu_views.center_view(black, cfg), *outputs]:
        torch.testing.assert_close(out, expected.expand_as(out), rtol=0, atol=1e-5)


def test_rejects_non_uint8_or_channelless_input() -> None:
    with pytest.raises(ValueError):
        gpu_views.to_float(torch.zeros(2, 8, 8, 3))
    with pytest.raises(ValueError):
        gpu_views.to_float(torch.zeros(2, 8, 8, 4, dtype=torch.uint8))
