# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
"""Compares the device-side augmentation with the PIL reference pipeline.

One-off research script for `pipeline_comparison.md` (same folder); every table
in the note is generated here and written with `lejepa_norm.reporting`.

Images: 512 natural 96 px images made offline from the photos bundled with
scikit-image (random square crops, downsampled with PIL bicubic, like STL-10's
downsampled photos). Everything runs on the CPU in fp32 with fixed seeds.

Usage:
    uv run python research_notes/gpu_augmentation/compare_pipelines.py
"""

import datetime
import itertools
import pathlib
import platform
import subprocess

import numpy as np
import PIL
import PIL.Image
import scipy.stats
import skimage.data
import torch
import torchvision
import torchvision.ops
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as tv_functional_v1
import torchvision.transforms.v2.functional as tv_functional

from lejepa_norm import config
from lejepa_norm import reporting
from lejepa_norm.data import gpu_views
from lejepa_norm.data import view_spec
from lejepa_norm.data import views

NOTE = pathlib.Path(__file__).with_name("pipeline_comparison.md")
SIDE = 96
NUM_IMAGES = 512
REPEATS = 4  # augmented views per image and pipeline in the distribution tables
PARITY_IMAGES = 64
PIXELS_PER_CHANNEL = 200_000
CFG = config.DataConfig()
SPECS = {"global": view_spec.global_spec(CFG), "local": view_spec.local_spec(CFG)}
BICUBIC = tv_functional.InterpolationMode.BICUBIC
PHOTOS = (
    "astronaut",
    "coffee",
    "chelsea",
    "rocket",
    "immunohistochemistry",
    "hubble_deep_field",
    "retina",
)


def natural_images(n: int, seed: int = 0) -> np.ndarray:
    """uint8 `(n, 96, 96, 3)`: random square crops of bundled photos, downsampled."""
    photos = [getattr(skimage.data, name)()[..., :3] for name in PHOTOS]
    rng = np.random.default_rng(seed)
    out = np.empty((n, SIDE, SIDE, 3), dtype=np.uint8)
    for i in range(n):
        photo = photos[rng.integers(len(photos))]
        side = int(rng.integers(SIDE, min(photo.shape[:2]) // 2 + 1))
        top = int(rng.integers(0, photo.shape[0] - side + 1))
        left = int(rng.integers(0, photo.shape[1] - side + 1))
        crop = PIL.Image.fromarray(photo[top : top + side, left : left + side])
        out[i] = np.asarray(crop.resize((SIDE, SIDE), PIL.Image.Resampling.BICUBIC))
    return out


def _diff_row(
    op: str, reference: str, got: torch.Tensor, want: torch.Tensor
) -> dict[str, reporting.Cell]:
    signed = (got.float() - want.float()) * 255
    diff = signed.abs()
    return {
        "op": op,
        "compared with": reference,
        "mean signed diff, ours - ref (/255)": float(signed.mean()),
        "max abs diff (/255)": float(diff.max()),
        "mean abs diff (/255)": float(diff.mean()),
        "p99.9 abs diff (/255)": float(
            torch.quantile(diff.flatten()[:1_000_000], 0.999)
        ),
    }


def _pil(image: torch.Tensor) -> PIL.Image.Image:
    """Float `(3, H, W)` in [0, 1] to a PIL image (rounded to uint8)."""
    array = (image * 255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return PIL.Image.fromarray(array)


def _from_pil(image) -> torch.Tensor:
    return tv_functional_v1.to_tensor(image)


def _per_sample(fn, images: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [fn(img, float(v)) for img, v in zip(images, values, strict=True)]
    )


def op_parity(raw: torch.Tensor) -> list[dict[str, reporting.Cell]]:
    """Per-op differences on explicitly supplied, identical parameters."""
    x = gpu_views.to_float(raw)
    gen = torch.Generator().manual_seed(1)
    n = len(x)
    rows: list[dict[str, reporting.Cell]] = []
    b, c, s, h = view_spec.COLOR_JITTER
    factors = {
        "brightness": 1 - b + 2 * b * torch.rand(n, generator=gen),
        "contrast": 1 - c + 2 * c * torch.rand(n, generator=gen),
        "saturation": 1 - s + 2 * s * torch.rand(n, generator=gen),
        "hue": -h + 2 * h * torch.rand(n, generator=gen),
    }
    ops = {
        "brightness": (gpu_views.adjust_brightness, tv_functional.adjust_brightness),
        "contrast": (gpu_views.adjust_contrast, tv_functional.adjust_contrast),
        "saturation": (gpu_views.adjust_saturation, tv_functional.adjust_saturation),
        "hue": (gpu_views.adjust_hue, tv_functional.adjust_hue),
    }
    pil_ops = {
        "brightness": tv_functional_v1.adjust_brightness,
        "contrast": tv_functional_v1.adjust_contrast,
        "saturation": tv_functional_v1.adjust_saturation,
        "hue": tv_functional_v1.adjust_hue,
    }
    for name, (ours, theirs) in ops.items():
        values = factors[name]
        got = ours(x, values)
        rows.append(
            _diff_row(name, "v2 tensor op", got, _per_sample(theirs, x, values))
        )
        pil = torch.stack(
            [
                _from_pil(pil_ops[name](_pil(img), float(v)))
                for img, v in zip(x, values, strict=True)
            ]
        )
        rows.append(_diff_row(name, "PIL backend (reference pipeline)", got, pil))
    gray = gpu_views.grayscale(x).expand_as(x)
    rows.append(
        _diff_row(
            "grayscale",
            "v2 tensor op",
            gray,
            tv_functional.rgb_to_grayscale(x, num_output_channels=3),
        )
    )
    pil_gray = torch.stack(
        [
            _from_pil(tv_functional_v1.rgb_to_grayscale(_pil(img), 3))  # pyright: ignore[reportArgumentType]
            for img in x
        ]
    )
    rows.append(
        _diff_row("grayscale", "PIL backend (reference pipeline)", gray, pil_gray)
    )
    k = SPECS["global"].blur_kernel_size
    sigma = 0.1 + 1.9 * torch.rand(n, generator=gen)
    blurred = gpu_views.gaussian_blur(x, sigma, k)
    rows.append(
        _diff_row(
            "gaussian blur",
            "v2 tensor op",
            blurred,
            torch.stack(
                [
                    tv_functional.gaussian_blur(img, [k, k], [float(sg)] * 2)
                    for img, sg in zip(x, sigma, strict=True)
                ]
            ),
        )
    )
    pil_blur = torch.stack(
        [
            _from_pil(
                tv_functional_v1.gaussian_blur(_pil(img), [k, k], [float(sg)] * 2)  # pyright: ignore[reportArgumentType]
            )
            for img, sg in zip(x, sigma, strict=True)
        ]
    )
    rows.append(
        _diff_row(
            "gaussian blur", "PIL backend (reference pipeline)", blurred, pil_blur
        )
    )
    flip_mask = torch.arange(n) % 2 == 0
    flipped = torch.where(flip_mask.reshape(-1, 1, 1, 1), x.flip(-1), x)
    want = torch.stack(
        [
            tv_functional.horizontal_flip(img) if bool(f) else img
            for img, f in zip(x, flip_mask, strict=True)
        ]
    )
    rows.append(_diff_row("horizontal flip", "v2 tensor op", flipped, want))
    rows.append(
        _diff_row(
            "normalize (units: normalized values x 255)",
            "v2 tensor op",
            gpu_views.normalize(x, CFG),
            tv_functional.normalize(x, list(CFG.mean), list(CFG.std)),
        )
    )
    return rows


def _roi_align(x: torch.Tensor, p: gpu_views.ViewParams, size: int) -> torch.Tensor:
    """Bilinear crop + resize with `torchvision.ops.roi_align` (for context)."""
    boxes = torch.stack(
        [
            torch.arange(len(x)).float(),
            p.left.float(),
            p.top.float(),
            (p.left + p.width).float(),
            (p.top + p.height).float(),
        ],
        dim=1,
    )
    return torchvision.ops.roi_align(x, boxes, size, aligned=True, sampling_ratio=2)


def crop_parity(raw: torch.Tensor) -> list[dict[str, reporting.Cell]]:
    """Crop + resize on identical boxes vs torchvision bicubic and PIL."""
    x = gpu_views.to_float(raw)
    rows: list[dict[str, reporting.Cell]] = []
    for name, spec in SPECS.items():
        p = gpu_views.sample_params(
            spec, len(x), SIDE, SIDE, torch.Generator().manual_seed(2)
        )
        boxes = list(
            zip(
                p.top.tolist(),
                p.left.tolist(),
                p.height.tolist(),
                p.width.tolist(),
                strict=True,
            )
        )
        ours = gpu_views.resized_crop(x, p.top, p.left, p.height, p.width, spec.size)
        tensor_ref = torch.stack(
            [
                tv_functional.resized_crop(
                    img, *box, [spec.size, spec.size], BICUBIC, antialias=True
                )
                for img, box in zip(x, boxes, strict=True)
            ]
        )
        pil_ref = torch.stack(
            [
                _from_pil(
                    tv_functional_v1.resized_crop(
                        PIL.Image.fromarray(img.numpy()),  # pyright: ignore[reportArgumentType]
                        *box,
                        [spec.size, spec.size],
                        tv_transforms.InterpolationMode.BICUBIC,
                    )
                )
                for img, box in zip(raw, boxes, strict=True)
            ]
        )
        label = f"crop + resize, {name} ({spec.size} px)"
        rows.append(
            _diff_row(label, "v2 tensor bicubic, antialias (float)", ours, tensor_ref)
        )
        rows.append(
            _diff_row(
                label, "PIL bicubic (reference pipeline)", ours.clamp(0, 1), pil_ref
            )
        )
        rows.append(
            _diff_row(
                f"roi_align bilinear, {name} (not used)",
                "PIL bicubic (reference pipeline)",
                _roi_align(x, p, spec.size).clamp(0, 1),
                pil_ref,
            )
        )
    return rows


def _reference_view(
    image: torch.Tensor, p: gpu_views.ViewParams, spec: view_spec.ViewSpec
) -> torch.Tensor:
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
        fns = (
            (tv_functional.adjust_brightness, p.brightness),
            (tv_functional.adjust_contrast, p.contrast),
            (tv_functional.adjust_saturation, p.saturation),
            (tv_functional.adjust_hue, p.hue),
        )
        for op in p.order.tolist():
            fn, value = fns[op]
            out = fn(out, float(value))
    if bool(p.grayscale):
        out = tv_functional.rgb_to_grayscale(out, num_output_channels=3)
    if bool(p.blur):
        k = spec.blur_kernel_size
        out = tv_functional.gaussian_blur(out, [k, k], [float(p.sigma)] * 2)
    return out


def _pil_reference_view(
    image: np.ndarray, p: gpu_views.ViewParams, spec: view_spec.ViewSpec
) -> torch.Tensor:
    """The same parameters through torchvision's PIL code path (uint8 steps)."""
    out = tv_functional_v1.resized_crop(
        PIL.Image.fromarray(image),  # pyright: ignore[reportArgumentType]
        int(p.top),
        int(p.left),
        int(p.height),
        int(p.width),
        [spec.size, spec.size],
        tv_transforms.InterpolationMode.BICUBIC,
    )
    if bool(p.flip):
        out = tv_functional_v1.hflip(out)
    if bool(p.jitter):
        fns = (
            (tv_functional_v1.adjust_brightness, p.brightness),
            (tv_functional_v1.adjust_contrast, p.contrast),
            (tv_functional_v1.adjust_saturation, p.saturation),
            (tv_functional_v1.adjust_hue, p.hue),
        )
        for op in p.order.tolist():
            fn, value = fns[op]
            out = fn(out, float(value))
    if bool(p.grayscale):
        out = tv_functional_v1.rgb_to_grayscale(out, num_output_channels=3)
    if bool(p.blur):
        k = spec.blur_kernel_size
        out = tv_functional_v1.gaussian_blur(out, [k, k], [float(p.sigma)] * 2)
    return _from_pil(out)


def full_view_parity(raw: torch.Tensor) -> list[dict[str, reporting.Cell]]:
    """Whole pipeline (sampled parameters) vs the same parameters through v2 ops."""
    x = gpu_views.to_float(raw)
    rows: list[dict[str, reporting.Cell]] = []
    for name, spec in SPECS.items():
        p = gpu_views.sample_params(
            spec, len(x), SIDE, SIDE, torch.Generator().manual_seed(3)
        )
        mean = torch.tensor(CFG.mean).view(1, 3, 1, 1)
        std = torch.tensor(CFG.std).view(1, 3, 1, 1)
        got = gpu_views.apply_view(x, p, spec, CFG) * std + mean  # back to [0, 1]
        want = torch.stack(
            [_reference_view(x[i], p.select(i), spec) for i in range(len(x))]
        )
        pil = torch.stack(
            [
                _pil_reference_view(raw[i].numpy(), p.select(i), spec)
                for i in range(len(x))
            ]
        )
        label = f"full {name} view, before normalization"
        rows.append(
            _diff_row(label, "v2 tensor ops, same parameters and order", got, want)
        )
        rows.append(
            _diff_row(label, "PIL code path, same parameters and order", got, pil)
        )
    return rows


def _pil_views(raw: np.ndarray, spec_name: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    multi = views.MultiViewTransform(CFG)
    transform = multi._global if spec_name == "global" else multi._local  # pyright: ignore[reportPrivateUsage]
    images = [PIL.Image.fromarray(img) for img in raw]
    return torch.stack(
        [transform(img) for _ in range(REPEATS) for img in images]  # pyright: ignore[reportArgumentType]
    )


def _gpu_views(raw: np.ndarray, spec_name: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    batch = torch.from_numpy(raw)
    return torch.cat(
        [
            gpu_views.augment(batch, SPECS[spec_name], CFG, generator)
            for _ in range(REPEATS)
        ]
    )


def _wasserstein(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(scipy.stats.wasserstein_distance(a.numpy(), b.numpy()))


def _ks(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(scipy.stats.ks_2samp(a.numpy(), b.numpy()).statistic)  # pyright: ignore[reportAttributeAccessIssue]


def distributions(
    raw: np.ndarray,
) -> tuple[list[dict[str, reporting.Cell]], list[dict[str, reporting.Cell]]]:
    """Output statistics of both pipelines on the same images.

    Noise floor: the PIL pipeline against itself with another seed.
    """
    pixel_rows: list[dict[str, reporting.Cell]] = []
    view_rows: list[dict[str, reporting.Cell]] = []
    rng = np.random.default_rng(0)
    for name in SPECS:
        pil_a = _pil_views(raw, name, seed=0)
        pil_b = _pil_views(raw, name, seed=1)
        gpu = _gpu_views(raw, name, seed=0)
        for ch, channel in enumerate("RGB"):

            def pixels(t: torch.Tensor, ch: int = ch) -> torch.Tensor:
                flat = t[:, ch].flatten()
                idx = rng.choice(len(flat), PIXELS_PER_CHANNEL, replace=False)
                return flat[torch.from_numpy(idx)]

            a, b, g = pixels(pil_a), pixels(pil_b), pixels(gpu)
            pixel_rows.append(
                {
                    "view": name,
                    "channel": channel,
                    "PIL mean": float(pil_a[:, ch].mean()),
                    "GPU mean": float(gpu[:, ch].mean()),
                    "PIL std": float(pil_a[:, ch].std()),
                    "GPU std": float(gpu[:, ch].std()),
                    "W1 PIL vs GPU": _wasserstein(a, g),
                    "W1 PIL vs PIL (seed 1)": _wasserstein(a, b),
                    "KS PIL vs GPU": _ks(a, g),
                    "KS PIL vs PIL (seed 1)": _ks(a, b),
                }
            )
        # Per-view summaries: what the augmentation does to a whole image.
        stats = {
            "per-view mean": lambda t: t.mean(dim=(1, 2, 3)),
            "per-view std": lambda t: t.std(dim=(1, 2, 3)),
            "per-view saturation (channel spread)": lambda t: (
                t.amax(dim=1) - t.amin(dim=1)
            ).mean(dim=(1, 2)),
            "per-view sharpness (mean abs horizontal gradient)": lambda t: (
                (t[..., 1:] - t[..., :-1]).abs().mean(dim=(1, 2, 3))
            ),
        }
        for stat, fn in stats.items():
            a, b, g = fn(pil_a), fn(pil_b), fn(gpu)
            view_rows.append(
                {
                    "view": name,
                    "statistic": stat,
                    "PIL mean": float(a.mean()),
                    "GPU mean": float(g.mean()),
                    "W1 PIL vs GPU": _wasserstein(a, g),
                    "W1 PIL vs PIL (seed 1)": _wasserstein(a, b),
                    "KS PIL vs GPU": _ks(a, g),
                    "KS PIL vs PIL (seed 1)": _ks(a, b),
                }
            )
    return pixel_rows, view_rows


def sampled_parameters() -> list[dict[str, reporting.Cell]]:
    """Rates and crop statistics of the GPU sampler vs torchvision's sampler."""
    n = 40_000
    rows: list[dict[str, reporting.Cell]] = []
    p = gpu_views.sample_params(
        SPECS["global"], n, SIDE, SIDE, torch.Generator().manual_seed(4)
    )
    for name, flags, rate in (
        ("flip", p.flip, view_spec.FLIP_P),
        ("color jitter applied", p.jitter, view_spec.COLOR_JITTER_P),
        ("grayscale", p.grayscale, view_spec.GRAYSCALE_P),
        ("blur applied (global)", p.blur, view_spec.BLUR_P),
    ):
        rows.append(
            {
                "quantity": name,
                "GPU sampler": float(flags.float().mean()),
                "torchvision": rate,
                "test": f"binomial sd {float((rate * (1 - rate) / n) ** 0.5):.4f}",
            }
        )
    perms = {perm: i for i, perm in enumerate(itertools.permutations(range(4)))}
    counts = np.bincount([perms[tuple(o)] for o in p.order.tolist()], minlength=24)
    rows.append(
        {
            "quantity": "jitter op order (24 permutations)",
            "GPU sampler": f"counts {counts.min()}..{counts.max()}",
            "torchvision": "uniform (randperm)",
            "test": f"chi2 p = {float(scipy.stats.chisquare(counts).pvalue):.3f}",
        }
    )
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    image = torch.zeros(3, SIDE, SIDE)
    for name, spec in SPECS.items():
        top, left, h, w = gpu_views.sample_crops(
            n, SIDE, SIDE, spec.scale, torch.Generator().manual_seed(5)
        )
        ref = np.array(
            [
                tv_transforms.RandomResizedCrop.get_params(
                    image, list(spec.scale), list(view_spec.CROP_RATIO)
                )
                for _ in range(10_000)
            ]
        )
        area = (h * w).double().numpy() / SIDE**2
        ref_area = ref[:, 2] * ref[:, 3] / SIDE**2
        log_aspect = torch.log(w / h).double().numpy()
        ref_log_aspect = np.log(ref[:, 3] / ref[:, 2])
        rows.append(
            {
                "quantity": f"{name} crop area fraction, mean",
                "GPU sampler": float(area.mean()),
                "torchvision": float(ref_area.mean()),
                "test": "KS p = "
                f"{float(scipy.stats.ks_2samp(area, ref_area).pvalue):.3f}",  # pyright: ignore[reportAttributeAccessIssue]
            }
        )
        rows.append(
            {
                "quantity": f"{name} crop log aspect ratio, sd",
                "GPU sampler": float(log_aspect.std()),
                "torchvision": float(ref_log_aspect.std()),
                "test": "KS p = "
                f"{float(scipy.stats.ks_2samp(log_aspect, ref_log_aspect).pvalue):.3f}",  # pyright: ignore[reportAttributeAccessIssue]
            }
        )
    return rows


def provenance() -> list[dict[str, reporting.Cell]]:
    """Versions and time of the run."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).parent,
    ).stdout.strip()
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        {"item": "HEAD when run", "value": commit},
        {"item": "run datetime (UTC)", "value": now},
        {"item": "torch", "value": torch.__version__},
        {"item": "torchvision", "value": torchvision.__version__},
        {"item": "Pillow", "value": PIL.__version__},
        {"item": "python", "value": platform.python_version()},
        {"item": "device", "value": "cpu (fp32)"},
    ]


def main() -> None:
    """Computes every table and writes it into the note."""
    raw = natural_images(NUM_IMAGES)
    parity_raw = torch.from_numpy(raw[:PARITY_IMAGES])
    pixel_rows, view_rows = distributions(raw)
    sections = {
        "provenance": reporting.markdown_table(provenance()),
        "op_parity": reporting.markdown_table(
            op_parity(parity_raw), float_format="{:.3g}"
        ),
        "crop_parity": reporting.markdown_table(
            crop_parity(parity_raw), float_format="{:.3g}"
        ),
        "full_view_parity": reporting.markdown_table(
            full_view_parity(parity_raw), float_format="{:.3g}"
        ),
        "sampled_parameters": reporting.markdown_table(sampled_parameters()),
        "pixel_distributions": reporting.markdown_table(pixel_rows),
        "view_statistics": reporting.markdown_table(view_rows),
    }
    for name, content in sections.items():
        reporting.update_section(NOTE, name, content)
        print(f"## {name}\n{content}\n")


if __name__ == "__main__":
    main()
