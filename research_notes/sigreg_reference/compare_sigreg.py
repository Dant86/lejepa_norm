"""Compares our SIGReg and invariance loss with the official LeJEPA code.

One-off research script for `reference_comparison.md` (same folder). The official
code (https://github.com/rbalestr-lab/lejepa, CC BY-NC 4.0) is NOT part of this
repository: clone it anywhere outside the repo and pass its root with
`--official-dir` (or `LEJEPA_OFFICIAL_DIR`). The script imports the official
`lejepa` package from that path and, for the training-example conventions, executes
snippets read at run time from the clone's `MINIMAL.md`. Nothing from the official
repository is copied here.

Slice directions: the official `SlicingUnivariateTest` seeds a fresh
`torch.Generator` with its step counter (0 on the first call) and draws a `(d, K)`
standard normal matrix; our `SIGReg` draws the same `(d, K)` matrix from the
generator it is given. Seeding ours with 0 and using a fresh official module per
evaluation therefore gives both implementations the SAME directions, so the
comparisons below contain no slice noise.

Usage:
    uv run python research_notes/sigreg_reference/compare_sigreg.py \
        --official-dir <path to official clone>
"""

import argparse
import dataclasses
import datetime
import importlib
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import types
from typing import Any

import torch

from lejepa_norm import config
from lejepa_norm import reporting
from lejepa_norm.losses import batch as batch_lib
from lejepa_norm.losses import invariance
from lejepa_norm.losses import lejepa as lejepa_loss
from lejepa_norm.losses import sigreg

NOTE = pathlib.Path(__file__).with_name("reference_comparison.md")
DIM = 256
NUM_SLICES = 1024
NUM_POINTS = 17
T_MAX = 3.0
SAMPLE_SIZES = (256, 1536, 6144)
FLOOR_REPEATS = 10
NUM_VIEWS = 6  # our base config: 2 global + 4 local views
# Excess over the noise floor below which an excess ratio is not reported.
MIN_EXCESS = 0.5
BATCH = 256
LAMBDA = 0.05


@dataclasses.dataclass(frozen=True)
class Input:
    """A synthetic input: the affine map `z = scale * x + shift`, x ~ N(0, I_d)."""

    scale: torch.Tensor
    shift: float = 0.0

    def sample(self, n: int, seed: int) -> torch.Tensor:
        """Draws `(n, d)` samples."""
        gen = torch.Generator().manual_seed(seed)
        return self.transform(torch.randn(n, DIM, generator=gen))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the map to standard normal samples of any leading shape."""
        return x * self.scale + self.shift


_RANK16 = (torch.arange(DIM) < 16).float()
INPUTS: dict[str, Input] = {
    "N(0, I)": Input(torch.ones(DIM)),
    # Per-coordinate variances evenly spread over [0.25, 1.75] (trace = d).
    "anisotropic (var 0.25..1.75)": Input(torch.linspace(0.25, 1.75, DIM).sqrt()),
    # Rank-16 collapse: only the first 16 coordinates are non-zero.
    "collapsed (rank 16)": Input(_RANK16),
    "radius x1.1": Input(torch.full((DIM,), 1.1)),
    "mean shift +0.1": Input(torch.ones(DIM), shift=0.1),
}
GAUSSIAN = INPUTS["N(0, I)"]


def load_official(root: pathlib.Path) -> Any:
    """Imports the official `lejepa` package from a clone outside the repo."""
    if not (root / "lejepa" / "__init__.py").is_file():
        raise FileNotFoundError(f"no lejepa package under {root}")
    sys.path.insert(0, str(root))
    return importlib.import_module("lejepa")


def official_commit(root: pathlib.Path) -> str:
    """Returns the commit hash checked out in the official clone."""
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def official_stat(lib: Any, num_slices: int = NUM_SLICES) -> Any:
    """Fresh official sliced Epps-Pulley module (step counter 0, so seed 0)."""
    ep = lib.univariate.EppsPulley(t_max=T_MAX, n_points=NUM_POINTS)
    return lib.multivariate.SlicingUnivariateTest(ep, num_slices=num_slices)


def our_stat(num_slices: int = NUM_SLICES, seed: int = 0) -> sigreg.SIGReg:
    """Our statistic with directions drawn from a generator seeded `seed`."""
    gen = torch.Generator().manual_seed(seed)
    return sigreg.SIGReg(
        num_slices=num_slices, num_points=NUM_POINTS, t_max=T_MAX, generator=gen
    )


def _python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL)


def minimal_snippets(root: pathlib.Path) -> tuple[Any, str, str]:
    """Reads the training example's SIGReg class and loss lines from MINIMAL.md.

    The example hard-codes `device="cuda"` when drawing directions; that single
    token is replaced by `"cpu"` so it runs here. Nothing else is changed.

    Returns:
        The example's SIGReg class, and the right-hand sides of its `inv_loss = `
        and `lejepa_loss = ` assignments.
    """
    blocks = _python_blocks((root / "MINIMAL.md").read_text())
    source = next(b for b in blocks if "class SIGReg" in b)
    start, stop = source.index("class SIGReg"), source.index("class ViTEncoder")
    cls_src = source[start:stop]
    if cls_src.count('device="cuda"') != 1:
        raise RuntimeError("MINIMAL.md SIGReg no longer matches the expected form")
    namespace: dict[str, Any] = {"torch": torch}
    exec(cls_src.replace('device="cuda"', 'device="cpu"'), namespace)  # noqa: S102
    loop = next(b for b in blocks if "inv_loss =" in b)
    inv = re.search(r"^\s*inv_loss = (.+)$", loop, flags=re.MULTILINE)
    total = re.search(r"^\s*lejepa_loss = (.+)$", loop, flags=re.MULTILINE)
    if inv is None or total is None:
        raise RuntimeError("MINIMAL.md loss lines not found")
    return namespace["SIGReg"], inv.group(1), total.group(1)


def _f(x: torch.Tensor | float) -> float:
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def table_same_inputs(lib: Any, floor: float) -> str:
    """Our vs official statistic on identical samples and directions."""
    rows: list[dict[str, reporting.Cell]] = []
    for name, inp in INPUTS.items():
        for n in SAMPLE_SIZES:
            z = inp.sample(n, seed=1)
            ours = _f(our_stat()(z))
            theirs = _f(official_stat(lib)(z))
            rows.append(
                {
                    "input": name,
                    "N": n,
                    "ours": ours,
                    "official": theirs,
                    "ours / official": ours / theirs,
                    "rel. diff": abs(ours - theirs) / abs(theirs),
                    "(ours - floor) / N": (ours - floor) / n,
                }
            )
    return reporting.markdown_table(rows)


def table_quadrature(lib: Any, ours: sigreg.SIGReg) -> str:
    """Noise floor implied by each implementation's quadrature weights."""
    ep = lib.univariate.EppsPulley(t_max=T_MAX, n_points=NUM_POINTS)
    t = ep.t.double()
    official_floor = float((ep.weights.double() * (1 - torch.exp(-t.square()))).sum())
    # integral_{-a}^{a} exp(-t^2 / 2) (1 - exp(-t^2)) dt, in closed form.
    exact = math.sqrt(2 * math.pi) * math.erf(T_MAX / math.sqrt(2)) - math.sqrt(
        2 * math.pi / 3
    ) * math.erf(T_MAX * math.sqrt(1.5))
    rows: list[dict[str, reporting.Cell]] = [
        {"quantity": "nodes", "ours": ours.num_points, "official": int(t.numel())},
        {"quantity": "t range", "ours": f"[0, {ours.t_max}]", "official": "[0, 3]"},
        {
            "quantity": "E[T] under H0 (from weights)",
            "ours": ours.noise_floor,
            "official": official_floor,
        },
        {
            "quantity": "E[T] under H0, exact integral on [-3, 3]",
            "ours": exact,
            "official": exact,
        },
    ]
    return reporting.markdown_table(rows, float_format="{:.6f}")


def table_floor(lib: Any, floor: float) -> str:
    """Monte Carlo noise floor on N(0, I) for several sample sizes."""
    rows: list[dict[str, reporting.Cell]] = []
    for n in SAMPLE_SIZES:
        ours_vals: list[float] = []
        theirs_vals: list[float] = []
        for rep in range(FLOOR_REPEATS):
            z = GAUSSIAN.sample(n, seed=100 + rep)
            # Different directions per repeat; identical for both implementations.
            ours_vals.append(_f(our_stat(seed=rep)(z)))
            stat = official_stat(lib)
            stat.global_step.fill_(rep)
            theirs_vals.append(_f(stat(z)))
        o, t = torch.tensor(ours_vals), torch.tensor(theirs_vals)
        rows.append(
            {
                "N": n,
                "theory": floor,
                "ours mean": _f(o.mean()),
                "ours sd": _f(o.std()),
                "official mean": _f(t.mean()),
                "official sd": _f(t.std()),
                "max rel. diff": _f(((o - t).abs() / t).max()),
            }
        )
    return reporting.markdown_table(rows)


def make_views(inp: Input, s: float, seed: int) -> torch.Tensor:
    """`(V, B, d)` views of one shared image code, mapped through `inp`.

    Before the map, view `v` of image `i` is `sqrt(1 - s^2) c_i + s e_vi`
    (`c`, `e` i.i.d. N(0, I)): `s = 1` gives independent views, `s = 0` identical
    views, and every view is marginally N(0, I) before the map.
    """
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(1, BATCH, DIM, generator=gen)
    e = torch.randn(NUM_VIEWS, BATCH, DIM, generator=gen)
    return inp.transform(math.sqrt(1 - s * s) * c + s * e)


def table_training_layout(lib: Any, minimal_sigreg: Any, floor: float) -> str:
    """Concatenated-views (ours) vs per-view (official example) SIGReg."""
    cases: list[tuple[str, float]] = [
        ("N(0, I)", 1.0),
        ("N(0, I)", 0.5),
        ("N(0, I)", 0.1),
        ("N(0, I)", 0.0),
        ("anisotropic (var 0.25..1.75)", 1.0),
        ("radius x1.1", 1.0),
        ("radius x1.1", 0.5),
        ("mean shift +0.1", 1.0),
    ]
    rows: list[dict[str, reporting.Cell]] = []
    for name, s in cases:
        z = make_views(INPUTS[name], s, seed=7)
        flat = z.reshape(-1, DIM)
        ours_concat = _f(our_stat()(flat))
        off_concat = _f(official_stat(lib)(flat))
        off_view = _f(official_stat(lib)(z))
        # The example draws from the global RNG; seed 0 gives our directions.
        torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
        minimal = _f(minimal_sigreg()(z))
        ours_view = _f(torch.stack([our_stat(num_slices=256)(v) for v in z]).mean())
        rows.append(
            {
                "views": name,
                "view noise s": s,
                "ours concat (N=1536)": ours_concat,
                "official lib, concat": off_concat,
                "official lib, per view (N=256)": off_view,
                "official MINIMAL (256 slices)": minimal,
                "ours per view (256 slices)": ours_view,
                "concat / per view": ours_concat / off_view,
                "excess ratio": (
                    (ours_concat - floor) / (off_view - floor)
                    if off_view - floor > MIN_EXCESS
                    else None
                ),
            }
        )
    return reporting.markdown_table(rows)


def table_invariance(inv_expr: str) -> str:
    """Our invariance loss vs the official example's expression on the same views."""
    rows: list[dict[str, reporting.Cell]] = []
    for s in (1.0, 0.5, 0.1):
        z = make_views(GAUSSIAN, s, seed=11)
        for layout, n_global in (
            ("all 6 global", NUM_VIEWS),
            ("2 global + 4 local", 2),
        ):
            is_global = torch.arange(NUM_VIEWS) < n_global
            z_ours = z.clone().requires_grad_(True)
            ours = invariance.invariance_loss(batch_lib.ViewBatch(z_ours, is_global))
            ours.backward()  # pyright: ignore[reportUnknownMemberType]
            z_off = z.clone().requires_grad_(True)
            theirs = eval(inv_expr, {"torch": torch}, {"proj": z_off})  # noqa: S307
            theirs.backward()
            assert z_ours.grad is not None and z_off.grad is not None
            rows.append(
                {
                    "view noise s": s,
                    "our layout": layout,
                    "ours": _f(ours),
                    "official (all views = center)": _f(theirs),
                    "ours / official": _f(ours) / _f(theirs),
                    "max abs grad diff": _f((z_ours.grad - z_off.grad).abs().max()),
                }
            )
    return reporting.markdown_table(rows)


def table_total(lib: Any, inv_expr: str, total_expr: str) -> str:
    """Our `LeJEPALoss` total vs the official combination line, same inputs."""
    rows: list[dict[str, reporting.Cell]] = []
    for s in (1.0, 0.5):
        z = make_views(GAUSSIAN, s, seed=13)
        is_global = torch.ones(NUM_VIEWS, dtype=torch.bool)
        cfg = config.LossConfig(lambda_=LAMBDA)
        loss = lejepa_loss.LeJEPALoss(cfg, generator=torch.Generator().manual_seed(0))
        ours, _ = loss(batch_lib.ViewBatch(z, is_global))
        scope = {
            "sigreg_loss": official_stat(lib)(z.reshape(-1, DIM)),
            "inv_loss": eval(inv_expr, {"torch": torch}, {"proj": z}),  # noqa: S307
            "cfg": types.SimpleNamespace(lamb=LAMBDA),
        }
        theirs = eval(total_expr, {"torch": torch}, scope)  # noqa: S307
        rows.append(
            {
                "view noise s": s,
                "lambda": LAMBDA,
                "ours total": _f(ours),
                "official formula (same SIGReg convention)": _f(theirs),
                "rel. diff": abs(_f(ours) - _f(theirs)) / _f(theirs),
            }
        )
    return reporting.markdown_table(rows)


def table_precision(lib: Any) -> str:
    """Effect of the official example's bf16 autocast on the statistic."""
    rows: list[dict[str, reporting.Cell]] = []
    for name in ("N(0, I)", "radius x1.1"):
        z = INPUTS[name].sample(1536, seed=1)
        ours = _f(our_stat()(z))
        fp32 = _f(official_stat(lib)(z))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf16 = _f(official_stat(lib)(z))
            ours_ac = _f(our_stat()(z))
        rows.append(
            {
                "input (N=1536)": name,
                "ours fp32": ours,
                "ours inside bf16 autocast": ours_ac,
                "official fp32": fp32,
                "official inside bf16 autocast": bf16,
                "bf16 / fp32 (official)": bf16 / fp32,
            }
        )
    return reporting.markdown_table(rows)


def table_lambda(excess_ratio: float) -> str:
    """Lambda that gives the same SIGReg-excess / invariance balance."""

    def odds(lam: float) -> float:
        return lam / (1 - lam)

    def from_odds(o: float) -> float:
        return o / (1 + o)

    rows: list[dict[str, reporting.Cell]] = []
    for label, r in (
        ("theory (r = V = 6)", float(NUM_VIEWS)),
        ("measured", excess_ratio),
    ):
        rows.append(
            {
                "excess ratio r": f"{label}: {r:.3f}",
                "official lambda equivalent to our 0.05": from_odds(r * odds(LAMBDA)),
                "our lambda equivalent to official 0.05": from_odds(odds(LAMBDA) / r),
            }
        )
    return reporting.markdown_table(rows)


def main() -> None:
    """Runs all comparisons and writes the tables into the note."""
    parser = argparse.ArgumentParser(description="Compare SIGReg with LeJEPA.")
    parser.add_argument(
        "--official-dir",
        type=pathlib.Path,
        default=os.environ.get("LEJEPA_OFFICIAL_DIR"),
        help="Root of a clone of rbalestr-lab/lejepa (outside this repo).",
    )
    args = parser.parse_args()
    if args.official_dir is None:
        parser.error("pass --official-dir or set LEJEPA_OFFICIAL_DIR")
    root = pathlib.Path(args.official_dir).resolve()
    lib = load_official(root)
    minimal_sigreg, inv_expr, total_expr = minimal_snippets(root)
    ours = our_stat()
    floor = ours.noise_floor

    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance: list[dict[str, reporting.Cell]] = [
        {"item": "official commit", "value": official_commit(root)},
        {"item": "run datetime (UTC)", "value": now},
        {"item": "torch", "value": torch.__version__},
        {"item": "python", "value": platform.python_version()},
        {"item": "device", "value": "cpu"},
    ]
    sections = {
        "provenance": reporting.markdown_table(provenance),
        "quadrature": table_quadrature(lib, ours),
        "same_inputs": table_same_inputs(lib, floor),
        "noise_floor": table_floor(lib, floor),
        "training_layout": table_training_layout(lib, minimal_sigreg, floor),
        "invariance": table_invariance(inv_expr),
        "total": table_total(lib, inv_expr, total_expr),
        "precision": table_precision(lib),
    }
    # Measured excess ratio: radius x1.1 with independent views.
    z = make_views(INPUTS["radius x1.1"], 1.0, seed=7)
    excess = (_f(our_stat()(z.reshape(-1, DIM))) - floor) / (
        _f(official_stat(lib)(z)) - floor
    )
    sections["lambda"] = table_lambda(excess)
    for name, content in sections.items():
        reporting.update_section(NOTE, name, content)
        print(f"## {name}\n{content}\n")


if __name__ == "__main__":
    main()
