"""End-to-end: fake embedding dumps -> apps/analyze.py -> tables (+ PNGs)."""

import importlib.util
import json
import os
import pathlib
import types

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm.eval import dumps
from lejepa_norm.eval import shards

REPO = pathlib.Path(__file__).parent.parent
RUN = "fake_run"
NUM_CLASSES, K, D_Z, D_H = 4, 4, 16, 24
# Read at import: other tests may load `.env` files that leak into os.environ.
_CHROME_ENV = os.environ.get(paths.KALEIDO_CHROME_VAR)


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _views(
    rng: numpy.random.Generator,
    labels: numpy.ndarray,
    dim: int,
    log_scale: numpy.ndarray,
    noise: numpy.ndarray,
) -> numpy.ndarray:
    """Class-clustered directions, per-image radius, per-view jitter."""
    centers = numpy.random.default_rng(123).normal(size=(NUM_CLASSES, dim)) * 0.6
    n = labels.size
    base = centers[labels][:, None, :] + rng.normal(size=(n, 1, dim))
    views = base + noise[:, None, None] * rng.normal(size=(n, K + 1, dim))
    views[:, shards.CENTER_VIEW] = base[:, 0]
    views /= numpy.linalg.norm(views, axis=-1, keepdims=True)
    log_r = numpy.log(numpy.sqrt(dim)) + log_scale[:, None]
    log_r = log_r + 0.05 * rng.normal(size=(n, K + 1))
    return (views * numpy.exp(log_r)[..., None]).astype(numpy.float32)


def _shard(
    rng: numpy.random.Generator,
    labels: numpy.ndarray,
    image_id: numpy.ndarray,
    corruption: list[str],
    severity: numpy.ndarray,
) -> shards.EmbeddingShard:
    # Radius shrinks and views get noisier with severity (known answer).
    log_scale = 0.3 * rng.normal(size=labels.size) - 0.15 * severity
    noise = 0.3 + 0.2 * severity
    return shards.EmbeddingShard(
        z=_views(rng, labels, D_Z, log_scale, noise),
        h=_views(rng, labels, D_H, log_scale, noise),
        image_id=image_id.astype(numpy.int64),
        label=labels.astype(numpy.int64),
        corruption=numpy.array(corruption),
        severity=severity.astype(numpy.int8),
    )


def _write_fake_run(embed_dir: pathlib.Path, runs_dir: pathlib.Path) -> None:
    rng = numpy.random.default_rng(0)
    n = 120
    zeros = numpy.zeros(n, dtype=numpy.int64)
    test_labels = rng.integers(0, NUM_CLASSES, n)
    sets = {
        dumps.STL10_TRAIN: _shard(
            rng, rng.integers(0, NUM_CLASSES, n), numpy.arange(n), [""] * n, zeros
        ),
        dumps.STL10_TEST: _shard(rng, test_labels, numpy.arange(n), [""] * n, zeros),
        dumps.DTD_TEST: _shard(
            rng,
            rng.integers(0, NUM_CLASSES, 40),
            numpy.arange(40),
            [""] * 40,
            numpy.full(40, 3),
        ),
        dumps.SVHN_TEST: _shard(
            rng,
            rng.integers(0, NUM_CLASSES, 40),
            numpy.arange(40),
            [""] * 40,
            numpy.full(40, 5),
        ),
    }
    for name, shard in sets.items():
        out = dumps.set_dir(embed_dir, RUN, name)
        out.mkdir(parents=True)
        shards.save_shard(out / "shard_000.npz", shard)
    # STL-10-C split across two shard files to exercise concatenation.
    corrupt_dir = dumps.set_dir(embed_dir, RUN, dumps.STL10_C)
    corrupt_dir.mkdir(parents=True)
    ids = numpy.arange(60)
    for name in ("contrast", "pixelate"):
        for level in (1, 3, 5):
            shards.save_shard(
                corrupt_dir / f"{name}_{level}.npz",
                _shard(
                    rng,
                    test_labels[ids],
                    ids,
                    [name] * ids.size,
                    numpy.full(ids.size, level),
                ),
            )
    run_dir = runs_dir / RUN
    run_dir.mkdir(parents=True)
    with (run_dir / "metrics.jsonl").open("w") as f:
        for step in range(5):
            f.write(json.dumps({"step": step, "kappa_z": 1.0 + 1 / (step + 1)}) + "\n")


def _cell(table_line: str, column: int) -> float:
    return float(table_line.split("|")[column + 1].split()[0])


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Fake embedding dumps and metrics, written once; the tests only read them."""
    root = tmp_path_factory.mktemp("fake_run")
    _write_fake_run(root / "embed", root / "runs")
    return root


def _setup_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, fake_run: pathlib.Path
) -> None:
    # Empty means "unset" for load_paths and blocks values from any `.env` file.
    monkeypatch.setenv(paths.KALEIDO_CHROME_VAR, _CHROME_ENV or "")
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    monkeypatch.setenv(paths.RUNS_DIR_VAR, str(fake_run / "runs"))
    monkeypatch.setenv(paths.EMBED_DIR_VAR, str(fake_run / "embed"))
    monkeypatch.setenv(paths.NOTES_DIR_VAR, str(tmp_path / "notes"))
    monkeypatch.chdir(tmp_path)  # no stray .env is picked up


def _argv(*extra: str) -> list[str]:
    return [
        "--config",
        str(REPO / "configs" / "base.yaml"),
        "--run-name",
        RUN,
        "--note",
        "idea/exp.md",
        "--set",
        "eval.num_bootstrap=10",
        "--set",
        "eval.knn_k=5",
        "--set",
        "eval.max_kernel_images=100",
        *extra,
    ]


def test_analyze_writes_tables_into_note(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, fake_run: pathlib.Path
) -> None:
    _setup_env(tmp_path, monkeypatch, fake_run)
    argv = _argv("--no-figures")
    app = _load_app()
    assert app.main(argv) == 0
    note = tmp_path / "notes" / "idea" / "exp.md"
    first = note.read_text()
    assert app.main(argv) == 0
    assert note.read_text() == first  # re-running is idempotent

    for name in (
        "inputs",
        "marginal",
        "icc",
        "redundancy",
        "difficulty",
        "radius_shift",
        "downstream",
        "selective",
        "ood",
        "retrieval",
    ):
        assert f"BEGIN GENERATED: analyze.{name}" in first, name
    assert "_No rows._" not in first
    lines = first.splitlines()

    # Paired radius shift is negative and grows with severity (by construction).
    shift = [ln for ln in lines if ln.startswith("| z | contrast |")]
    means = [_cell(ln, 3) for ln in shift]
    assert len(means) == 3 and means[0] > means[1] > means[2] and means[0] < 0

    # Severity lowers the radius: raw Spearman with rho_bar is negative.
    severity_row = next(ln for ln in lines if ln.startswith("| z | severity |"))
    assert _cell(severity_row, 2) < -0.3

    # Classes overlap but are clustered: the probe on u_hat beats chance (0.25).
    probe_row = next(ln for ln in lines if ln.startswith("| z | augmented | u_hat |"))
    assert _cell(probe_row, 3) > 0.5

    assert list(note.parent.glob("*.png")) == []  # --no-figures


def test_analyze_redundancy_controls_mode_writes_only_its_section(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, fake_run: pathlib.Path
) -> None:
    _setup_env(tmp_path, monkeypatch, fake_run)
    argv = _argv("--redundancy-controls", "--set", "eval.num_permutations=8")
    app = _load_app()
    assert app.main(argv) == 0
    note = tmp_path / "notes" / "idea" / "exp.md"
    text = note.read_text()
    assert "BEGIN GENERATED: analyze.redundancy_controls" in text
    assert "BEGIN GENERATED: analyze.marginal" not in text  # nothing else runs
    assert list(note.parent.glob("*.png")) == []
    rows = [ln for ln in text.splitlines() if ln.startswith(("| z |", "| h |"))]
    # Per space: same views, two half-to-half, two single-view setups + reliability.
    assert len(rows) == 2 * 6
    same_views = next(ln for ln in rows if ln.startswith("| z | same views"))
    assert "| 8 |" in same_views  # permutations
    assert app.main(argv) == 0
    assert note.read_text() == text  # idempotent


def test_analyze_aggregates_controls_notes_per_lambda(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, fake_run: pathlib.Path
) -> None:
    _setup_env(tmp_path, monkeypatch, fake_run)
    app = _load_app()
    assert (
        app.main(_argv("--redundancy-controls", "--set", "eval.num_permutations=8"))
        == 0
    )
    # The run needs a resolved config for the aggregation to find its lambda.
    cfg = config.load_config(REPO / "configs" / "base.yaml", ["loss.lambda=0.05"])
    config.save_resolved(cfg, fake_run / "runs" / RUN)
    note = tmp_path / "notes" / "idea" / "summary.md"
    argv = [
        "--aggregate-controls",
        "idea/exp.md",
        "idea/exp.md",
        "--note",
        "idea/summary.md",
    ]
    assert app.main(argv) == 0
    text = note.read_text()
    assert "analyze.redundancy_controls_aggregate" in text
    rows = [ln for ln in text.splitlines() if ln.startswith("| 0.05 |")]
    assert len(rows) == 2 * 6
    assert all("| 2 |" in ln for ln in rows)  # two "seeds" (the same note twice)
    assert app.main(argv) == 0
    assert note.read_text() == text  # idempotent


def test_analyze_transfer_mode_and_aggregation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, fake_run: pathlib.Path
) -> None:
    _setup_env(tmp_path, monkeypatch, fake_run)
    app = _load_app()
    argv = _argv("--transfer", "--no-figures")
    assert app.main(argv) == 0
    note = tmp_path / "notes" / "idea" / "exp.md"
    text = note.read_text()
    assert "BEGIN GENERATED: analyze.radius_transfer" in text
    assert "BEGIN GENERATED: analyze.marginal" not in text  # nothing else runs
    assert "h is not analyzed; LeJEPA's product is z." in text
    rows = [
        ln
        for ln in text.splitlines()
        if ln.startswith(("| contrast |", "| pixelate |"))
    ]
    assert len(rows) == 6  # two corruptions x three severities
    assert any(ln.startswith("| clean (reference) |") for ln in text.splitlines())
    assert list(note.parent.glob("*.png")) == []
    assert app.main(argv) == 0
    assert note.read_text() == text  # idempotent

    cfg = config.load_config(REPO / "configs" / "base.yaml", ["loss.lambda=0.05"])
    config.save_resolved(cfg, fake_run / "runs" / RUN)
    summary = tmp_path / "notes" / "idea" / "summary.md"
    agg = ["--aggregate-transfer", "idea/exp.md", "idea/exp.md", "--note"]
    assert app.main([*agg, "idea/summary.md"]) == 0
    lines = summary.read_text().splitlines()
    agg_rows = [ln for ln in lines if ln.startswith("| 0.05 |")]
    assert len(agg_rows) == 7 and all("| 2 |" in ln for ln in agg_rows)


def test_analyze_redundancy_controls_needs_a_run() -> None:
    with pytest.raises(SystemExit):
        _load_app().main(["--sweep", "x", "--note", "n.md", "--redundancy-controls"])


@pytest.mark.slow  # renders PNGs through headless Chrome
def test_analyze_renders_figures_into_note(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, fake_run: pathlib.Path
) -> None:
    if not plotting.chrome_available(
        pathlib.Path(_CHROME_ENV) if _CHROME_ENV else None
    ):
        pytest.skip("Chrome for Kaleido is not available")
    _setup_env(tmp_path, monkeypatch, fake_run)
    assert _load_app().main(_argv()) == 0
    note = tmp_path / "notes" / "idea" / "exp.md"
    pngs = sorted(p.name for p in note.parent.glob("*.png"))
    assert "exp_radius_hist_z.png" in pngs
    assert "exp_radius_shift_h.png" in pngs
    assert "exp_kappa_training.png" in pngs
    assert "![radius histogram of z](exp_radius_hist_z.png)" in note.read_text()
