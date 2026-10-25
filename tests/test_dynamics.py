"""Epoch sweep: planted onset of the redundancy is found, at the right epoch."""

import importlib.util
import pathlib

import numpy
import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import dumps
from lejepa_norm.eval import dynamics
from lejepa_norm.eval import features
from lejepa_norm.eval import shards

N, K, D = 200, 4, 8


def _quiet(message: str) -> None:
    del message


CFG = config.EvalConfig(cv_folds=5, max_kernel_images=N)


def _shard(dependence: float, seed: int) -> shards.EmbeddingShard:
    """Test dump whose log-radius follows the direction with weight `dependence`.

    Views are `(N, K + 1, D)` with the center view first.
    """
    rng = numpy.random.default_rng(seed)
    base = features.direction(rng.standard_normal((N, 1, D)))
    unit = features.direction(base + 0.1 * rng.standard_normal((N, K + 1, D)))
    smooth = numpy.sin(3 * base[:, 0, 0]) + 4 * base[:, 0, 1] * base[:, 0, 2]
    own = rng.standard_normal(N)
    rho = dependence * smooth + (1 - dependence) * own
    rho = rho[:, None] + 0.05 * rng.standard_normal((N, K + 1))
    z = (unit * numpy.exp(rho)[..., None]).astype(numpy.float32)
    return shards.EmbeddingShard(
        z=z,
        h=z.copy(),
        image_id=numpy.arange(N),
        label=numpy.zeros(N, dtype=numpy.int64),
        corruption=numpy.array([""] * N),
        severity=numpy.zeros(N, dtype=numpy.int8),
    )


def _write(
    embed_dir: pathlib.Path, run: str, epoch: int, dep: float, seed: int
) -> None:
    directory = dumps.set_dir(embed_dir, dynamics.dump_name(run, epoch), "stl10_test")
    directory.mkdir(parents=True)
    shards.save_shard(directory / "part-00000.npz", _shard(dep, seed))


def test_sweep_tracks_planted_onset_and_aggregates_over_seeds(
    tmp_path: pathlib.Path,
) -> None:
    runs = ("run_s0", "run_s1")
    for i, run in enumerate(runs):
        # Independent at epoch 10, strongly dependent at epoch 20.
        _write(tmp_path, run, 10, 0.0, 10 + i)
        _write(tmp_path, run, 20, 1.0, 20 + i)
    assert dynamics.available_epochs(tmp_path, runs[0]) == [10, 20]

    results = {
        run: dynamics.run_epoch_sweep(tmp_path, run, CFG, 0, 3, progress=_quiet)
        for run in runs
    }
    for rows in results.values():
        early, late = rows
        assert early["epoch"] == 10 and late["epoch"] == 20
        assert float(early["r2_same"]) < 0.2
        assert float(late["r2_same"]) > 0.7
        assert float(late["r2_cross"]) > 0.5
        assert float(late["reliability"]) > 0.9
        assert float(late["null_max"]) < 0.2

    table = dynamics.aggregate(results)
    assert [r["epoch"] for r in table] == [10, 20]
    assert all(r["seeds"] == 2 and "±" in str(r["r2_same"]) for r in table)
    curve = dynamics.curve(results, "r2_same")
    assert curve.epochs == [10, 20] and curve.mean[1] > curve.mean[0] + 0.5


def test_unit_is_resumable_and_settings_are_part_of_the_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "run_s0", 10, 1.0, 1)
    messages: list[str] = []
    first = dynamics.run_unit(tmp_path, "run_s0", 10, CFG, 0, 2, messages.append)
    assert dynamics.unit_path(tmp_path, "run_s0", 10).is_file()
    assert any("setup" in m for m in messages) and "done" in messages[-1]

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("recomputed a finished unit")

    monkeypatch.setattr(dynamics, "epoch_row", boom)
    again = dynamics.run_unit(tmp_path, "run_s0", 10, CFG, 0, 2, messages.append)
    assert again == first and "skipped" in messages[-1]
    with pytest.raises(AssertionError):  # other settings: not reused
        dynamics.run_unit(tmp_path, "run_s0", 10, CFG, 0, 3, _quiet)
    # A torn shard is recomputed rather than trusted.
    dynamics.unit_path(tmp_path, "run_s0", 10).write_text("{")
    with pytest.raises(AssertionError):
        dynamics.run_unit(tmp_path, "run_s0", 10, CFG, 0, 2, _quiet)


def test_collect_reads_shards_and_reports_missing_units(
    tmp_path: pathlib.Path,
) -> None:
    for run in ("run_s0", "run_s1"):
        _write(tmp_path, run, 10, 0.0, 1)
        _write(tmp_path, run, 20, 1.0, 2)
    for epoch in (10, 20):
        dynamics.run_unit(tmp_path, "run_s0", epoch, CFG, 0, 2, _quiet)
    dynamics.run_unit(tmp_path, "run_s1", 10, CFG, 0, 2, _quiet)
    messages: list[str] = []
    runs = dynamics.collect(tmp_path, ["run_s0", "run_s1"], CFG, 0, 2, messages.append)
    assert [r["epoch"] for r in runs["run_s0"]] == [10, 20]
    assert [r["epoch"] for r in runs["run_s1"]] == [10]
    assert messages == ["run_s1 epoch 20: no valid shard, left out"]
    with pytest.raises(FileNotFoundError):
        dynamics.collect(tmp_path, ["run_s0"], CFG, 0, 5, _quiet)  # other settings


def test_checkpoint_epoch_parses_periodic_names_only() -> None:
    assert dynamics.checkpoint_epoch("runs/x/epoch_0030.pt") == 30
    with pytest.raises(ValueError):
        dynamics.checkpoint_epoch("last.pt")


def test_sweep_without_dumps_fails(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        dynamics.run_epoch_sweep(tmp_path, "nope", CFG, 0, 0, progress=_quiet)


def test_analyze_epoch_sweep_mode_writes_tables(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i, run in enumerate(("run_s0", "run_s1")):
        _write(tmp_path / "embed", run, 10, 0.0, 10 + i)
        _write(tmp_path / "embed", run, 20, 1.0, 20 + i)
    monkeypatch.setenv(paths.KALEIDO_CHROME_VAR, "")
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    monkeypatch.setenv(paths.RUNS_DIR_VAR, str(tmp_path / "runs"))
    monkeypatch.setenv(paths.EMBED_DIR_VAR, str(tmp_path / "embed"))
    monkeypatch.setenv(paths.NOTES_DIR_VAR, str(tmp_path / "notes"))
    monkeypatch.chdir(tmp_path)  # no stray .env is picked up
    repo = pathlib.Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        "analyze_app", repo / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    common = [
        "--config",
        str(repo / "configs" / "base.yaml"),
        "--set",
        "eval.num_permutations=2",
        "--set",
        f"eval.max_kernel_images={N}",
    ]
    for run in ("run_s0", "run_s1"):
        for epoch in ("10", "20"):
            assert app.main([*common, "--epoch-unit", run, epoch]) == 0
    assert not (tmp_path / "notes").exists()  # units write shards, not notes
    argv = [
        *common,
        "--epoch-sweep",
        "run_s0",
        "run_s1",
        "--note",
        "idea/sweep.md",
        "--no-figures",
    ]
    assert app.main(argv) == 0
    text = (tmp_path / "notes" / "idea" / "sweep.md").read_text()
    for section in ("dynamics.table", "dynamics.run.run_s0", "dynamics.run.run_s1"):
        assert f"BEGIN GENERATED: {section}" in text
    assert "| 10 | 2 |" in text and "| 20 | 2 |" in text
