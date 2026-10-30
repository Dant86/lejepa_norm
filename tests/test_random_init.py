"""Random-init baseline: checkpoint contract, BatchNorm modes, embed app mode."""

import importlib.util
import pathlib
import types

import numpy
import pytest
import torch

import helpers
from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.eval import dumps
from lejepa_norm.models import checkpoint
from lejepa_norm.models import random_init

REPO = pathlib.Path(__file__).parent.parent
SETTINGS = [s.split("=", 1) for s in helpers.TINY_MODEL]
CPU = torch.device("cpu")


def _cfg() -> config.ExperimentConfig:
    return config.load_config(
        REPO / "configs" / "base.yaml", [f"{k}={v}" for k, v in SETTINGS]
    )


def _stats(model: torch.nn.Module) -> list[torch.Tensor]:
    return [
        t
        for m in model.modules()
        if isinstance(m, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d)
        for t in (m.running_mean, m.running_var)
        if t is not None
    ]


def _weights(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _write(
    path: pathlib.Path, seed: int, mode: str, data_dir: pathlib.Path
) -> checkpoint.Checkpoint:
    random_init.write_checkpoint(
        path,
        _cfg(),
        seed,
        mode,  # pyright: ignore[reportArgumentType]
        data_dir,
        num_calibration_images=8,
        device=CPU,
    )
    return checkpoint.load(path)


def test_calibration_sets_running_stats_without_touching_weights(
    fake_stl10_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    cal = _write(tmp_path / "cal.pt", 3, "calibrated", fake_stl10_dir)
    ident = _write(tmp_path / "id.pt", 3, "identity", fake_stl10_dir)
    assert cal.epoch == ident.epoch == 0
    # Same initialization seed -> identical weights in both modes.
    for a, b in zip(
        _weights(cal.backbone) + _weights(cal.projector),
        _weights(ident.backbone) + _weights(ident.projector),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    # Identity mode keeps the initial statistics (mean 0, var 1); calibration moves
    # them to the data statistics.
    init = _stats(ident.backbone)
    for i, t in enumerate(init):
        expected = torch.zeros_like(t) if i % 2 == 0 else torch.ones_like(t)
        assert torch.equal(t, expected)
    moved = _stats(cal.backbone)
    assert not any(torch.allclose(m, i) for m, i in zip(moved, init, strict=True))


def test_seeds_differ_and_calibration_is_deterministic(
    fake_stl10_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    flat: list[torch.Tensor] = []
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        ckpt = _write(tmp_path / f"{name}.pt", seed, "calibrated", fake_stl10_dir)
        flat.append(
            torch.cat(
                [t.flatten() for t in _stats(ckpt.backbone) + _weights(ckpt.backbone)]
            )
        )
    torch.testing.assert_close(flat[0], flat[1])
    assert not torch.allclose(flat[0], flat[2])


def test_embed_app_random_init_writes_checkpoint_and_dump(
    fake_stl10_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in [
        (paths.DATA_DIR_VAR, fake_stl10_dir),
        (paths.RUNS_DIR_VAR, tmp_path / "runs"),
        (paths.EMBED_DIR_VAR, tmp_path / "embed"),
        (paths.NOTES_DIR_VAR, tmp_path / "notes"),
    ]:
        monkeypatch.setenv(var, str(value))
    spec = importlib.util.spec_from_file_location(
        "embed_app", REPO / "apps" / "embed.py"
    )
    assert spec is not None and spec.loader is not None
    app: types.ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    overrides = [x for k, v in SETTINGS for x in ("--set", f"{k}={v}")]
    argv = [
        "--config",
        str(REPO / "configs" / "base.yaml"),
        "--env-file",
        str(tmp_path / "no.env"),
        "--random-init",
        "2",
        "--run-name",
        "rand",
        "--sets",
        "stl10_test,stl10_train",
        "--set",
        "eval.num_views=2",
        "--set",
        "eval.batch_size=4",
        *overrides,
    ]
    assert app.main(argv) == 0
    ckpt = tmp_path / "runs" / "rand" / "last.pt"
    assert checkpoint.load(ckpt).epoch == 0
    dump = dumps.load_run(tmp_path / "embed", "rand")
    assert dump["stl10_test"].z.shape == (4, 3, 16)
    assert numpy.isfinite(dump["stl10_train"].z).all()
    # A rerun reuses the checkpoint (identical weights) instead of redrawing it.
    before = ckpt.stat().st_mtime_ns
    assert app.main([*argv, "--overwrite"]) == 0
    assert ckpt.stat().st_mtime_ns == before
    with pytest.raises(SystemExit):  # exactly one of --checkpoint / --random-init
        app.main(["--run-name", "x"])
