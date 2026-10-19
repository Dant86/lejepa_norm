"""Sweep summaries: runs grouped by lambda from resolved configs and metrics logs."""

import importlib.util
import json
import os
import pathlib
import types

import pytest

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm import plotting
from lejepa_norm.eval import sweep

REPO = pathlib.Path(__file__).parent.parent
_CHROME_ENV = os.environ.get(paths.KALEIDO_CHROME_VAR)
LAMBDAS = (0.02, 0.1)
SEEDS = (0, 1, 2)
TARGET_EPOCHS = 3
PARTIAL = (0.1, 2)  # this run stops after epoch 2


def _kappa(lam: float, seed: int, epoch: int) -> float:
    return (20.0 if lam == 0.02 else 8.0) + seed - epoch


def _write_run(
    runs: pathlib.Path, name: str, lam: float, seed: int, epochs: int
) -> None:
    cfg = config.ExperimentConfig(seed=seed)
    cfg.loss.lambda_ = lam
    cfg.train.epochs = TARGET_EPOCHS
    config.save_resolved(cfg, runs / name)
    lines = [json.dumps({"kind": "step", "step": 1, "epoch": 1, "kappa_z": 99.0})]
    for epoch in range(1, epochs + 1):
        record: dict[str, object] = {
            "kind": "epoch",
            "step": 10 * epoch,
            "epoch": epoch,
            "kappa_z_epoch_mean": _kappa(lam, seed, epoch),
            "z_norm_mean_epoch_mean": 14.0 + epoch,
            "z_norm_std_epoch_mean": 2.0,
            "loss_dist_epoch_mean": 5.0,
        }
        if epoch in (1, 3):  # k-NN is only logged on some epochs
            record["knn_acc_h"] = 0.1 * epoch
        lines.append(json.dumps(record))
    (runs / name / "metrics.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture
def runs_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    runs = tmp_path / "runs"
    for lam in LAMBDAS:
        for seed in SEEDS:
            epochs = 2 if (lam, seed) == PARTIAL else TARGET_EPOCHS
            _write_run(runs, f"exp_lam{lam:g}_s{seed}", lam, seed, epochs)
    _write_run(runs, "other_lam0.05_s0", 0.05, 0, TARGET_EPOCHS)  # different prefix
    (runs / "exp_lam9_s9").mkdir()  # no resolved config: ignored
    return runs


def test_runs_are_grouped_by_the_lambda_in_their_resolved_config(
    runs_dir: pathlib.Path,
) -> None:
    runs = sweep.load_runs(runs_dir, "exp")
    assert [r.name for r in runs] == [
        f"exp_lam{lam:g}_s{seed}" for lam in LAMBDAS for seed in SEEDS
    ]
    groups = sweep.by_lambda(runs)
    assert list(groups) == [0.02, 0.1]
    assert all(len(group) == 3 for group in groups.values())
    assert {r.target_epochs for r in runs} == {TARGET_EPOCHS}
    assert [r.complete for r in runs] == [True] * 5 + [False]


def test_curves_use_only_epochs_every_seed_has_reached(
    runs_dir: pathlib.Path,
) -> None:
    groups = sweep.by_lambda(sweep.load_runs(runs_dir, "exp"))
    full = sweep.curve(groups[0.02], sweep.KAPPA_Z)
    assert full.epochs == [1.0, 2.0, 3.0]
    # kappa(lambda=0.02) = 20 + seed - epoch, seeds 0..2 -> mean 20 + 1 - epoch.
    assert full.mean == pytest.approx([20.0, 19.0, 18.0])
    assert full.low == pytest.approx([19.0, 18.0, 17.0])
    assert full.high == pytest.approx([21.0, 20.0, 19.0])
    # One lambda=0.1 seed stopped after epoch 2, which caps the common epochs.
    partial = sweep.curve(groups[0.1], sweep.KAPPA_Z)
    assert partial.epochs == [1.0, 2.0]
    # k-NN exists at epochs 1 and 3, but not at 3 for the partial run.
    assert sweep.curve(groups[0.1], "knn_acc_h").epochs == [1.0]
    assert sweep.curve(groups[0.02], "knn_acc_h").epochs == [1.0, 3.0]


def test_summary_and_run_tables_report_targets_and_progress(
    runs_dir: pathlib.Path,
) -> None:
    runs = sweep.load_runs(runs_dir, "exp")
    target, low_lam, high_lam = sweep.summary_rows(runs)
    assert target["z_norm_mean"] == "15.98"  # sqrt(256 - 1/2)
    assert target["kappa_z"] == "1"
    assert (low_lam["lambda"], low_lam["seeds"], low_lam["epoch"]) == ("0.02", 3, 3)
    assert low_lam["kappa_z"] == "18 ± 1"  # epoch 3: 17, 18, 19
    assert (high_lam["lambda"], high_lam["epoch"]) == ("0.1", 2)  # capped by PARTIAL

    rows = sweep.run_rows(runs)
    status = {str(r["run"]): r["status"] for r in rows}
    assert status["exp_lam0.1_s2"] == "partial"
    assert status["exp_lam0.02_s0"] == "complete"
    partial = next(r for r in rows if r["run"] == "exp_lam0.1_s2")
    assert partial["epochs"] == "2/3"
    assert partial["knn_acc_h"] == pytest.approx(0.1)  # latest value, from epoch 1


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_app", REPO / "apps" / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, runs_dir: pathlib.Path
) -> None:
    monkeypatch.setenv(paths.RUNS_DIR_VAR, str(runs_dir))
    monkeypatch.setenv(paths.NOTES_DIR_VAR, str(tmp_path / "notes"))
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    monkeypatch.setenv(paths.EMBED_DIR_VAR, str(tmp_path / "embed"))
    if _CHROME_ENV:
        monkeypatch.setenv(paths.KALEIDO_CHROME_VAR, _CHROME_ENV)
    else:
        monkeypatch.delenv(paths.KALEIDO_CHROME_VAR, raising=False)
    monkeypatch.chdir(tmp_path)  # keep a real `.env` out of the lookup


def test_analyze_sweep_mode_writes_tables_idempotently(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, runs_dir: pathlib.Path
) -> None:
    _setup_env(tmp_path, monkeypatch, runs_dir)
    argv = ["--sweep", "exp", "--note", "idea/sweep.md", "--no-figures"]
    app = _load_app()
    assert app.main(argv) == 0
    note = tmp_path / "notes" / "idea" / "sweep.md"
    text = note.read_text()
    assert "target (N(0, I))" in text and "15.98" in text
    assert "| 0.02 | 3 | 3 | 18 ± 1 |" in text  # lambda, seeds, epoch, kappa_z
    assert "| 0.1 | 3 | 2 |" in text  # capped at epoch 2 by the partial run
    assert "partial" in text and "exp_lam0.1_s2" in text
    assert "other_lam" not in text  # other prefixes stay out
    assert list(note.parent.glob("*.png")) == []  # --no-figures

    assert app.main(argv) == 0
    assert note.read_text() == text  # re-running rewrites the sections in place


def test_analyze_sweep_mode_fails_when_no_run_matches(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, runs_dir: pathlib.Path
) -> None:
    _setup_env(tmp_path, monkeypatch, runs_dir)
    argv = ["--sweep", "nothing", "--note", "idea/sweep.md", "--no-figures"]
    assert _load_app().main(argv) == 1
    assert not (tmp_path / "notes").exists()


@pytest.mark.slow  # renders PNGs through headless Chrome
def test_analyze_sweep_mode_renders_curves(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, runs_dir: pathlib.Path
) -> None:
    if not plotting.chrome_available(
        pathlib.Path(_CHROME_ENV) if _CHROME_ENV else None
    ):
        pytest.skip("Chrome for Kaleido is not available")
    _setup_env(tmp_path, monkeypatch, runs_dir)
    argv = ["--sweep", "exp", "--note", "idea/sweep.md"]
    assert _load_app().main(argv) == 0
    note = tmp_path / "notes" / "idea" / "sweep.md"
    pngs = sorted(p.name for p in note.parent.glob("*.png"))
    assert "sweep_curve_kappa_z.png" in pngs
    assert "sweep_curve_z_norm_mean.png" in pngs
    assert "![" in note.read_text()
