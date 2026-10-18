"""End-to-end smoke test: train -> embed -> analyze on tiny fake data (CPU).

Each app is exercised through its real `main`, with only the filesystem as the
interface between them, exactly as on the cluster.
"""

import importlib.util
import os
import pathlib
import types

import numpy
import pytest

from lejepa_norm import paths
from lejepa_norm.data import layout
from lejepa_norm.eval import dumps
from lejepa_norm.models import checkpoint

REPO = pathlib.Path(__file__).parent.parent
RUN = "smoke"
NOTE = "smoke_idea/smoke_experiment.md"
CLASSES = 3

SIZES = {
    ("stl10", "unlabeled"): 16,
    ("stl10", "train"): 12,
    ("stl10", "test"): 24,
    ("dtd", "test"): 12,
    ("svhn", "test"): 12,
}
CORRUPTIONS = {"gaussian_noise": (1, 3), "contrast": (1, 3)}

SMALL_MODEL = (
    "model.projector_hidden=[32]",
    "model.embed_dim=16",
    "loss.num_slices=16",
    "data.num_workers=0",
    "eval.batch_size=8",
    "eval.knn_k=3",
    "eval.num_views=3",
    "eval.num_bootstrap=20",
    "eval.max_kernel_images=100",
)
TRAIN_SETTINGS = (
    "train.batch_size=4",
    "train.epochs=2",
    "train.warmup_epochs=1",
    "train.log_every_steps=1",
    "train.knn_every_epochs=1",
    "train.checkpoint_every_epochs=1",
)


def _load_app(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_app", REPO / "apps" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path: pathlib.Path, array: numpy.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.save(path, array)


def _write_fake_data(root: pathlib.Path) -> None:
    rng = numpy.random.default_rng(0)

    def images(n: int) -> numpy.ndarray:
        return rng.integers(0, 256, size=(n, 96, 96, 3), dtype=numpy.uint8)

    for (dataset, split), n in SIZES.items():
        labels = numpy.arange(n, dtype=numpy.int64) % CLASSES
        if split == "unlabeled":
            labels[:] = -1
        _save(layout.images_path(root, dataset, split), images(n))
        _save(layout.labels_path(root, dataset, split), labels)
    for name, severities in CORRUPTIONS.items():
        for severity in severities:
            _save(
                layout.corrupted_images_path(root, name, severity),
                images(SIZES[("stl10", "test")]),
            )


@pytest.fixture
def workspace(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    for var, name in (
        (paths.DATA_DIR_VAR, "data"),
        (paths.RUNS_DIR_VAR, "runs"),
        (paths.EMBED_DIR_VAR, "embed"),
        (paths.NOTES_DIR_VAR, "notes"),
    ):
        monkeypatch.setenv(var, str(tmp_path / name))
    monkeypatch.delenv(paths.KALEIDO_CHROME_VAR, raising=False)
    # analyze looks for a `.env` upwards from the cwd; keep it away from the real one.
    monkeypatch.chdir(tmp_path)
    _write_fake_data(tmp_path / "data")
    return tmp_path


def _sets(*settings: str) -> list[str]:
    return [arg for item in settings for arg in ("--set", item)]


def test_train_embed_analyze_produces_a_note(workspace: pathlib.Path) -> None:
    base = str(REPO / "configs" / "base.yaml")
    run_dir = workspace / "runs" / RUN

    _load_app("train").main(
        [
            "--config",
            base,
            "--run-name",
            RUN,
            "--env-file",
            os.devnull,
            *_sets(*SMALL_MODEL, *TRAIN_SETTINGS),
        ]
    )
    last = run_dir / "last.pt"
    assert last.exists()
    assert checkpoint.load(last).epoch == 2

    assert (
        _load_app("embed").main(
            [
                "--config",
                base,
                "--run-name",
                RUN,
                "--checkpoint",
                str(last),
                "--env-file",
                os.devnull,
                *_sets(*SMALL_MODEL),
            ]
        )
        == 0
    )
    sets = dumps.load_run(workspace / "embed", RUN)
    assert set(sets) == set(dumps.EVAL_SETS)
    # center view + K augmented views, for every image of every set.
    assert sets[dumps.STL10_TEST].z.shape == (24, 4, 16)
    assert sets[dumps.STL10_C].z.shape[0] == 2 * 2 * 24

    assert (
        _load_app("analyze").main(
            [
                "--config",
                base,
                "--run-name",
                RUN,
                "--note",
                NOTE,
                "--no-figures",
                *_sets(*SMALL_MODEL),
            ]
        )
        == 0
    )
    note = (workspace / "notes" / NOTE).read_text()
    assert f"run `{RUN}`" in note
    assert note.count("| ") > 20  # generated markdown tables are present
    assert "nan" not in note.lower().replace("nanosecond", "")
