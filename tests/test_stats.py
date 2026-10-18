import importlib.util
import json
import pathlib
import sys

import numpy
import pytest

from lejepa_norm import paths
from lejepa_norm.data import layout
from lejepa_norm.data import stats

REPO = pathlib.Path(__file__).parent.parent


def _write_split(root: pathlib.Path, split: str, images: numpy.ndarray) -> None:
    path = layout.images_path(root, "stl10", split)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.save(path, images)


@pytest.fixture
def data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    rng = numpy.random.default_rng(0)
    # Different channel distributions, so a channel mix-up would be caught.
    scale = numpy.array([255, 200, 120])

    def images(n: int) -> numpy.ndarray:
        return (rng.random((n, 8, 8, 3)) * scale).astype(numpy.uint8)

    _write_split(tmp_path, "unlabeled", images(30))
    _write_split(tmp_path, "train", images(10))
    # `test` must never contribute: make it wildly different.
    _write_split(tmp_path, "test", numpy.full((10, 8, 8, 3), 255, dtype=numpy.uint8))
    return tmp_path


def test_stats_match_direct_computation_and_ignore_test_split(
    data_dir: pathlib.Path,
) -> None:
    pretraining = (
        numpy.concatenate(
            [
                numpy.load(layout.images_path(data_dir, "stl10", "unlabeled")),
                numpy.load(layout.images_path(data_dir, "stl10", "train")),
            ]
        ).astype(numpy.float64)
        / 255.0
    )
    mean, std = stats.channel_stats(data_dir, chunk_size=7)  # uneven chunking
    numpy.testing.assert_allclose(mean, pretraining.mean(axis=(0, 1, 2)), rtol=1e-12)
    numpy.testing.assert_allclose(std, pretraining.std(axis=(0, 1, 2)), rtol=1e-9)


def test_app_stats_stage_writes_json_and_prints_override(
    data_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data_dir))
    for var in (paths.RUNS_DIR_VAR, paths.EMBED_DIR_VAR, paths.NOTES_DIR_VAR):
        monkeypatch.setenv(var, str(data_dir / "unused"))
    monkeypatch.setattr(
        sys, "argv", ["prepare_data", "--stage", "stats", "--env-file", "/dev/null"]
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_data_app", REPO / "apps" / "prepare_data.py"
    )
    assert spec is not None and spec.loader is not None
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.main()

    written = json.loads((data_dir / "stl10" / stats.STATS_FILE).read_text())
    mean, std = stats.channel_stats(data_dir)
    assert written["mean"] == pytest.approx(mean)
    assert written["std"] == pytest.approx(std)
    assert "--set data.mean=[" in capsys.readouterr().out
