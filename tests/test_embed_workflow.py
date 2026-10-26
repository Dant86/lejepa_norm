"""End-to-end: fake checkpoint + fake prepared data -> apps/embed.py -> dumps."""

import collections.abc
import gc
import importlib.util
import pathlib
import types
import typing
import weakref

import numpy
import PIL.Image
import pytest
import torch

from lejepa_norm import config
from lejepa_norm import paths
from lejepa_norm.data import layout
from lejepa_norm.data import views
from lejepa_norm.eval import dumps
from lejepa_norm.eval import embedding
from lejepa_norm.eval import shards
from lejepa_norm.models import backbones
from lejepa_norm.models import checkpoint
from lejepa_norm.models import projector as projector_lib
import tiny_backbone

REPO = pathlib.Path(__file__).parent.parent
RUN = "fake_run"
K, D_Z, D_H = 3, 8, 512
SIZES = {
    "stl10_train": 7,
    "stl10_test": 6,
    "dtd_test": 5,
    "svhn_test": 4,
    "places_test": 3,
}
CORRUPTIONS = {"fog": (1, 3), "noise": (2,)}


def _load_app() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "embed_app", REPO / "apps" / "embed.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_array(path: pathlib.Path, array: numpy.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.save(path, array)


def _fake_data(root: pathlib.Path) -> None:
    rng = numpy.random.default_rng(0)

    def images(n: int) -> numpy.ndarray:
        return rng.integers(0, 256, size=(n, 96, 96, 3), dtype=numpy.uint8)

    splits = {
        ("stl10", "train"): SIZES["stl10_train"],
        ("stl10", "test"): SIZES["stl10_test"],
        ("dtd", "test"): SIZES["dtd_test"],
        ("svhn", "test"): SIZES["svhn_test"],
        ("places365", "val"): SIZES["places_test"],
    }
    for (dataset, split), n in splits.items():
        _save_array(layout.images_path(root, dataset, split), images(n))
        _save_array(
            layout.labels_path(root, dataset, split), rng.integers(0, 10, size=n)
        )
    for name, severities in CORRUPTIONS.items():
        for severity in severities:
            _save_array(
                layout.corrupted_images_path(root, name, severity),
                images(SIZES["stl10_test"]),
            )


def _save_checkpoint(path: pathlib.Path, model: config.ModelConfig) -> None:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    cfg = config.ExperimentConfig(model=model)
    backbone, feat_dim = backbones.build_backbone(cfg.model)
    head = projector_lib.build_projector(feat_dim, cfg.model)
    checkpoint.save(path, backbone, head, cfg, epoch=1)


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Fake prepared data and two checkpoints, built once and only read."""
    root = tmp_path_factory.mktemp("embed_shared")
    _fake_data(root / "data")
    # The cheap test backbone (default for the workflow tests) ...
    tiny = config.ModelConfig(
        backbone=tiny_backbone.NAME, projector_hidden=(16,), embed_dim=D_Z
    )
    _save_checkpoint(root / "tiny.pt", tiny)
    # ... and the real production ResNet-18, covered by one end-to-end test.
    resnet = config.ModelConfig(projector_hidden=(16,), embed_dim=D_Z)
    _save_checkpoint(root / "resnet.pt", resnet)
    return root


def _make_workspace(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    shared: pathlib.Path,
    checkpoint_name: str,
) -> pathlib.Path:
    """Per-test dirs; data and the checkpoint are symlinks to the shared ones."""
    (tmp_path / "data").symlink_to(shared / "data", target_is_directory=True)
    (tmp_path / "ckpt.pt").symlink_to(shared / checkpoint_name)
    for var, name in [
        (paths.DATA_DIR_VAR, "data"),
        (paths.RUNS_DIR_VAR, "runs"),
        (paths.EMBED_DIR_VAR, "embed"),
        (paths.NOTES_DIR_VAR, "notes"),
    ]:
        monkeypatch.setenv(var, str(tmp_path / name))
    return tmp_path


@pytest.fixture
def workspace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, shared: pathlib.Path
) -> pathlib.Path:
    """Workspace with the tiny test backbone."""
    return _make_workspace(tmp_path, monkeypatch, shared, "tiny.pt")


@pytest.fixture
def resnet_workspace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, shared: pathlib.Path
) -> pathlib.Path:
    """Workspace with the real ResNet-18 checkpoint."""
    return _make_workspace(tmp_path, monkeypatch, shared, "resnet.pt")


def _run(workspace: pathlib.Path, *extra: str) -> int:
    app = _load_app()
    return app.main(
        [
            "--config",
            str(REPO / "configs" / "base.yaml"),
            "--env-file",
            str(workspace / "no.env"),
            "--checkpoint",
            str(workspace / "ckpt.pt"),
            "--run-name",
            RUN,
            "--set",
            f"eval.num_views={K}",
            "--set",
            "eval.batch_size=4",
            "--set",
            "data.num_workers=0",
            *extra,
        ]
    )


def _dump(workspace: pathlib.Path) -> dict[str, shards.EmbeddingShard]:
    return dumps.load_run(workspace / "embed", RUN)


def test_embed_writes_loadable_dump_with_correct_contents(
    resnet_workspace: pathlib.Path,
) -> None:
    """Also the one embed test on the real ResNet-18 (production model path)."""
    workspace = resnet_workspace
    assert _run(workspace) == 0
    dump = _dump(workspace)
    assert set(dump) == set(dumps.EVAL_SETS)

    for name, n in SIZES.items():
        s = dump[name]
        assert s.z.shape == (n, K + 1, D_Z)
        assert s.h.shape == (n, K + 1, D_H)
        assert s.image_id.tolist() == list(range(n))
        assert set(s.corruption.tolist()) == {""}
        assert set(s.severity.tolist()) == {0}

    # stl10c: every (corruption, severity) present, ids index stl10_test.
    c = dump["stl10c"]
    n_test = SIZES["stl10_test"]
    variants = {(k, v) for k, vs in CORRUPTIONS.items() for v in vs}
    assert c.z.shape[0] == len(variants) * n_test
    seen = {
        (str(corr), int(sev))
        for corr, sev in zip(c.corruption, c.severity, strict=True)
    }
    assert seen == variants
    assert c.image_id.min() == 0 and c.image_id.max() == n_test - 1
    stl10_test_labels = numpy.load(
        layout.labels_path(workspace / "data", "stl10", "test")
    )
    numpy.testing.assert_array_equal(c.label, stl10_test_labels[c.image_id])

    # View 0 is exactly the model on the un-augmented center view.
    ckpt = checkpoint.load(workspace / "ckpt.pt")
    ckpt.backbone.eval()
    ckpt.projector.eval()
    center = views.CenterTransform(ckpt.cfg.data)
    raw = numpy.load(layout.images_path(workspace / "data", "stl10", "test"))
    batch = torch.stack(
        [center(PIL.Image.fromarray(img)) for img in raw]  # pyright: ignore[reportUnknownMemberType]
    )
    with torch.no_grad():
        h = ckpt.backbone(batch)
        z = ckpt.projector(h)
    s = dump["stl10_test"]
    numpy.testing.assert_allclose(s.h[:, shards.CENTER_VIEW], h.numpy(), atol=1e-3)
    numpy.testing.assert_allclose(s.z[:, shards.CENTER_VIEW], z.numpy(), atol=1e-3)

    # Augmented views differ from the center view and from each other.
    for k in range(1, K + 1):
        assert not numpy.allclose(s.z[:, k], s.z[:, shards.CENTER_VIEW], atol=1e-3)
    assert not numpy.allclose(s.z[:, 1], s.z[:, 2], atol=1e-3)


def test_embed_is_reproducible_and_seed_dependent(workspace: pathlib.Path) -> None:
    assert _run(workspace, "--sets", "stl10_test") == 0
    first = _dump(workspace)["stl10_test"]
    assert _run(workspace, "--sets", "stl10_test", "--overwrite") == 0
    again = _dump(workspace)["stl10_test"]
    numpy.testing.assert_array_equal(first.z, again.z)
    numpy.testing.assert_array_equal(first.h, again.h)

    # Views are made on the device from one generator per shard: the number of
    # workers does not matter at all ...
    workers = ("--set", "data.num_workers=2")
    assert _run(workspace, "--sets", "stl10_test", "--overwrite", *workers) == 0
    rerun = _dump(workspace)["stl10_test"]
    numpy.testing.assert_array_equal(first.z, rerun.z)
    numpy.testing.assert_array_equal(first.h, rerun.h)
    # ... and the shard's parameters are drawn up front, so the batch size only
    # changes float rounding inside the model.
    batch = ("--set", "eval.batch_size=3")
    assert _run(workspace, "--sets", "stl10_test", "--overwrite", *batch) == 0
    rerun = _dump(workspace)["stl10_test"]
    numpy.testing.assert_allclose(first.z, rerun.z, rtol=0, atol=1e-5)
    numpy.testing.assert_allclose(first.h, rerun.h, rtol=0, atol=1e-5)

    assert (
        _run(workspace, "--sets", "stl10_test", "--overwrite", "--set", "seed=1") == 0
    )
    other = _dump(workspace)["stl10_test"]
    numpy.testing.assert_array_equal(
        first.z[:, shards.CENTER_VIEW], other.z[:, shards.CENTER_VIEW]
    )
    assert not numpy.allclose(first.z[:, 1:], other.z[:, 1:], atol=1e-3)


def test_embed_refuses_existing_shards_without_overwrite(
    workspace: pathlib.Path,
) -> None:
    assert _run(workspace, "--sets", "svhn_test", "--max-images", "2") == 0
    with pytest.raises(FileExistsError):
        _run(workspace, "--sets", "svhn_test,dtd_test")
    assert not (workspace / "embed" / RUN / "dtd_test").exists()  # failed up front
    assert _run(workspace, "--sets", "svhn_test", "--overwrite") == 0
    assert _dump(workspace)["svhn_test"].z.shape[0] == SIZES["svhn_test"]


def test_embed_splits_into_sorted_shards_and_caps_images(
    workspace: pathlib.Path,
) -> None:
    app_cfg = config.ExperimentConfig()
    app_cfg.eval.num_views = 1
    app_cfg.eval.batch_size = 4
    app_cfg.data.num_workers = 0
    counts = embedding.embed_run(
        workspace / "ckpt.pt",
        app_cfg,
        workspace / "data",
        workspace / "embed",
        RUN,
        eval_sets=["stl10_train"],
        max_images=6,
        shard_size=4,
        device=torch.device("cpu"),
    )
    assert counts == {"stl10_train": 6}
    files = sorted((workspace / "embed" / RUN / "stl10_train").glob("*.npz"))
    assert [f.name for f in files] == ["part-00000.npz", "part-00001.npz"]
    assert [shards.load_shard(f).z.shape[0] for f in files] == [4, 2]
    loaded = dumps.load_set(files[0].parent)
    assert loaded is not None
    assert loaded.image_id.tolist() == list(range(6))


def test_embed_resume_writes_only_missing_shards(workspace: pathlib.Path) -> None:
    # Reference: an uninterrupted dump with several shards per set.
    def embed(run_name: str, **kwargs: bool) -> dict[str, int]:
        app_cfg = config.ExperimentConfig()
        app_cfg.eval.num_views = 2
        app_cfg.eval.batch_size = 4
        app_cfg.data.num_workers = 0
        return embedding.embed_run(
            workspace / "ckpt.pt",
            app_cfg,
            workspace / "data",
            workspace / "embed",
            run_name,
            eval_sets=["stl10_train", "stl10c"],
            shard_size=3,
            device=torch.device("cpu"),
            **kwargs,
        )

    full = embed("full")
    full_dir = workspace / "embed" / "full"

    # Interrupted job: some shards are complete, one was cut off mid-write (a
    # leftover `.partial` file), and the rest were never started.
    run_dir = workspace / "embed" / RUN
    kept: dict[pathlib.Path, int] = {}
    for name, part in (("stl10_train", 0), ("stl10c", 1), ("stl10c", 2)):
        src = full_dir / name / f"part-{part:05d}.npz"
        dest = run_dir / name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        kept[dest] = dest.stat().st_mtime_ns
    (run_dir / "stl10c" / "part-00000.npz.partial").write_bytes(b"truncated")

    # Without --resume or --overwrite, existing shards are still an error, and
    # the two flags cannot be combined.
    with pytest.raises(FileExistsError):
        embed(RUN)
    with pytest.raises(ValueError, match="mutually exclusive"):
        embed(RUN, resume=True, overwrite=True)
    with pytest.raises(SystemExit):
        _run(workspace, "--resume", "--overwrite")

    assert embed(RUN, resume=True) == full
    for path, mtime in kept.items():  # finished shards were not rewritten
        assert path.stat().st_mtime_ns == mtime
    for name in ("stl10_train", "stl10c"):
        want = sorted(p.name for p in (full_dir / name).glob("*.npz"))
        assert sorted(p.name for p in (run_dir / name).glob("*.npz")) == want
        for file_name in want:
            got = shards.load_shard(run_dir / name / file_name)
            ref = shards.load_shard(full_dir / name / file_name)
            for field in ("z", "h", "image_id", "label", "corruption", "severity"):
                numpy.testing.assert_array_equal(
                    getattr(got, field), getattr(ref, field), err_msg=field
                )

    # A shard from a different job (here: other images) is not silently kept.
    (run_dir / "stl10_train" / "part-00001.npz").write_bytes(
        (full_dir / "stl10_train" / "part-00002.npz").read_bytes()
    )
    with pytest.raises(ValueError, match="different job"):
        embed(RUN, resume=True)


def test_stl10c_arrays_are_loaded_one_at_a_time(
    workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arrays are read into RAM, so the 50 STL-10-C arrays must not all be alive."""
    real = typing.cast(
        collections.abc.Callable[..., object], embedding.datasets.load_corrupted
    )
    refs: list[weakref.ReferenceType[object]] = []

    def spy(*args: object, **kwargs: object) -> object:
        gc.collect()
        # At most the previous array (still held by the running loop) may be alive.
        assert sum(ref() is not None for ref in refs) <= 1
        dataset = real(*args, **kwargs)
        refs.append(weakref.ref(dataset))
        return dataset

    monkeypatch.setattr(embedding.datasets, "load_corrupted", spy)
    assert _run(workspace, "--sets", "stl10c", "--overwrite") == 0
    assert len(refs) == sum(len(v) for v in CORRUPTIONS.values())
