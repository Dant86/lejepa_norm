"""Shared fixtures; importing `helpers` registers the test-only backbone."""

import pathlib

import pytest

import helpers


@pytest.fixture(scope="session")
def fake_stl10_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Prepared fake STL-10 shared by training tests. Tests must only read it."""
    root = tmp_path_factory.mktemp("fake_stl10")
    helpers.write_fake_stl10(root, {"unlabeled": 6, "train": 4, "test": 4})
    return root
