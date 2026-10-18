import os
import pathlib

import pytest

from lejepa_norm import paths

_VARS = [
    paths.DATA_DIR_VAR,
    paths.RUNS_DIR_VAR,
    paths.EMBED_DIR_VAR,
    paths.NOTES_DIR_VAR,
    paths.KALEIDO_CHROME_VAR,
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _VARS:
        # Set-then-delete so monkeypatch records "unset" and removes anything
        # load_dotenv writes into os.environ during the test.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


def _write_env(path: pathlib.Path, **overrides: str) -> pathlib.Path:
    values = {
        paths.DATA_DIR_VAR: "/d",
        paths.RUNS_DIR_VAR: "/r",
        paths.EMBED_DIR_VAR: "/e",
        paths.NOTES_DIR_VAR: "/n",
    }
    values.update(overrides)
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    return path


def test_loads_from_env_file(tmp_path: pathlib.Path) -> None:
    env = _write_env(tmp_path / ".env")
    loaded = paths.load_paths(env)
    assert loaded.data_dir == pathlib.Path("/d")
    assert loaded.notes_dir == pathlib.Path("/n")
    assert loaded.kaleido_chrome is None


def test_real_environment_beats_env_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.RUNS_DIR_VAR, "/from-env")
    env = _write_env(tmp_path / ".env", **{paths.KALEIDO_CHROME_VAR: "/chrome"})
    loaded = paths.load_paths(env)
    assert loaded.runs_dir == pathlib.Path("/from-env")
    assert loaded.kaleido_chrome == pathlib.Path("/chrome")


def test_missing_variable_names_the_variable(tmp_path: pathlib.Path) -> None:
    env = tmp_path / ".env"
    env.write_text(f"{paths.DATA_DIR_VAR}=/d\n")
    with pytest.raises(KeyError, match=paths.RUNS_DIR_VAR):
        paths.load_paths(env)


def test_env_file_values_do_not_leak_between_tests(tmp_path: pathlib.Path) -> None:
    paths.load_paths(_write_env(tmp_path / ".env"))
    assert paths.DATA_DIR_VAR in os.environ  # visible within this test...


def test_environment_is_clean_in_the_next_test() -> None:
    assert paths.DATA_DIR_VAR not in os.environ  # ...but gone afterwards
