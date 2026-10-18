"""Filesystem locations, read from environment variables (optionally via `.env`).

No path is ever hard-coded: every location comes from the environment so the same
code runs on a laptop and on the cluster. `.env` is git-ignored; `.env.example`
documents the variables with placeholder values.
"""

import dataclasses
import os
import pathlib

import dotenv

DATA_DIR_VAR = "LEJEPA_NORM_DATA_DIR"
RUNS_DIR_VAR = "LEJEPA_NORM_RUNS_DIR"
EMBED_DIR_VAR = "LEJEPA_NORM_EMBED_DIR"
NOTES_DIR_VAR = "LEJEPA_NORM_NOTES_DIR"
KALEIDO_CHROME_VAR = "LEJEPA_NORM_KALEIDO_CHROME"


@dataclasses.dataclass(frozen=True)
class Paths:
    """Resolved project locations.

    Attributes:
        data_dir: Prepared datasets (STL-10, STL-10-C, DTD, SVHN).
        runs_dir: Checkpoints, resolved configs, and `metrics.jsonl`.
        embed_dir: Embedding dumps produced by `apps/embed.py`.
        notes_dir: `research_notes/` in the repo checkout.
        kaleido_chrome: Chrome binary used by Kaleido, or None if unset (only
            needed when exporting plots).
    """

    data_dir: pathlib.Path
    runs_dir: pathlib.Path
    embed_dir: pathlib.Path
    notes_dir: pathlib.Path
    kaleido_chrome: pathlib.Path | None


def _require(name: str) -> pathlib.Path:
    value = os.environ.get(name)
    if not value:
        raise KeyError(
            f"Environment variable {name} is not set. Copy .env.example to .env "
            "and fill it in."
        )
    return pathlib.Path(value).expanduser()


def load_paths(env_file: str | os.PathLike[str] | None = None) -> Paths:
    """Builds `Paths` from the environment.

    Variables already set in the environment take precedence over `.env`.

    Args:
        env_file: Explicit `.env` file. If None, `.env` is searched for upwards
            from the current working directory (missing is fine).

    Returns:
        The resolved paths.

    Raises:
        KeyError: A required variable is unset or empty.
    """
    if env_file is None:
        env_file = dotenv.find_dotenv(usecwd=True)
    if env_file:
        dotenv.load_dotenv(env_file, override=False)
    chrome = os.environ.get(KALEIDO_CHROME_VAR)
    return Paths(
        data_dir=_require(DATA_DIR_VAR),
        runs_dir=_require(RUNS_DIR_VAR),
        embed_dir=_require(EMBED_DIR_VAR),
        notes_dir=_require(NOTES_DIR_VAR),
        kaleido_chrome=pathlib.Path(chrome).expanduser() if chrome else None,
    )
