import pathlib
import subprocess
import sys

SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "check_imports.py"


def _run(tmp_path: pathlib.Path, source: str) -> subprocess.CompletedProcess[str]:
    target = tmp_path / "mod.py"
    target.write_text(source)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(target)], capture_output=True, text=True
    )


def test_allows_modules_and_typing(tmp_path: pathlib.Path) -> None:
    result = _run(
        tmp_path,
        "import os\n"
        "from os import path\n"
        "from typing import Any\n"
        "from collections.abc import Mapping\n",
    )
    assert result.returncode == 0, result.stdout


def test_rejects_importing_classes_and_functions(tmp_path: pathlib.Path) -> None:
    result = _run(tmp_path, "from pathlib import Path\nfrom os.path import join\n")
    assert result.returncode == 1
    assert "Path" in result.stdout
    assert "join" in result.stdout
