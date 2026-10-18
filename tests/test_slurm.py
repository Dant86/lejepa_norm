"""`run_preemptable` (slurm/common.sh) around fake apps, run locally with bash."""

import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")

# A fake app. `handler`: SIGUSR1 sets a flag, the app "checkpoints" and exits 99,
# like apps/train.py. `plain`: no handler, so SIGUSR1 kills it, like apps/embed.py.
# `exit:N`: exits N without being signalled.
_APP = """
import pathlib, signal, sys, time
mode, workdir = sys.argv[1], pathlib.Path(sys.argv[2])
flag = []
if mode == "handler":
    signal.signal(signal.SIGUSR1, lambda *_: flag.append(1))
if mode.startswith("exit:"):
    sys.exit(int(mode.split(":")[1]))
(workdir / "ready").touch()
deadline = time.monotonic() + 60
while not flag and time.monotonic() < deadline:
    time.sleep(0.05)
(workdir / "checkpointed").touch()
sys.exit(99)
"""

pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def _start(tmp_path: pathlib.Path, mode: str) -> subprocess.Popen[str]:
    assert BASH is not None
    env_file = tmp_path / "test.env"
    env_file.write_text(f"LEJEPA_NORM_RUNS_DIR={tmp_path / 'runs'}\n")
    app = tmp_path / "app.py"
    app.write_text(_APP)
    script = (
        f'source "{REPO / "slurm" / "common.sh"}"\n'
        f'run_preemptable "{sys.executable}" "{app}" {mode} "{tmp_path}"\n'
    )
    return subprocess.Popen(
        [BASH, "-c", script],
        env={**os.environ, "LEJEPA_NORM_ENV_FILE": str(env_file)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _signal_when_ready(
    proc: subprocess.Popen[str], tmp_path: pathlib.Path, signum: int
) -> tuple[int, str]:
    deadline = time.monotonic() + 30
    while not (tmp_path / "ready").exists():
        assert proc.poll() is None, proc.communicate()[0]
        assert time.monotonic() < deadline, "fake app never started"
        time.sleep(0.05)
    proc.send_signal(signum)
    out, _ = proc.communicate(timeout=60)
    return proc.returncode, out


def test_usr1_checkpoints_and_requests_requeue(tmp_path: pathlib.Path) -> None:
    proc = _start(tmp_path, "handler")
    code, out = _signal_when_ready(proc, tmp_path, signal.SIGUSR1)
    assert code == 99, out
    assert (tmp_path / "checkpointed").exists()  # the app saw the forwarded USR1
    assert "USR1" in out


@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGTERM])
def test_app_without_handler_is_requeued_only_on_usr1(
    tmp_path: pathlib.Path, signum: int
) -> None:
    proc = _start(tmp_path, "plain")
    code, out = _signal_when_ready(proc, tmp_path, signum)
    assert code == (99 if signum == signal.SIGUSR1 else 143), out
    assert not (tmp_path / "checkpointed").exists()  # the app was killed


@pytest.mark.parametrize("app_code", [0, 3, 99])
def test_app_exit_code_is_propagated(tmp_path: pathlib.Path, app_code: int) -> None:
    proc = _start(tmp_path, f"exit:{app_code}")
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == app_code, out
