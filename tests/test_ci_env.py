"""Regression guard: the CLI tests must pass under CI's colour environment.

CI sets GITHUB_ACTIONS / FORCE_COLOR / PY_COLORS, which make Typer force ANSI output.
The app honours NO_COLOR per invocation (conftest sets it), so help stays plain. This
test reproduces a CI process in a subprocess: if Typer stops honouring the
per-invocation markup mode, the child run goes red even on a developer machine.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_cli_suite_passes_under_ci_colour_env():
    env = dict(os.environ)
    env.update(
        GITHUB_ACTIONS="true",
        CI="true",
        FORCE_COLOR="1",
        PY_COLORS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for key in ("NO_COLOR", "COLUMNS", "TERM"):
        env.pop(key, None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_cli.py",
            "tests/test_cli_refine.py",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (
        f"CLI tests failed under CI colour env:\n{proc.stdout}\n{proc.stderr}"
    )
