"""NO_COLOR must win over CI colour-forcing variables on every help and error path.

Every case runs a subprocess with an environment built from scratch, so the
developer's or runner's own colour variables never leak into the arms.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ESC = "\x1b"

HELP_ARGV = [
    [],
    ["--help"],
    ["generate", "--help"],
    ["refine", "--help"],
    ["palettes", "--help"],
    ["presets", "--help"],
]
FORCERS = [
    {"GITHUB_ACTIONS": "true"},
    {"FORCE_COLOR": "1"},
    {"PY_COLORS": "1"},
]
_RUNNER = "from pixelgen.cli import app; app(prog_name='pixelgen')"


def _env(tmp_path: Path, extra: dict[str, str]) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "")}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    env["PIXELGEN_HOME"] = str(tmp_path / "home")
    env["COLUMNS"] = "200"
    env.update(extra)
    return env


def _run(cmd: list[str], tmp_path: Path, extra: dict[str, str]):
    return subprocess.run(
        cmd,
        cwd=tmp_path,
        env=_env(tmp_path, extra),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _run_app(argv: list[str], tmp_path: Path, extra: dict[str, str]):
    return _run([sys.executable, "-c", _RUNNER, *argv], tmp_path, extra)


def _esc_count(proc) -> int:
    return (proc.stdout + proc.stderr).count(ESC)


def _forcer_id(forcer: dict[str, str]) -> str:
    return next(iter(forcer))


@pytest.mark.parametrize("forcer", FORCERS, ids=_forcer_id)
@pytest.mark.parametrize("argv", HELP_ARGV, ids=lambda a: " ".join(a) or "no-args")
def test_no_color_help_has_no_ansi(tmp_path, forcer, argv):
    proc = _run_app(argv, tmp_path, {**forcer, "NO_COLOR": "1"})
    assert proc.returncode == (2 if not argv else 0), proc.stderr
    assert _esc_count(proc) == 0, (proc.stdout + proc.stderr)[:500]


def _error_cases(tmp_path: Path):
    return [
        (["generate", "--bogus"], 2),
        (["generate"], 2),
        (["generate", "x", "--scale", "0"], 2),
        (["refine", "x", "--from", str(tmp_path / "missing.json")], 2),
        (["generate", "x", "--size", "16"], 1),
    ]


@pytest.mark.parametrize("index", range(5))
def test_no_color_error_paths_have_no_ansi(tmp_path, index):
    argv, code = _error_cases(tmp_path)[index]
    proc = _run_app(argv, tmp_path, {"GITHUB_ACTIONS": "true", "NO_COLOR": "1"})
    assert proc.returncode == code, (argv, proc.stdout, proc.stderr)
    assert proc.stderr.startswith("pixelgen: error:"), proc.stderr
    assert _esc_count(proc) == 0, (proc.stdout + proc.stderr)[:500]


def test_no_color_console_script_help_has_no_ansi(tmp_path):
    script = Path(sys.executable).parent / "pixelgen"
    proc = _run([str(script), "--help"], tmp_path, {"GITHUB_ACTIONS": "true", "NO_COLOR": "1"})
    assert proc.returncode == 0, proc.stderr
    assert _esc_count(proc) == 0, (proc.stdout + proc.stderr)[:500]


@pytest.mark.parametrize(
    "env",
    [
        {"NO_COLOR": "1"},
        {"GITHUB_ACTIONS": "true", "NO_COLOR": "1"},
        {},
    ],
    ids=["no-color", "ci-no-color", "rich-path"],
)
def test_no_color_no_args_help_on_stdout(tmp_path, env):
    proc = _run_app([], tmp_path, env)
    assert proc.returncode == 2
    assert "Usage:" in proc.stdout
    assert proc.stderr == ""
    if "NO_COLOR" in env:
        assert _esc_count(proc) == 0


@pytest.mark.parametrize(
    "argv, extra",
    [
        (["--help"], {"FORCE_COLOR": "1"}),
        (["generate", "--help"], {"FORCE_COLOR": "1"}),
        (["--help"], {"FORCE_COLOR": "1", "NO_COLOR": ""}),
    ],
    ids=["root", "generate", "empty-no-color"],
)
def test_colour_still_emitted_without_no_color(tmp_path, argv, extra):
    proc = _run_app(argv, tmp_path, extra)
    assert proc.returncode == 0, proc.stderr
    assert _esc_count(proc) >= 1
