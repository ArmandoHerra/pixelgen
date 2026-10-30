"""PXG-18: Click/Typer parse errors become one ``pixelgen: error:`` line, exit 2."""

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

import pixelgen.cli as cli
from pixelgen import __version__
from pixelgen.cli import EXIT_RUNTIME, EXIT_USAGE, app

CI_ENV = {"GITHUB_ACTIONS": "true", "NO_COLOR": "1", "COLUMNS": "200"}
PLAIN_ENV = {"NO_COLOR": "1", "COLUMNS": "200"}

_SHIPPED = "from pixelgen.cli import app; app(prog_name='pixelgen')"
_CONTROL = (
    "import typer.core; import pixelgen.cli as cli; "
    "cli.app.info.cls = typer.core.TyperGroup; cli.app(prog_name='pixelgen')"
)


def _run(code, argv, tmp_path, env_extra):
    env = dict(os.environ)
    for key in ("GITHUB_ACTIONS", "NO_COLOR", "FORCE_COLOR", "PY_COLORS"):
        env.pop(key, None)
    env.update(env_extra)
    env["PIXELGEN_HOME"] = str(tmp_path / "home")
    return subprocess.run(
        [sys.executable, "-c", code, *argv],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=60,
    )


def assert_one_line(stderr: str, path: str) -> None:
    assert stderr.count("\n") == 1 and stderr.endswith("\n"), repr(stderr)
    assert stderr.startswith("pixelgen: error: "), repr(stderr)
    assert stderr.endswith(f"(see '{path} --help')\n"), repr(stderr)
    assert "\x1b" not in stderr
    assert "╭" not in stderr and "Usage:" not in stderr


_LEDGER = [
    ("unkflag-generate", ["generate", "--bogus", "x"], "pixelgen generate", "--bogus"),
    ("gen-missing-arg", ["generate"], "pixelgen generate", None),
    ("gen-badint", ["generate", "x", "--size", "big"], "pixelgen generate", "big"),
    ("refine-from-unreadable", ["refine", "x", "--from", "u.json"], "pixelgen refine", "u.json"),
    ("unknown-verb", ["frobnicate"], "pixelgen", "frobnicate"),
    ("top-unknown-flag", ["--bogus"], "pixelgen", "--bogus"),
    ("size-no-value", ["generate", "x", "--size"], "pixelgen generate", "--size"),
    ("palette-no-value", ["generate", "x", "--palette"], "pixelgen generate", "--palette"),
    ("from-no-value", ["refine", "x", "--from"], "pixelgen refine", "--from"),
]


@pytest.mark.parametrize("label,argv,path,token", _LEDGER, ids=[c[0] for c in _LEDGER])
def test_ledger_repros_print_one_product_line(label, argv, path, token, tmp_path):
    target = tmp_path / "u.json"
    restore = False
    if label == "refine-from-unreadable":
        target.write_text(json.dumps({"x": 1}))
        target.chmod(0o000)
        restore = True
    try:
        if restore and os.access(target, os.R_OK):
            pytest.skip("running as root: chmod 000 file is still readable")
        proc = _run(_SHIPPED, argv, tmp_path, CI_ENV)
    finally:
        if restore:
            target.chmod(0o600)
    assert proc.returncode == 2, proc.stderr
    assert proc.stdout == b""
    stderr = proc.stderr.decode()
    assert_one_line(stderr, path)
    if token:
        assert token in stderr


_BREADTH = [
    ("generate-flag", ["generate", "--bogus"], "pixelgen generate"),
    ("refine-flag", ["refine", "--bogus"], "pixelgen refine"),
    ("palettes-flag", ["palettes", "--bogus"], "pixelgen palettes"),
    ("presets-flag", ["presets", "--bogus"], "pixelgen presets"),
    ("refine-missing", ["refine"], "pixelgen refine"),
    ("palettes-extra", ["palettes", "extra"], "pixelgen palettes"),
    ("generate-h", ["generate", "x", "-h"], "pixelgen generate"),
    ("generate-version", ["generate", "--version"], "pixelgen generate"),
    ("unknown-verb", ["frobnicate"], "pixelgen"),
    ("badint", ["generate", "x", "--size", "big"], "pixelgen generate"),
    ("gen-o-no-value", ["generate", "x", "-o"], "pixelgen generate"),
    ("gen-scale-no-value", ["generate", "x", "--scale"], "pixelgen generate"),
    ("refine-from-no-value", ["refine", "x", "--from"], "pixelgen refine"),
    ("gen-flag-given-value", ["generate", "x", "--show-spec=1"], "pixelgen generate"),
    ("gen-pair-flag-given-value", ["generate", "x", "--outline=1"], "pixelgen generate"),
    ("root-flag-given-value", ["--version=1"], "pixelgen"),
]


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: calls.append(1))
    return calls


@pytest.mark.parametrize("label,argv,path", _BREADTH, ids=[c[0] for c in _BREADTH])
def test_every_parse_error_shape_is_one_usage_line(label, argv, path, llm_calls):
    result = CliRunner().invoke(app, argv, prog_name="pixelgen")
    assert result.exit_code == 2
    assert result.stdout == ""
    assert_one_line(result.stderr, path)
    assert llm_calls == []


def test_parse_error_line_collapses_whitespace():
    from pixelgen.cli import _USAGE_ERROR, _parse_error_line

    assert _parse_error_line(_USAGE_ERROR("first\n   second")) == "pixelgen: error: first second"
    ctx = types.SimpleNamespace(command_path="pixelgen generate", command=None)
    assert (
        _parse_error_line(_USAGE_ERROR("first\n   second", ctx=ctx))
        == "pixelgen: error: first second (see 'pixelgen generate --help')"
    )


@pytest.mark.parametrize("label,argv,path", _BREADTH, ids=[c[0] for c in _BREADTH])
def test_parse_errors_exit_with_exit_usage(label, argv, path, llm_calls):
    result = CliRunner().invoke(app, argv, prog_name="pixelgen")
    assert result.exit_code == EXIT_USAGE == 2
    assert result.exit_code != EXIT_RUNTIME


_HELP_ARGV = [
    [],
    ["--help"],
    ["generate", "--help"],
    ["refine", "--help"],
    ["palettes", "--help"],
    ["presets", "--help"],
    ["--version"],
]


@pytest.mark.parametrize("env_name", ["ci", "plain"])
@pytest.mark.parametrize("argv", _HELP_ARGV, ids=lambda a: " ".join(a) or "noargs")
def test_help_surface_bytes_unchanged(env_name, argv, tmp_path):
    env = CI_ENV if env_name == "ci" else PLAIN_ENV
    shipped = _run(_SHIPPED, argv, tmp_path, env)
    control = _run(_CONTROL, argv, tmp_path, env)
    assert (shipped.returncode, shipped.stdout, shipped.stderr) == (
        control.returncode,
        control.stdout,
        control.stderr,
    )
    if argv == []:
        assert shipped.returncode == 2 and shipped.stdout and shipped.stderr == b""
    elif argv == ["--help"]:
        assert shipped.returncode == 0
    elif argv == ["--version"]:
        assert shipped.stdout == f"pixelgen {__version__}\n".encode()


@pytest.mark.parametrize(
    "env_extra",
    [{"GITHUB_ACTIONS": "true", "COLUMNS": "200"}, {"COLUMNS": "200"}],
    ids=["rich-ci", "rich-plain"],
)
def test_noargs_rich_path_matches_control(env_extra, tmp_path):
    shipped = _run(_SHIPPED, [], tmp_path, env_extra)
    control = _run(_CONTROL, [], tmp_path, env_extra)
    assert (shipped.returncode, shipped.stdout, shipped.stderr) == (
        control.returncode,
        control.stdout,
        control.stderr,
    )
    assert shipped.returncode == 2
    assert shipped.stdout
    assert shipped.stderr == b""


def test_every_verb_parses_with_pixelgen_command():
    import typer.main

    root = typer.main.get_command(app)
    assert isinstance(root, cli.PixelgenGroup)
    assert {"generate", "refine", "palettes", "presets"} <= set(root.commands)
    for name, command in root.commands.items():
        assert isinstance(command, cli.PixelgenCommand), name


def test_attach_parse_ctx_fills_only_a_missing_ctx():
    from pixelgen.cli import _USAGE_ERROR, _attach_parse_ctx

    outer = types.SimpleNamespace(command_path="pixelgen generate", command=None)
    inner = types.SimpleNamespace(command_path="pixelgen generate", command=None)
    with pytest.raises(_USAGE_ERROR) as filled:
        with _attach_parse_ctx(outer):
            raise _USAGE_ERROR("m")
    assert filled.value.ctx is outer
    with pytest.raises(_USAGE_ERROR) as kept:
        with _attach_parse_ctx(outer):
            raise _USAGE_ERROR("m", ctx=inner)
    assert kept.value.ctx is inner
    with pytest.raises(ValueError) as other:
        with _attach_parse_ctx(outer):
            raise ValueError("v")
    assert not hasattr(other.value, "ctx")


_EXISTING = [
    (
        "scale",
        ["generate", "x", "--scale", "0"],
        "pixelgen: error: --scale must be between 1 and 64\n",
    ),
    ("empty-prompt", ["generate", " "], "pixelgen: error: prompt is empty\n"),
    (
        "size-width",
        ["generate", "x", "--size", "8", "--width", "8"],
        "pixelgen: error: --size cannot be combined with --width or --height\n",
    ),
]


@pytest.mark.parametrize("label,argv,expected", _EXISTING, ids=[c[0] for c in _EXISTING])
def test_existing_usage_lines_unchanged(label, argv, expected, llm_calls):
    result = CliRunner().invoke(app, argv, prog_name="pixelgen")
    assert result.exit_code == 2
    assert result.stderr == expected


def test_cli_imports_no_private_click():
    tree = ast.parse(Path(cli.__file__).read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for n in names:
            if n == "click" or n.startswith(("click.", "typer._click")):
                bad.append(f"{type(node).__name__} {n}")
        if isinstance(node, ast.Attribute) and node.attr == "_click":
            bad.append("Attribute _click")
        if isinstance(node, ast.Name) and node.id == "_click":
            bad.append("Name _click")
    assert bad == []
