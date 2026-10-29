"""Pins for the sweep-proven CLI contract points no other test covers (PXG-23).

Everything is offline: in-process rows use ``FakeLLM`` through the ``cli.make_llm`` seam, and
the subprocess rows (AC7) run a driver that replaces the Anthropic client with a raising guard
and counts every construction.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import FakeLLM
from PIL import Image
from typer.testing import CliRunner

import pixelgen.cli as cli
import pixelgen.pipeline as pipeline
from pixelgen.cli import EXIT_RUNTIME, EXIT_USAGE, app
from pixelgen.export import TokenUsage, load_last, load_sidecar
from pixelgen.palettes import load_palette
from pixelgen.spec import ArtSpec

runner = CliRunner()


def _grid(side: int = 16) -> list[list[int]]:
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5) % 16
            for c in range(side)
        ]
        for r in range(side)
    ]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.llm: FakeLLM | None = None
        self.make_llm_calls = 0
        monkeypatch.chdir(tmp_path)

    def invoke(self, args, *script, key=True):
        self.llm = FakeLLM(*script)
        if key:
            self.mp.setenv("ANTHROPIC_API_KEY", "dummy-key")

        def fake_make():
            self.make_llm_calls += 1
            return self.llm

        self.mp.setattr(cli, "make_llm", fake_make)
        return runner.invoke(app, args)

    def generate(self, args, key=True):
        return self.invoke(["generate", *args], {}, {"grid": _grid()}, key=key)

    def seed(self, name, directory):
        """Write a refine source via ``render_grid`` (no LLM); returns the sidecar path."""
        spec = ArtSpec(subject=name, width=8, height=8, palette="pico8")
        grid = [[(r * 3 + c * 5 + 1) % 16 for c in range(8)] for r in range(8)]
        result = pipeline.render_grid(
            spec,
            grid,
            load_palette("pico8"),
            output=directory / name,
            scale=8,
            overwrite=False,
            backend="llm-grid",
            model="fake-model",
            token_usage=TokenUsage(input_tokens=1, output_tokens=1, calls=1),
        )
        return result.export.sidecar_path


@pytest.fixture
def h(monkeypatch, tmp_path):
    return _Harness(monkeypatch, tmp_path)


def _error_line(result, code, *tokens):
    assert result.exit_code == code, (result.stdout, result.stderr, result.exception)
    lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, result.stderr
    line = lines[0]
    assert line.startswith("pixelgen: error:")
    assert "ANTHROPIC_API_KEY" not in line
    for token in tokens:
        assert token in line
    return line


def _tree(root: Path) -> set[str]:
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if "pixelgen-home" not in p.relative_to(root).parts
    }


# --- AC1 ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        pytest.param("日本語のドット絵", "pixelgen", id="cjk"),
        pytest.param("?!...;:", "pixelgen", id="punctuation-only"),
        pytest.param("..", "pixelgen", id="dotdot"),
        pytest.param("../..", "pixelgen", id="dotdot-dotdot"),
        pytest.param("‮", "pixelgen", id="rlo-only"),
        pytest.param("a/b", "a-b", id="forward-slash"),
        pytest.param("a\\b", "a-b", id="backslash"),
        pytest.param("red\x1b[31mapple", "red-31mapple", id="ansi-escape"),
        pytest.param("red\napple", "red-apple", id="newline"),
        pytest.param("abc‮def", "abcdef", id="rlo-inside-word"),
    ],
)
def test_contract_slug_table(prompt, expected):
    slug = pipeline.slugify_prompt(prompt)
    assert slug == expected
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", slug)
    assert len(slug) <= 50


def test_contract_slug_reaches_the_default_filename(h):
    # Arrange / Act
    cjk = h.generate(["日本語", "--size", "16"])
    # Assert
    assert cjk.exit_code == 0, (cjk.stdout, cjk.stderr)
    out = h.tmp / "out"
    assert (out / "pixelgen.png").is_file()
    assert (out / "pixelgen@8x.png").is_file()
    assert (out / "pixelgen.json").is_file()


def test_contract_slug_traversal_prompt_stays_inside_out(h):
    before = _tree(h.tmp)
    result = h.generate(["../../etc/passwd", "--size", "16"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    added = _tree(h.tmp) - before
    assert added == {
        "out",
        "out/etc-passwd.png",
        "out/etc-passwd@8x.png",
        "out/etc-passwd.json",
    }


# --- AC2 ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("argv", "token"),
    [
        pytest.param(["--palette", "nope"], "endesga32", id="palette"),
        pytest.param(["--preset", "nope"], "16bit-rpg", id="preset"),
        pytest.param(["--dither", "sparkle"], "dither", id="dither"),
        pytest.param(["--view", "sideways"], "view", id="view"),
        pytest.param(["--background", "plaid"], "background", id="background"),
        pytest.param(["--backend", "nope"], "llm-grid", id="backend"),
        pytest.param(["--size", "7"], "", id="size"),
        pytest.param(["--scale", "0"], "--scale", id="scale"),
        pytest.param(["-o", "<existing-dir>"], "existing-dir", id="output-dir"),
    ],
)
def test_contract_generate_usage_error_beats_missing_key(h, argv, token):
    (h.tmp / "existing-dir").mkdir()
    argv = ["existing-dir" if a == "<existing-dir>" else a for a in argv]

    result = h.invoke(["generate", "x", *argv], key=False)

    _error_line(result, EXIT_USAGE, *([token] if token else []))
    assert h.make_llm_calls == 0
    assert h.llm.call_count == 0
    assert not (h.tmp / "out").exists()


# --- AC3 ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("argv", "token"),
    [
        pytest.param(["--palette", "../../../../etc/passwd"], "endesga32", id="palette-etc"),
        pytest.param(["--palette", "../palettes/pico8"], "endesga32", id="palette-real"),
        pytest.param(["--preset", "../../../../etc/passwd"], "16bit-rpg", id="preset-etc"),
        pytest.param(["--preset", "../presets/icon"], "16bit-rpg", id="preset-real"),
    ],
)
def test_contract_traversal_names_are_unknown(h, argv, token):
    result = h.invoke(["generate", "x", *argv], key=False)

    _error_line(result, EXIT_USAGE, token)
    assert h.make_llm_calls == 0


# --- AC4 ------------------------------------------------------------------------------------
def _bad_from(tmp: Path, kind: str) -> Path:
    if kind == "directory":
        path = tmp / "adir"
        path.mkdir()
    elif kind == "empty":
        path = tmp / "empty.json"
        path.write_bytes(b"")
    elif kind == "array":
        path = tmp / "array.json"
        path.write_text("[]")
    else:
        path = tmp / "image.json"
        Image.new("RGBA", (1, 1)).save(path, format="PNG")
    return path


@pytest.mark.parametrize("kind", ["directory", "empty", "array", "png-bytes"])
def test_contract_refine_bad_from_beats_missing_key(h, kind):
    source = _bad_from(h.tmp, kind)

    result = h.invoke(["refine", "x", "--from", str(source)], key=False)

    _error_line(result, EXIT_USAGE)
    assert h.make_llm_calls == 0
    assert not [p for p in h.tmp.rglob("*-v2*")]


# --- AC5 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("row", ["file-symlink", "dir-alias-generate", "dir-alias-refine"])
def test_contract_output_aliases(h, row):
    real = h.tmp / "real"
    real.mkdir()
    realdir = h.tmp / "realdir"
    realdir.mkdir()
    (h.tmp / "link").symlink_to(realdir)
    if row == "file-symlink":
        _check_file_symlink(h, real)
    elif row == "dir-alias-generate":
        _check_dir_alias_generate(h, realdir)
    else:
        _check_dir_alias_refine(h, realdir)


def _check_file_symlink(h, real):
    # a symlink at an explicit -o is replaced, never written through
    (real / "keep.png").write_bytes(b"precious\n")
    out = h.tmp / "out"
    out.mkdir()
    (out / "x.png").symlink_to(real / "keep.png")

    result = h.generate(["a red apple", "--size", "16", "-o", "out/x.png"])

    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert not (out / "x.png").is_symlink()
    with Image.open(out / "x.png") as img:
        assert img.mode == "RGBA"
        assert img.size == (16, 16)
    assert _digest(real / "keep.png") == hashlib.sha256(b"precious\n").hexdigest()
    assert [p.name for p in real.iterdir()] == ["keep.png"]


def _check_dir_alias_generate(h, realdir):
    # a symlinked output directory is followed, and stays a symlink
    result = h.generate(["a red apple", "--size", "16", "-o", "link/y.png"])

    assert result.exit_code == 0, (result.stdout, result.stderr)
    for name in ("y.png", "y@8x.png", "y.json"):
        assert os.path.samefile(h.tmp / "link" / name, realdir / name)
    assert (h.tmp / "link").is_symlink()
    assert load_last().sidecar_path.resolve() == (realdir / "y.json").resolve()


def _check_dir_alias_refine(h, realdir):
    # refine reads and writes through the directory alias
    h.seed("knight", realdir)
    before = {p.name: _digest(p) for p in realdir.glob("knight*")}
    assert len(before) == 3
    grid = [[(r * 3 + c * 5 + 2) % 16 for c in range(8)] for r in range(8)]

    result = h.invoke(["refine", "x", "--from", "link/knight.json"], {"spec": {}, "grid": grid})

    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert os.path.samefile(h.tmp / "link" / "knight-v2.json", realdir / "knight-v2.json")
    after = {p.name: _digest(p) for p in realdir.glob("knight*") if "-v2" not in p.name}
    assert after == before


# --- AC6 ------------------------------------------------------------------------------------
def test_contract_output_paths_spaces_and_unicode(h):
    target = Path("out") / "my dir" / "ñandú ☃" / "sprite.png"

    result = h.generate(["a red apple", "--size", "16", "-o", str(target)])

    assert result.exit_code == 0, (result.stdout, result.stderr)
    directory = h.tmp / target.parent
    assert sorted(p.name for p in directory.iterdir()) == [
        "sprite.json",
        "sprite.png",
        "sprite@8x.png",
    ]
    assert str(directory / "sprite.png") in result.stdout
    load_sidecar(directory / "sprite.json")
    assert load_last().sidecar_path.resolve() == (directory / "sprite.json").resolve()


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root can write /dev; would create /dev/null.png",
)
def test_contract_output_paths_dev_null_is_refused_before_spend(h):
    result = h.generate(["a red apple", "--size", "16", "-o", "/dev/null"])

    _error_line(result, EXIT_RUNTIME, "/dev")
    assert h.make_llm_calls == 0
    assert h.llm.call_count == 0


# --- AC7 ------------------------------------------------------------------------------------
# Ambient variables that change colour or terminal detection; scrubbed so CI cannot leak in.
_SCRUBBED_ENV = (
    "GITHUB_ACTIONS",
    "FORCE_COLOR",
    "PY_COLORS",
    "CLICOLOR_FORCE",
    "TTY_COMPATIBLE",
    "TTY_INTERACTIVE",
    "NO_COLOR",
    "TERM",
)

_VARIANTS = {
    "default": {"TERM": "xterm-256color"},
    "no_color": {"TERM": "xterm-256color", "NO_COLOR": "1"},
    "dumb": {"TERM": "dumb"},
}

_DRIVER = r"""
import atexit, json, os, sys
from pathlib import Path
home = os.environ.get("PIXELGEN_HOME")
if not home or Path(home).resolve() == (Path.home() / ".pixelgen").resolve():
    sys.exit(97)
tests_dir, script_path, counts_path, _sep, *argv = sys.argv[1:]
sys.path.insert(0, tests_dir)
import anthropic
state = {"made": 0, "hits": 0}
class _Guard:
    def __init__(self, *a, **k):
        state["hits"] += 1
        raise RuntimeError("real Anthropic client constructed")
anthropic.Anthropic = _Guard
anthropic.AsyncAnthropic = _Guard
from fakes import FakeLLM
import pixelgen.cli as cli
llm = FakeLLM(*json.load(open(script_path)))
def _make():
    state["made"] += 1
    return llm
cli.make_llm = _make
def _write():
    Path(counts_path).write_text(
        "make_llm=%d call_count=%d anthropic_constructed=%d"
        % (state["made"], llm.call_count, state["hits"])
    )
atexit.register(_write)
cli.app(prog_name="pixelgen", args=argv)
"""

_TESTS_DIR = str(Path(__file__).resolve().parent)


def _child_env(tmp: Path, variant: str, *, key: bool) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _SCRUBBED_ENV and not k.startswith("ANTHROPIC_")
    }
    env["PIXELGEN_HOME"] = str(tmp / "home")
    env["COLUMNS"] = "80"
    env.update(_VARIANTS[variant])
    if key:
        env["ANTHROPIC_API_KEY"] = "dummy-key"
    return env


def _drive(tmp: Path, variant: str, argv: list[str], script: list, *, key: bool, tag: str):
    script_path = tmp / f"{tag}-script.json"
    counts_path = tmp / f"{tag}-counts.txt"
    script_path.write_text(json.dumps(script))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _DRIVER,
            _TESTS_DIR,
            str(script_path),
            str(counts_path),
            "--",
            *argv,
        ],
        cwd=tmp,
        env=_child_env(tmp, variant, key=key),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 97, "driver refused an unsafe PIXELGEN_HOME"
    assert counts_path.is_file(), (proc.returncode, proc.stdout, proc.stderr)
    counts = dict(item.split("=") for item in counts_path.read_text().split())
    return proc, counts


_GEN_SCRIPT = [{}, {"grid": _grid()}]
_REFINE_SCRIPT = [{"spec": {}, "grid": _grid()}]

# (id, argv, script, key, exit code, stdout token or None)
_COMMANDS = [
    (
        "generate-show-spec",
        ["generate", "a red apple", "--size", "16", "--show-spec"],
        _GEN_SCRIPT,
        True,
        0,
        "Resolved spec",
    ),
    ("refine", ["refine", "add a hat"], _REFINE_SCRIPT, True, 0, "-v2.png"),
    ("generate-unknown-palette", ["generate", "x", "--palette", "nope"], [], True, 2, None),
    ("generate-no-key", ["generate", "x", "--size", "16"], [], False, 1, None),
    ("refine-missing-from", ["refine", "x", "--from", "missing.json"], [], True, 2, None),
    ("generate-bad-flag", ["generate", "x", "--no-such-flag"], [], True, 2, None),
]


@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize(
    ("argv", "script", "key", "code", "stdout_token"),
    [pytest.param(*c[1:], id=c[0]) for c in _COMMANDS],
)
def test_contract_non_tty_no_ansi(tmp_path, variant, argv, script, key, code, stdout_token):
    if argv[0] == "refine" and code == 0:
        # refine reads last.json written by a first generate in the same PIXELGEN_HOME
        first, first_counts = _drive(
            tmp_path,
            variant,
            ["generate", "a red apple", "--size", "16"],
            _GEN_SCRIPT,
            key=True,
            tag="seed",
        )
        assert first.returncode == 0, first.stderr
        assert first_counts["anthropic_constructed"] == "0"

    proc, counts = _drive(tmp_path, variant, argv, script, key=key, tag="row")

    assert proc.returncode == code, (proc.stdout, proc.stderr)
    combined = proc.stdout + proc.stderr
    assert [ln for ln in combined.splitlines() if "\x1b" in ln] == []
    assert counts["anthropic_constructed"] == "0"
    if code != 0:
        assert proc.stderr.splitlines()[0].startswith("pixelgen: error:")
        assert counts["make_llm"] == "0"
        assert counts["call_count"] == "0"
    if stdout_token is not None:
        assert stdout_token in proc.stdout
