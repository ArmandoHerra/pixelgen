"""PXG-40: one ``pixelgen: usage:`` stderr line per successful generate/refine.

Offline: in-process rows use ``FakeLLM`` through the ``cli.make_llm`` seam; the colour rows run a
child process whose Anthropic client is replaced by a raising guard.
"""

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import FakeLLM
from typer.testing import CliRunner

import pixelgen.cli as cli
from pixelgen.cli import app
from pixelgen.errors import LLMError

runner = CliRunner()

PREFIX = "pixelgen: usage: "
PRINT_SUMMARY_SHA = "d25fbe302a725a3d34cfd6ceaaed6f48d1700711791e395efc94efcf56910b5d"
RECORD_KEYS = {"step", "provider", "model", "unit", "calls", "quantity"}


def _grid(side: int = 16) -> list[list[int]]:
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5) % 16
            for c in range(side)
        ]
        for r in range(side)
    ]


def _ulines(stderr: str) -> list[str]:
    return [ln for ln in stderr.splitlines() if ln.startswith("pixelgen: usage:")]


def payload(line: str):
    return json.loads(line.removeprefix(PREFIX))


def file_usage(path: Path):
    return json.loads(Path(path).read_text())["usage"]


def _sidecar_from(stdout: str) -> Path:
    token = stdout.split("sidecar ", 1)[1].split(",", 1)[0].strip()
    return Path(token)


class _Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.llm = FakeLLM()
        self.make_llm_calls = 0
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

        def fake_make():
            self.make_llm_calls += 1
            return self.llm

        monkeypatch.setattr(cli, "make_llm", fake_make)

    def invoke(self, args, *script, key=True):
        self.llm = FakeLLM(*script)
        self.make_llm_calls = 0
        if not key:
            self.mp.delenv("ANTHROPIC_API_KEY", raising=False)
        return runner.invoke(app, args)

    def generate(self, extra=(), prompt="a red apple", size="16"):
        return self.invoke(["generate", prompt, "--size", size, *extra], {}, {"grid": _grid()})

    def refine(self, extra=(), spec=None):
        return self.invoke(["refine", "add a hat", *extra], {"spec": spec or {}, "grid": _grid()})


@pytest.fixture
def h(monkeypatch, tmp_path):
    return _Harness(monkeypatch, tmp_path)


def _symbol(name):
    sym = getattr(cli, name, None)
    assert sym is not None, f"pixelgen.cli.{name} is missing"
    return sym


# --- AC1 ------------------------------------------------------------------------------------
def test_generate_prints_one_usage_line(h):
    result = h.generate()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert len(_ulines(result.stderr)) == 1
    sidecar = _sidecar_from(result.stdout)
    expected = _symbol("usage_line")(file_usage(sidecar)) + "\n"
    assert result.stderr == expected
    assert h.make_llm_calls == 1
    assert h.llm.call_count == 2


# --- AC2 ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "extra",
    [
        pytest.param([], id="default"),
        pytest.param(["-o", "sprite.png"], id="output"),
        pytest.param(["--show-spec"], id="show-spec"),
        pytest.param(["--aseprite"], id="aseprite"),
    ],
)
def test_generate_usage_line_matches_sidecar(h, extra):
    result = h.generate(extra)

    assert result.exit_code == 0, (result.stdout, result.stderr)
    lines = _ulines(result.stderr)
    assert len(lines) == 1
    records = payload(lines[0])
    sidecar = _sidecar_from(result.stdout)
    assert records == file_usage(sidecar)
    assert records
    assert [r["step"] for r in records] == ["infer_spec", "generate"]
    token_calls = json.loads(sidecar.read_text())["token_usage"]["calls"]
    assert sum(r["calls"] for r in records) == token_calls == 2
    if "--show-spec" in extra:
        assert "Resolved spec" in result.stdout


# --- AC3 ------------------------------------------------------------------------------------
def test_refine_usage_line_is_last_and_from_new_sidecar(h):
    first = h.generate()
    assert first.exit_code == 0, first.stderr
    source = _sidecar_from(first.stdout)

    plain = h.refine()

    assert plain.exit_code == 0, (plain.stdout, plain.stderr)
    assert len(plain.stderr.splitlines()) == 1
    assert len(_ulines(plain.stderr)) == 1
    new_sidecar = _sidecar_from(plain.stdout)
    assert new_sidecar != source
    records = payload(_ulines(plain.stderr)[0])
    assert records == file_usage(new_sidecar)
    assert [r["step"] for r in records] == ["refine"]
    assert records != file_usage(source)
    assert [r["step"] for r in file_usage(source)] == ["infer_spec", "generate"]

    locked = h.refine(spec={"width": 32, "palette": "gameboy", "preset": "nes"})

    assert locked.exit_code == 0, (locked.stdout, locked.stderr)
    lines = locked.stderr.splitlines()
    assert len(lines) == 2
    assert lines[0] == (
        "pixelgen: note: refine keeps the source's width, palette, preset; "
        "that requested change was ignored"
    )
    assert lines[1].startswith("pixelgen: usage:")
    assert payload(lines[1]) == file_usage(_sidecar_from(locked.stdout))


# --- AC4 ------------------------------------------------------------------------------------
def _wrote(stdout_paths) -> str:
    png, w, h_, preview, sidecar, ase = stdout_paths
    return (
        f"Wrote {png} ({w}×{h_}), preview {preview}, sidecar {sidecar}"
        + (f", aseprite {ase}" if ase else "")
        + "\n"
    )


def _paths_from(stdout: str):
    first = stdout.splitlines()[-1]
    png = first.split("Wrote ", 1)[1].split(" (", 1)[0]
    dims = first.split(" (", 1)[1].split(")", 1)[0]
    w, h_ = dims.split("×")
    preview = first.split("preview ", 1)[1].split(", sidecar", 1)[0]
    rest = first.split("sidecar ", 1)[1]
    sidecar, _, ase = rest.partition(", aseprite ")
    for path in (png, preview, sidecar, ase):
        assert not path or Path(path).is_absolute(), path
    assert Path(sidecar).exists()
    return png, w, h_, preview, sidecar, ase or None


def test_stdout_is_unchanged(h, monkeypatch):
    assert hashlib.sha256(inspect.getsource(cli.print_summary).encode()).hexdigest() == (
        PRINT_SUMMARY_SHA
    )
    for extra in ([], ["-o", "sprite.png"], ["--aseprite"]):
        result = h.generate(extra)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == _wrote(_paths_from(result.stdout))
        assert PREFIX not in result.stdout

    first = h.generate()
    assert first.exit_code == 0
    plain = h.refine()
    assert plain.exit_code == 0, plain.stderr
    assert plain.stdout == _wrote(_paths_from(plain.stdout))
    assert PREFIX not in plain.stdout

    shown = h.generate(["--show-spec", "-o", "shown.png"])
    assert shown.exit_code == 0
    _symbol("print_usage")
    monkeypatch.setattr(cli, "print_usage", lambda _path: None)
    silent = h.generate(["--show-spec", "-o", "shown.png"])
    assert silent.exit_code == 0
    assert shown.stdout == silent.stdout
    assert "Resolved spec" in shown.stdout
    assert silent.stderr == ""


# --- AC5 ------------------------------------------------------------------------------------
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

_CELLS = {
    "github-actions": {"GITHUB_ACTIONS": "true", "TERM": "xterm-256color"},
    "force-color": {"FORCE_COLOR": "1", "TERM": "xterm-256color"},
    "github-actions-force-color": {
        "GITHUB_ACTIONS": "true",
        "FORCE_COLOR": "1",
        "TERM": "xterm-256color",
    },
    "no-color-force-color": {"NO_COLOR": "1", "FORCE_COLOR": "1", "TERM": "xterm-256color"},
    "dumb": {"TERM": "dumb"},
}


@pytest.mark.parametrize("cell", list(_CELLS))
def test_usage_line_has_no_ansi_under_ci_colour_forcing(tmp_path, cell):
    script_path = tmp_path / "script.json"
    counts_path = tmp_path / "counts.txt"
    script_path.write_text(json.dumps([{}, {"grid": _grid()}]))
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _SCRUBBED_ENV and not k.startswith("ANTHROPIC_")
    }
    env["PIXELGEN_HOME"] = str(tmp_path / "home")
    env["COLUMNS"] = "80"
    env["ANTHROPIC_API_KEY"] = "dummy-key"
    env.update(_CELLS[cell])

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _DRIVER,
            str(Path(__file__).resolve().parent),
            str(script_path),
            str(counts_path),
            "--",
            "generate",
            "a red apple",
            "--size",
            "16",
        ],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert proc.stderr.count("\x1b") == 0
    lines = proc.stderr.splitlines()
    assert lines
    assert lines[-1].startswith("pixelgen: usage:")
    sidecar = _sidecar_from(proc.stdout)
    assert payload(lines[-1]) == file_usage(sidecar)
    counts = dict(item.split("=") for item in counts_path.read_text().split())
    assert counts["anthropic_constructed"] == "0"


# --- AC6 ------------------------------------------------------------------------------------
def _cell(h, name):
    """Run one failure cell; returns the result."""
    if name == "scale-0":
        return h.generate(["--scale", "0"])
    if name == "bad-palette":
        return h.generate(["--palette", "nope"])
    if name == "bad-flag":
        return h.generate(["--no-such-flag"])
    if name == "unavailable-backend":
        return h.generate(["--backend", "diffusion"])
    if name == "no-key":
        return h.invoke(["generate", "a red apple", "--size", "16"], key=False)
    if name == "post-spend-llm-error":
        return h.invoke(["generate", "a red apple", "--size", "16"], {}, LLMError("boom"))
    if name == "last-json-directory":
        (h.tmp / "pixelgen-home" / "last.json").mkdir(parents=True)
        return h.generate()
    seeded = h.generate()
    assert seeded.exit_code == 0, seeded.stderr
    if name == "refine-missing-from":
        return h.invoke(["refine", "x", "--from", "missing.json"])
    if name == "refine-no-previous":
        (h.tmp / "pixelgen-home" / "last.json").unlink()
        return h.invoke(["refine", "add a hat"], {"spec": {}, "grid": _grid()})
    if name == "refine-no-key":
        return h.invoke(["refine", "add a hat"], key=False)
    if name == "refine-post-spend-llm-error":
        return h.invoke(["refine", "add a hat"], LLMError("boom"))
    raise AssertionError(name)


_FAILURES = [
    ("scale-0", 2, 0),
    ("bad-palette", 2, 0),
    ("bad-flag", 2, 0),
    ("unavailable-backend", 1, 0),
    ("no-key", 1, 0),
    ("post-spend-llm-error", 1, 2),
    ("last-json-directory", 1, 2),
    ("refine-missing-from", 2, 0),
    ("refine-no-previous", 1, 0),
    ("refine-no-key", 1, 0),
    ("refine-post-spend-llm-error", 1, 1),
]


@pytest.mark.parametrize(("name", "code", "calls"), _FAILURES, ids=[f[0] for f in _FAILURES])
def test_no_usage_line_on_failure(h, name, code, calls):
    result = _cell(h, name)

    assert (result.exit_code, len(_ulines(result.stderr)), h.llm.call_count) == (code, 0, calls)
    if code == 1 or name == "refine-missing-from":
        assert result.stderr.splitlines()[0].startswith("pixelgen: error:")
    if name == "last-json-directory":
        sidecars = sorted((h.tmp / "out").glob("*.json"))
        assert len(sidecars) == 1
        assert file_usage(sidecars[0])


@pytest.mark.parametrize("argv", [["--help"], ["--version"], ["palettes"], ["presets"]])
def test_no_usage_line_for_non_run_verbs(h, argv):
    result = runner.invoke(app, argv)

    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert _ulines(result.stderr) == []


# --- AC7 ------------------------------------------------------------------------------------
def test_usage_line_format(h):
    usage_line = _symbol("usage_line")
    assert usage_line([]) == "pixelgen: usage: []"
    records = [
        {
            "step": "generate",
            "provider": "p",
            "model": None,
            "unit": "tokens",
            "calls": 1,
            "quantity": 3,
        },
        {
            "step": "refine",
            "provider": "p",
            "model": "módel\nx",
            "unit": "tokens",
            "calls": 2,
            "quantity": 4,
        },
    ]
    line = usage_line(records)
    assert "\n" not in line
    assert line.isascii()
    assert "null" in line
    assert payload(line) == records
    assert [list(r) for r in payload(line)] == [list(r) for r in records]

    result = h.generate()
    assert result.exit_code == 0
    assert "$" not in result.stderr
    for record in payload(_ulines(result.stderr)[0]):
        assert set(record) == RECORD_KEYS
