"""PXG-43: deeply nested JSON in a refine source is one error line, before spend."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fakes import FakeLLM
from typer.testing import CliRunner

import pixelgen
import pixelgen.cli as cli
import pixelgen.pipeline as pipeline
from pixelgen.cli import app
from pixelgen.export import TokenUsage
from pixelgen.palettes import load_palette
from pixelgen.spec import ArtSpec

runner = CliRunner()

# Python 3.14 bounds JSON recursion by the real C stack, so the depth at which
# ``json.loads`` overflows follows the stack size; 3.11-3.13 use a counter (3.11 about
# 1,000 levels, 3.12/3.13 about 10,000) and need a stack big enough to reach it, or they
# segfault (a 1 MiB thread crashed 3.12 and 3.13). Pin one roomy stack: in-process runs
# use a thread with this stack, the child process gets it as RLIMIT_STACK.
STACK_BYTES = 8 * 1024 * 1024
# 1,000,000 levels would need under 8.4 bytes of stack per level to fit in 8 MiB, which
# no build manages (the CI 3.14 build fitted 100,000 levels, so under 84 bytes per level).
# The margin claim is only what the module run on 3.11-3.14 shows.
DEEP = 1_000_000
THREAD_STACK_BYTES = STACK_BYTES
SHAPES = ("array", "object")
PRODUCT_REASONS = (
    "nests too deeply to parse",
    "sidecar must be a JSON object",
    "has no schema_version",
    "last.json must be a JSON object",
)
_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)


def run_on_fixed_stack(fn, stack=THREAD_STACK_BYTES):
    """Run ``fn()`` in a thread whose stack is ``stack`` bytes, whatever the host ulimit."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised in the caller
            box["error"] = exc

    previous = threading.stack_size(stack)
    try:
        thread = threading.Thread(target=target)
        thread.start()
        thread.join()
    finally:
        threading.stack_size(previous)
    if "error" in box:
        raise box["error"]
    return box["value"]


def _limit_stack():
    """preexec_fn for children: pin RLIMIT_STACK to STACK_BYTES (POSIX only)."""
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    limit = STACK_BYTES if hard == resource.RLIM_INFINITY else min(STACK_BYTES, hard)
    resource.setrlimit(resource.RLIMIT_STACK, (limit, hard))


def _child_preexec():
    try:
        import resource  # noqa: F401
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    return _limit_stack


def _nested(shape: str, n: int) -> str:
    if shape == "array":
        return "[" * n + "]" * n
    return '{"a":' * n + "0" + "}" * n


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_clean_failure(result, code, *fragments):
    assert result.exit_code == code, (result.stdout, result.stderr)
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)
    lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, result.stderr[:2000]
    assert lines[0].startswith("pixelgen: error:")
    for fragment in fragments:
        assert fragment in lines[0]
    return lines[0]


class Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.home = tmp_path / "pixelgen-home"
        self.home.mkdir(exist_ok=True)
        self.make_llm_calls = 0
        self.llm: FakeLLM | None = None
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PIXELGEN_HOME", str(self.home))

    @property
    def last(self) -> Path:
        return self.home / "last.json"

    def seed_sidecar(self) -> Path:
        spec = ArtSpec(subject="knight", width=8, height=8, palette="pico8")
        grid = [[(r * 3 + c * 5 + 1) % 16 for c in range(8)] for r in range(8)]
        result = pipeline.render_grid(
            spec,
            grid,
            load_palette(spec.palette),
            output=self.tmp / "out" / "knight",
            scale=8,
            overwrite=False,
            backend="llm-grid",
            model="fake-model",
            token_usage=TokenUsage(input_tokens=1, output_tokens=1, calls=1),
        )
        return result.export.sidecar_path

    def refine(self, from_=None, stack=THREAD_STACK_BYTES):
        self.llm = FakeLLM()
        self.mp.setenv("ANTHROPIC_API_KEY", "dummy-key")

        def fake_make():
            self.make_llm_calls += 1
            return self.llm

        self.mp.setattr(cli, "make_llm", fake_make)
        args = ["refine", "x"]
        if from_ is not None:
            args += ["--from", str(from_)]
        return run_on_fixed_stack(lambda: runner.invoke(app, args), stack)

    def assert_no_spend(self):
        assert self.make_llm_calls == 0
        assert self.llm is None or self.llm.call_count == 0
        assert not [p for p in self.tmp.rglob("*-v2*")]


@pytest.fixture
def h(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


def _source(h: Harness, source: str, text: str) -> tuple[Path, Path | None]:
    """Write ``text`` as the refine source; returns (file, --from arg or None)."""
    if source == "from":
        path = h.tmp / "d.json"
        path.write_text(text, encoding="utf-8")
        return path, path
    h.last.write_text(text, encoding="utf-8")
    return h.last, None


@pytest.mark.parametrize("shape", SHAPES)
def test_from_deep_json_is_one_error_line(h, shape):
    path = h.tmp / "d.json"
    path.write_text(_nested(shape, DEEP), encoding="utf-8")
    result = h.refine(from_=path)
    assert_clean_failure(result, 2, str(path), "nests too deeply to parse")
    assert not h.last.exists()
    h.assert_no_spend()


@pytest.mark.parametrize("shape", SHAPES)
def test_last_json_deep_json_is_one_error_line(h, shape):
    h.last.write_text(_nested(shape, DEEP), encoding="utf-8")
    before = _digest(h.last)
    result = h.refine()
    assert_clean_failure(result, 1, str(h.last), "nests too deeply to parse")
    assert _digest(h.last) == before
    h.assert_no_spend()


@pytest.mark.parametrize("depth", [500, 990, 1100, 5000, DEEP])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("source", ["from", "last"])
def test_depth_sweep_is_one_line(h, source, shape, depth):
    path, from_arg = _source(h, source, _nested(shape, depth))
    before = _digest(path)
    result = h.refine(from_=from_arg)
    line = assert_clean_failure(result, 2 if source == "from" else 1, str(path))
    assert any(reason in line for reason in PRODUCT_REASONS), line
    assert _digest(path) == before
    h.assert_no_spend()


@pytest.mark.parametrize("depth", [500, 9000, 60_000])
@pytest.mark.parametrize("source", ["from", "last"])
def test_deep_schema_version_is_one_bounded_line(h, source, depth):
    raw_version = "[" * depth + "]" * depth
    if source == "from":
        text = '{"schema_version":' + raw_version + "}"
    else:
        text = (
            '{"schema_version":1,"sidecar_path":"s.json","sidecar":{"schema_version":'
            + raw_version
            + "}}"
        )
    path, from_arg = _source(h, source, text)
    before = _digest(path)
    # Not an overflow test: the roomy 8 MiB stack lets these inputs parse so that
    # _check_version (D2) is what bounds the line. On a build where a cell still cannot
    # parse, D1 turns it into "nests too deeply"; that cell then proves only D1.
    result = h.refine(from_=from_arg, stack=STACK_BYTES)
    line = assert_clean_failure(result, 2 if source == "from" else 1, str(path))
    assert "unsupported" in line or "nests too deeply" in line, line[:300]
    assert len(line) < len(str(path)) + 200
    assert _digest(path) == before
    h.assert_no_spend()


def test_parseable_deep_value_keeps_validation_message(h):
    sidecar = h.seed_sidecar()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["x"] = json.loads("[" * 500 + "]" * 500)
    src = h.tmp / "deep-extra.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    h.last.write_text("seeded", encoding="utf-8")
    before = _digest(h.last)
    result = h.refine(from_=src)
    line = assert_clean_failure(result, 2, str(src), "invalid sidecar")
    assert "nests too deeply" not in line
    assert _digest(h.last) == before
    h.assert_no_spend()


def _child_env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _KEY_VARS}
    env["PIXELGEN_HOME"] = str(home)
    env["NO_COLOR"] = "1"
    return env


@pytest.mark.parametrize("source", ["from", "last"])
def test_real_process_deep_json(tmp_path, source):
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "pxhome"
    home.mkdir()
    env = _child_env(home)
    where = subprocess.run(
        [sys.executable, "-I", "-c", "import pixelgen;print(pixelgen.__file__)"],
        capture_output=True,
        text=True,
        env={**env, "PYTHONPATH": os.pathsep.join(sys.path)},
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    assert Path(where.stdout.strip()).resolve().is_relative_to(repo_root), where
    assert Path(pixelgen.__file__).resolve().is_relative_to(repo_root)
    target = tmp_path / "deep.json" if source == "from" else home / "last.json"
    target.write_text(_nested("array", DEEP), encoding="utf-8")
    before = _digest(target)
    cmd = [str(Path(sys.executable).parent / "pixelgen"), "refine", "x"]
    if source == "from":
        cmd += ["--from", str(target)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=60,
        stdin=subprocess.DEVNULL,
        preexec_fn=_child_preexec(),
    )
    assert proc.returncode == (2 if source == "from" else 1), proc.stderr[:2000]
    assert "Traceback" not in proc.stderr
    assert "Traceback" not in proc.stdout
    lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, proc.stderr[:2000]
    assert lines[0].startswith("pixelgen: error:")
    assert "nests too deeply to parse" in lines[0]
    assert _digest(target) == before
