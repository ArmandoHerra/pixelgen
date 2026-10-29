"""CLI tests for ``pixelgen refine`` (PXG-12). Everything runs on FakeLLM, offline."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import typer.main
from fakes import FakeLLM
from PIL import Image
from typer.testing import CliRunner

import pixelgen.cli as cli
import pixelgen.pipeline as pipeline
from pixelgen.backends.llm_grid import LLMGridGenerator
from pixelgen.cli import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, app
from pixelgen.errors import LLMError
from pixelgen.export import TokenUsage, load_last, load_sidecar
from pixelgen.palettes import load_palette
from pixelgen.spec import ArtSpec

runner = CliRunner()


def _grid(seed: int = 0, side: int = 8) -> list[list[int]]:
    """Valid pico8 grid; any two seeds that differ mod 16 differ in every cell."""
    return [[(r * 3 + c * 5 + seed) % 16 for c in range(side)] for r in range(side)]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(directory: Path) -> dict[str, str]:
    return {p.name: _digest(p) for p in sorted(directory.iterdir()) if p.is_file()}


def assert_clean_failure(result, code, *fragments):
    assert result.exit_code == code, (result.stdout, result.stderr)
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)
    lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, result.stderr
    assert lines[0].startswith("pixelgen: error:")
    for fragment in fragments:
        assert fragment in lines[0]
    return lines[0]


class Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.llm: FakeLLM | None = None
        self.make_llm_calls = 0
        self.refine_calls: list[dict] = []
        monkeypatch.chdir(tmp_path)

        original = LLMGridGenerator.refine
        calls = self.refine_calls

        def spy(gen, previous_spec, previous_grid, palette, instruction):
            calls.append({"previous_grid": previous_grid, "instruction": instruction})
            return original(gen, previous_spec, previous_grid, palette, instruction)

        monkeypatch.setattr(LLMGridGenerator, "refine", spy)

    def seed(self, name="knight", directory=None, grid=None, scale=8, **spec_fields):
        """Write a source via ``render_grid`` (no LLM); returns the sidecar path."""
        directory = directory or self.tmp / "out"
        fields = {"subject": name, "width": 8, "height": 8, "palette": "pico8", **spec_fields}
        spec = ArtSpec(**fields)
        palette = load_palette(spec.palette)
        grid = grid if grid is not None else _grid(1)
        result = pipeline.render_grid(
            spec,
            grid,
            palette,
            output=directory / name,
            scale=scale,
            overwrite=False,
            backend="llm-grid",
            model="fake-model",
            token_usage=TokenUsage(input_tokens=1, output_tokens=1, calls=1),
        )
        return result.export.sidecar_path

    def invoke(self, args, *script, key=True):
        self.llm = FakeLLM(*script)
        if key:
            self.mp.setenv("ANTHROPIC_API_KEY", "dummy-key")

        def fake_make():
            self.make_llm_calls += 1
            return self.llm

        self.mp.setattr(cli, "make_llm", fake_make)
        return runner.invoke(app, args)

    def refine(self, instruction, *script, from_=None, key=True):
        args = ["refine", instruction]
        if from_ is not None:
            args += ["--from", str(from_)]
        return self.invoke(args, *script, key=key)

    def png(self, path: Path):
        with Image.open(path) as img:
            assert img.mode == "RGBA"
            return np.array(img)


@pytest.fixture
def h(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


def _edit(grid, **spec):
    return {"spec": spec, "grid": grid}


# --- AC1 ------------------------------------------------------------------------------------


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == EXIT_OK
    for name in ("generate", "refine", "palettes", "presets"):
        assert name in result.stdout
    assert set(typer.main.get_command(app).commands) == {
        "generate",
        "refine",
        "palettes",
        "presets",
    }


# --- AC2 / AC3 ------------------------------------------------------------------------------


def test_refine_after_generate_writes_v2(h):
    v1_grid, v2_grid = _grid(1), _grid(7)
    gen = h.invoke(
        ["generate", "a tiny knight with a red cape", "--size", "8", "--palette", "pico8",
         "-o", str(h.tmp / "out" / "knight.png")],
        {},
        {"grid": v1_grid},
    )  # fmt: skip
    assert gen.exit_code == EXIT_OK, (gen.stdout, gen.stderr)
    out = h.tmp / "out"
    before = _snapshot(out)

    result = h.refine("make the cape blue", _edit(v2_grid))
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    for name in ("knight-v2.png", "knight-v2@8x.png", "knight-v2.json"):
        assert (out / name).exists(), name
    sidecar = load_sidecar(out / "knight-v2.json")
    assert sidecar.grid == v2_grid
    assert sidecar.grid != load_sidecar(out / "knight.json").grid
    palette = {tuple(c) for c in load_palette("pico8")}
    for pixel in h.png(out / "knight-v2.png").reshape(-1, 4):
        assert pixel[3] == 0 or tuple(pixel[:3]) in palette
    after = _snapshot(out)
    assert {k: after[k] for k in before} == before
    assert "knight-v2.png" in result.stdout
    assert "make the cape blue" in h.llm.calls[-1].user
    assert h.llm.call_count == 1


def test_second_refine_builds_on_v2(h):
    h.seed("knight", grid=_grid(1))
    out = h.tmp / "out"
    v2_grid, v3_grid = _grid(5), _grid(9)
    first = h.refine("make the cape blue", _edit(v2_grid))
    assert first.exit_code == EXIT_OK, (first.stdout, first.stderr)
    assert load_last().sidecar_path == out / "knight-v2.json"
    second = h.refine("add a sword", _edit(v3_grid))
    assert second.exit_code == EXIT_OK, (second.stdout, second.stderr)
    for name in ("knight-v3.png", "knight-v3@8x.png", "knight-v3.json"):
        assert (out / name).exists(), name
    user = h.llm.calls[-1].user
    assert json.dumps(v2_grid, separators=(",", ":")) in user
    assert json.dumps(_grid(1), separators=(",", ":")) not in user
    assert h.refine_calls[-1]["previous_grid"] == load_sidecar(out / "knight-v2.json").grid
    assert load_last().sidecar_path == out / "knight-v3.json"


# --- AC4 / AC5 / AC6 ------------------------------------------------------------------------


def test_from_overrides_last_json(h):
    slime = h.seed("slime", directory=h.tmp / "b", grid=_grid(2), scale=4)
    h.seed("apple", directory=h.tmp / "a", grid=_grid(3), scale=8)
    a_before = _snapshot(h.tmp / "a")
    result = h.refine("add eyes", _edit(_grid(11)), from_=slime)
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    for name in ("slime-v2.png", "slime-v2@4x.png", "slime-v2.json"):
        assert (h.tmp / "b" / name).exists(), name
    assert _snapshot(h.tmp / "a") == a_before
    assert h.refine_calls[-1]["previous_grid"] == _grid(2)
    assert load_last().sidecar_path == h.tmp / "b" / "slime-v2.json"


def test_from_v2_yields_v3(h):
    h.seed("knight", grid=_grid(1))
    v2 = h.tmp / "out" / "knight-v2"
    pipeline.render_grid(
        ArtSpec(subject="knight", width=8, height=8),
        _grid(2),
        load_palette("pico8"),
        output=v2,
        scale=8,
        overwrite=False,
        backend="llm-grid",
        model="fake-model",
        token_usage=TokenUsage(),
    )
    result = h.refine("x", _edit(_grid(4)), from_=h.tmp / "out" / "knight-v2.json")
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    out = h.tmp / "out"
    for name in ("knight-v3.png", "knight-v3@8x.png", "knight-v3.json"):
        assert (out / name).exists(), name
    assert not [p for p in out.iterdir() if "-v2-v2" in p.name]


def test_next_version_skips_existing(h):
    source = h.seed("knight", grid=_grid(1))
    out = h.tmp / "out"
    pipeline.render_grid(
        ArtSpec(subject="knight", width=8, height=8),
        _grid(2),
        load_palette("pico8"),
        output=out / "knight-v2",
        scale=8,
        overwrite=False,
        backend="llm-grid",
        model="m",
        token_usage=TokenUsage(),
    )
    (out / "knight-v3@8x.png").write_bytes(b"orphan")
    before = _snapshot(out)
    result = h.refine("x", _edit(_grid(6)), from_=source)
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    for name in ("knight-v4.png", "knight-v4@8x.png", "knight-v4.json"):
        assert (out / name).exists(), name
    after = _snapshot(out)
    assert {k: after[k] for k in before} == before


@pytest.mark.parametrize(
    ("source", "present", "expected"),
    [
        ("knight.json", [], "knight-v2"),
        ("knight-v2.json", [], "knight-v3"),
        ("knight-v9.json", ["knight-v10.json"], "knight-v11"),
        ("knight-v10.json", [], "knight-v11"),
        ("fighter-v1.json", [], "fighter-v1-v2"),
        ("a-v01.json", [], "a-v01-v2"),
        ("item10.json", [], "item10-v2"),
        ("knight.json", ["knight-v3@4x.png"], "knight-v4"),
    ],
)
def test_next_version_base_grammar(tmp_path, source, present, expected):
    for name in present:
        (tmp_path / name).write_bytes(b"x")
    assert pipeline.next_version_base(tmp_path / source) == tmp_path / expected


def test_next_version_base_grammar_missing_dir(tmp_path):
    missing = tmp_path / "nope" / "deeper"
    assert pipeline.next_version_base(missing / "slime.json") == missing / "slime-v2"


# --- AC7 ------------------------------------------------------------------------------------


def test_refine_renders_without_overwrite(h, monkeypatch):
    h.seed("knight")
    seen = []
    original = pipeline.render_grid

    def spy(*args, **kwargs):
        seen.append(kwargs["overwrite"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "render_grid", spy)
    result = h.refine("x", _edit(_grid(6)))
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    assert seen == [False]


def test_refine_collision_is_clean_failure(h, monkeypatch):
    source = h.seed("knight", grid=_grid(1))
    out = h.tmp / "out"
    first = h.refine("first", _edit(_grid(5)))
    assert first.exit_code == EXIT_OK
    monkeypatch.setattr(pipeline, "next_version_base", lambda _p: out / "knight-v2")
    before = _snapshot(out)
    last_before = _digest(load_last_path())
    result = h.refine("again", _edit(_grid(9)), from_=source)
    assert_clean_failure(result, EXIT_RUNTIME, "refusing to overwrite existing file", "knight-v2")
    assert _snapshot(out) == before
    assert _digest(load_last_path()) == last_before


def load_last_path() -> Path:
    return Path(os.environ["PIXELGEN_HOME"]) / "last.json"


# --- AC8 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["a", "b", "c", "d", "e"])
def test_refine_source_errors(h, case):
    expected = {
        "a": (EXIT_RUNTIME, ["no previous result", "--from"]),
        "b": (EXIT_USAGE, ["missing.json"]),
        "c": (EXIT_USAGE, ["notjson.json"]),
        "d": (EXIT_USAGE, ["schema_version"]),
        "e": (EXIT_RUNTIME, ["last.json"]),
    }[case]
    from_ = None
    if case == "b":
        from_ = h.tmp / "missing.json"
    elif case == "c":
        from_ = h.tmp / "notjson.json"
        from_.write_text("not json")
    elif case == "d":
        src = h.seed("knight")
        data = json.loads(src.read_text())
        data["schema_version"] = 2
        from_ = h.tmp / "v2schema.json"
        from_.write_text(json.dumps(data))
    elif case == "e":
        home = load_last_path().parent
        home.mkdir(parents=True, exist_ok=True)
        load_last_path().write_text("not json")
    result = h.refine("x", _edit(_grid(6)), from_=from_, key=False)
    assert_clean_failure(result, expected[0], *expected[1])
    assert h.make_llm_calls == 0
    assert h.llm.call_count == 0
    assert not list(h.tmp.rglob("*-v2*"))


def test_refine_source_ok_without_key_fails_on_key(h):
    source = h.seed("knight")
    result = h.refine("x", _edit(_grid(6)), from_=source, key=False)
    assert_clean_failure(result, EXIT_RUNTIME, "ANTHROPIC_API_KEY")
    assert h.make_llm_calls == 0
    assert not list(h.tmp.rglob("*-v2*"))


# --- AC9 ------------------------------------------------------------------------------------


def test_bare_refine_after_output_dir_deleted(h):
    h.seed("knight", grid=_grid(1))
    shutil.rmtree(h.tmp / "out")
    result = h.refine("add a hat", _edit(_grid(6)))
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    for name in ("knight-v2.png", "knight-v2@8x.png", "knight-v2.json"):
        assert (h.tmp / "out" / name).exists(), name
    assert h.refine_calls[-1]["previous_grid"] == _grid(1)


# --- AC10 -----------------------------------------------------------------------------------


def test_refine_locks_and_edits(h):
    h.seed("knight", outline=True, preset=None, notes=None)
    edit = _edit(
        _grid(6),
        width=16,
        palette="gameboy",
        preset="nes",
        outline=False,
        notes="blue cape",
    )
    result = h.refine("make the cape blue", edit)
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    spec = load_sidecar(h.tmp / "out" / "knight-v2.json").spec
    assert (spec.width, spec.height, spec.palette) == (8, 8, "pico8")
    assert spec.preset is None
    assert spec.outline is False
    assert spec.notes == "blue cape"
    assert h.png(h.tmp / "out" / "knight-v2.png").shape == (8, 8, 4)
    notes = [ln for ln in result.stderr.splitlines() if ln.startswith("pixelgen: note:")]
    assert len(notes) == 1
    assert "refine keeps the source's" in notes[0]
    for field in ("preset", "width", "palette"):
        assert field in notes[0]


# --- AC11 -----------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["bad_spec", "bad_grid"])
def test_invalid_refine_output_is_clean_failure(h, kind):
    h.seed("knight", grid=_grid(1))
    out = h.tmp / "out"
    if kind == "bad_spec":
        reply = _edit(_grid(6), dither="sparkle")
    else:
        reply = _edit([[0] * 8 for _ in range(7)])
    last_before = _digest(load_last_path())
    before = _snapshot(out)
    result = h.refine("x", reply, reply, reply)
    assert_clean_failure(result, EXIT_RUNTIME, "after 3 attempts")
    assert h.llm.call_count == 3
    assert _snapshot(out) == before
    assert not list(out.glob("knight-v2*"))
    assert _digest(load_last_path()) == last_before


# --- AC12 / AC13 ----------------------------------------------------------------------------


def test_refine_llm_error_is_clean_failure(h):
    h.seed("knight")
    result = h.refine("x", LLMError("upstream unavailable"))
    assert_clean_failure(result, EXIT_RUNTIME, "upstream unavailable")
    assert not list(h.tmp.rglob("*-v2*"))


def test_refine_blank_instruction(h):
    h.seed("knight")
    result = h.refine("   ", _edit(_grid(6)))
    assert_clean_failure(result, EXIT_USAGE, "refine needs a non-empty INSTRUCTION")
    assert h.llm.call_count == 0


# --- extra: preflight (X-PXG-38 rule 3) -----------------------------------------------------


def test_refine_preflight_refuses_unwritable_output_before_spend(h, monkeypatch):
    source = h.seed("knight")
    blocker = h.tmp / "afile"
    blocker.write_text("x")
    monkeypatch.setattr(pipeline, "next_version_base", lambda _p: blocker / "knight-v2")
    result = h.refine("x", _edit(_grid(6)), from_=source)
    assert_clean_failure(result, EXIT_RUNTIME, "cannot write output")
    assert h.make_llm_calls == 0
    assert h.llm.call_count == 0


def test_refine_preflight_refuses_unwritable_home_before_spend(h, monkeypatch):
    source = h.seed("knight")
    blocker = h.tmp / "home-is-a-file"
    blocker.write_text("x")
    monkeypatch.setenv("PIXELGEN_HOME", str(blocker))
    result = h.refine("x", _edit(_grid(6)), from_=source)
    assert_clean_failure(result, EXIT_RUNTIME, "pixelgen home")
    assert h.make_llm_calls == 0
    assert h.llm.call_count == 0
    assert not list((h.tmp / "out").glob("knight-v2*"))


def test_refine_help_shows_from_option():
    command = typer.main.get_command(app).commands["refine"]
    declared = {opt for p in command.params for opt in p.opts}
    assert "--from" in declared
    result = runner.invoke(app, ["refine", "--help"])
    assert result.exit_code == EXIT_OK
    assert "--from" in result.stdout
