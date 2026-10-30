"""CLI tests for ``--aseprite/--no-aseprite`` on generate and refine (PXG-29).

In-process ``CliRunner`` only, FakeLLM through the ``cli.make_llm`` seam, offline.
"""

import hashlib
import importlib.resources
import re
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest
import typer.main
from aseprite_oracle import parse_aseprite, to_sidecar_grid
from fakes import FakeLLM
from PIL import Image
from typer.testing import CliRunner

import pixelgen.cli as cli
import pixelgen.palettes
import pixelgen.pipeline as pipeline
import pixelgen.presets
from pixelgen.cli import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, app
from pixelgen.export import load_sidecar
from pixelgen.palettes import load_palette
from pixelgen.pixelate import grid_to_rgba

runner = CliRunner()
SLUG = "a-red-apple"


def _grid(seed: int = 0, side: int = 16, colours: int = 16) -> list[list[int]]:
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5 + seed) % colours
            for c in range(side)
        ]
        for r in range(side)
    ]


def assert_ase_matches_png(png: Path, ase: Path, palette: str = "pico8") -> None:
    """The .aseprite shows what the PNG shows (PXG-31); RGB under alpha 0 is ignored."""
    ours = np.array(Image.open(png).convert("RGBA"))
    grid = np.array(to_sidecar_grid(parse_aseprite(ase.read_bytes())), dtype=np.int64)
    theirs = grid_to_rgba(grid, load_palette(palette))
    assert ours.shape == theirs.shape
    both_clear = (ours[..., 3] == 0) & (theirs[..., 3] == 0)
    assert int(((ours != theirs).any(axis=2) & ~both_clear).sum()) == 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _names(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()}


def _shipped_palette_files() -> list[Path]:
    root = importlib.resources.files("pixelgen") / "palettes"
    return sorted(Path(str(p)) for p in root.iterdir() if p.name.endswith(".hex"))


def _write_big(path: Path) -> None:
    path.write_text("".join(f"{i:06x}\n" for i in range(256)), encoding="utf-8")


class Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.llm: FakeLLM | None = None
        self.make_llm_calls = 0
        monkeypatch.chdir(tmp_path)

    @property
    def out(self) -> Path:
        return self.tmp / "out"

    def invoke(self, args, *script, key=True):
        self.llm = FakeLLM(*script)
        if key:
            self.mp.setenv("ANTHROPIC_API_KEY", "dummy-key")
        else:
            self.mp.delenv("ANTHROPIC_API_KEY", raising=False)

        def fake_make():
            self.make_llm_calls += 1
            return self.llm

        self.mp.setattr(cli, "make_llm", fake_make)
        return runner.invoke(app, args)

    def generate(self, extra=(), *script, key=True, prompt="a red apple"):
        script = script or ({}, {"grid": _grid(1)})
        return self.invoke(["generate", prompt, "--size", "16", *extra], *script, key=key)

    def refine(self, extra=(), *script, key=True):
        script = script or ({"spec": {}, "grid": _grid(7)},)
        return self.invoke(["refine", "x", *extra], *script, key=key)

    def palette_world(self, *, all_big=False, preset=False) -> Path:
        d = self.tmp / "palworld"
        d.mkdir()
        for src in _shipped_palette_files():
            if all_big:
                _write_big(d / src.name)
            else:
                shutil.copy(src, d / src.name)
        _write_big(d / "big256.hex")
        self.mp.setattr(pixelgen.palettes, "_palette_dir", lambda: d)
        if preset:
            pd = self.tmp / "presetworld"
            pd.mkdir()
            real = importlib.resources.files("pixelgen") / "presets"
            for entry in real.iterdir():
                if entry.name.endswith(".yaml"):
                    (pd / entry.name).write_bytes(entry.read_bytes())
            (pd / "bigpal.yaml").write_text("palette: big256\n", encoding="utf-8")
            self.mp.setattr(pixelgen.presets, "_preset_dir", lambda: pd)
        return d


@pytest.fixture
def h(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


def assert_refusal(h, result, name):
    assert result.exit_code == EXIT_USAGE, (result.stdout, result.stderr)
    lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, result.stderr
    line = lines[0]
    assert line.startswith("pixelgen: error:")
    assert name in line
    assert "--aseprite" in line
    assert "255" in line
    assert h.llm.call_count == 0


def _strip(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# --- AC1 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["generate", "refine"])
def test_aseprite_flag_declared_on_both_verbs(verb):
    command = typer.main.get_command(app).commands[verb]
    param = next(p for p in command.params if p.name == "aseprite")
    assert param.opts == ["--aseprite"]
    assert param.secondary_opts == ["--no-aseprite"]
    assert param.default is False
    result = runner.invoke(app, [verb, "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"})
    assert result.exit_code == EXIT_OK
    assert "--aseprite" in _strip(result.stdout)


# --- AC2 ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "stem"),
    [
        (["-o", "out/x"], "x"),
        ([], SLUG),
        (["-o", "out/k.aseprite"], "k.aseprite"),
    ],
    ids=["explicit", "default", "dotted-aseprite"],
)
def test_generate_aseprite_writes_four_files(h, extra, stem):
    result = h.generate([*extra, "--aseprite"])
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    assert _names(h.out) == {f"{stem}.png", f"{stem}@8x.png", f"{stem}.json", f"{stem}.aseprite"}
    sidecar = load_sidecar(h.out / f"{stem}.json")
    assert_ase_matches_png(h.out / f"{stem}.png", h.out / f"{stem}.aseprite")
    assert sidecar.outputs.aseprite == f"{stem}.aseprite"
    png, preview = h.out / f"{stem}.png", h.out / f"{stem}@8x.png"
    side, ase_path = h.out / f"{stem}.json", h.out / f"{stem}.aseprite"
    assert result.stdout == (
        f"Wrote {png.resolve()} (16×16), preview {preview.resolve()}, "
        f"sidecar {side.resolve()}, aseprite {ase_path.resolve()}\n"
    )


# --- AC3 ----------------------------------------------------------------------------------


def test_refine_aseprite_is_explicit_never_inherited(h):
    gen = h.generate(["--aseprite"])
    assert gen.exit_code == EXIT_OK, (gen.stdout, gen.stderr)
    assert (h.out / f"{SLUG}.aseprite").exists()
    a = h.refine()
    assert a.exit_code == EXIT_OK, (a.stdout, a.stderr)
    assert not (h.out / f"{SLUG}-v2.aseprite").exists()
    v2 = load_sidecar(h.out / f"{SLUG}-v2.json")
    assert set(v2.outputs.model_dump()) == {"png", "preview", "scale"}


def test_refine_aseprite_is_explicit_never_inherited_flag_on(h):
    assert h.generate().exit_code == EXIT_OK
    b = h.refine(["--from", str(h.out / f"{SLUG}.json"), "--aseprite"])
    assert b.exit_code == EXIT_OK, (b.stdout, b.stderr)
    ase = h.out / f"{SLUG}-v2.aseprite"
    assert_ase_matches_png(h.out / f"{SLUG}-v2.png", ase)
    assert b.stdout.endswith(f", aseprite {ase.resolve()}\n")


# --- AC4 ----------------------------------------------------------------------------------


def test_default_name_steps_past_stray_aseprite(h, tmp_path):
    h.out.mkdir()
    stray = h.out / f"{SLUG}.aseprite"
    stray.write_bytes(b"stray\n")
    digest = _sha(stray)
    result = h.generate(["--aseprite"])
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    for ext in (".png", "@8x.png", ".json", ".aseprite"):
        assert (h.out / f"{SLUG}_2{ext}").exists(), ext
    assert not (h.out / f"{SLUG}.png").exists()
    assert _sha(stray) == digest
    assert h.llm.call_count == 2
    d = tmp_path / "unit"
    d.mkdir()
    (d / "x.aseprite").write_bytes(b"s")
    assert pipeline.next_free_output(d, "x", 8, aseprite=True) == d / "x_2"
    assert pipeline.next_free_output(d, "x", 8) == d / "x"


# --- AC5 ----------------------------------------------------------------------------------


def test_version_scan_steps_past_stray_aseprite(h, tmp_path):
    d = tmp_path / "unit"
    d.mkdir()
    (d / "knight-v2.aseprite").write_bytes(b"s")
    assert pipeline.next_version_base(d / "knight.json") == d / "knight-v3"
    (d / "knight-v5.aseprite").write_bytes(b"s")
    assert pipeline.next_version_base(d / "knight.json") == d / "knight-v6"

    assert h.generate().exit_code == EXIT_OK
    stray = h.out / f"{SLUG}-v2.aseprite"
    stray.write_bytes(b"stray\n")
    digest = _sha(stray)
    on = h.refine(["--from", str(h.out / f"{SLUG}.json"), "--aseprite"])
    assert on.exit_code == EXIT_OK, (on.stdout, on.stderr)
    for ext in (".png", "@8x.png", ".json", ".aseprite"):
        assert (h.out / f"{SLUG}-v3{ext}").exists(), ext
    assert not (h.out / f"{SLUG}-v2.png").exists()
    assert _sha(stray) == digest
    assert h.llm.call_count == 1


def test_version_scan_steps_past_stray_aseprite_flag_off(h):
    assert h.generate().exit_code == EXIT_OK
    (h.out / f"{SLUG}-v2.aseprite").write_bytes(b"stray\n")
    off = h.refine(["--from", str(h.out / f"{SLUG}.json")])
    assert off.exit_code == EXIT_OK, (off.stdout, off.stderr)
    for ext in (".png", "@8x.png", ".json"):
        assert (h.out / f"{SLUG}-v3{ext}").exists(), ext
    assert not (h.out / f"{SLUG}-v3.aseprite").exists()


# --- AC6 ----------------------------------------------------------------------------------


def test_explicit_output_overwrites_aseprite(h):
    h.out.mkdir()
    (h.out / "x.aseprite").write_bytes(b"old\n")
    result = h.generate(["-o", "out/x", "--aseprite"])
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    target = h.out / "x.aseprite"
    assert_ase_matches_png(h.out / "x.png", target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


# --- AC7 ----------------------------------------------------------------------------------


def test_explicit_palette_over_255_refused_before_spend(h):
    h.palette_world(preset=True)
    for key in (True, False):
        r = h.generate(["--palette", "big256", "--aseprite"], key=key)
        assert_refusal(h, r, "big256")
        assert h.make_llm_calls == 0
    r = h.generate(["--preset", "bigpal", "--aseprite"])
    assert_refusal(h, r, "big256")
    assert not h.out.exists() or not _names(h.out)
    ctl = h.generate(["--palette", "big256"])
    assert ctl.exit_code == EXIT_OK, (ctl.stdout, ctl.stderr)
    assert _names(h.out) == {f"{SLUG}.png", f"{SLUG}@8x.png", f"{SLUG}.json"}
    assert h.llm.call_count == 2
    # Control: a valid palette with the flag on and no key keeps the exit-1 key refusal.
    keyless = h.generate(["--aseprite"], key=False)
    assert keyless.exit_code == EXIT_RUNTIME
    assert h.llm.call_count == 0


# --- AC8 ----------------------------------------------------------------------------------


def test_inferred_palette_excludes_over_255(h):
    h.palette_world()
    shipped = ", ".join(p.stem for p in _shipped_palette_files())
    ok = h.generate(["--aseprite"], {"palette": "big256"}, {"grid": _grid(1)})
    assert ok.exit_code == EXIT_OK, (ok.stdout, ok.stderr)
    system = h.llm.calls[0].system
    assert "palette must be one of: " + shipped in system
    assert "big256" not in system
    assert load_sidecar(h.out / f"{SLUG}.json").spec.palette == "pico8"
    assert _names(h.out) == {f"{SLUG}.png", f"{SLUG}@8x.png", f"{SLUG}.json", f"{SLUG}.aseprite"}
    assert h.llm.call_count == 2


def test_inferred_palette_excludes_over_255_no_eligible(h):
    h.palette_world(all_big=True)
    r = h.generate(["--aseprite"])
    assert_refusal(h, r, "pico8")
    assert h.make_llm_calls == 0
    assert not h.out.exists() or not _names(h.out)


def test_inferred_palette_excludes_over_255_flag_off_control(h):
    h.palette_world()
    ctl = h.generate([], {"palette": "big256"}, {"grid": _grid(1)})
    assert ctl.exit_code == EXIT_OK, (ctl.stdout, ctl.stderr)
    assert "big256" in h.llm.calls[0].system


# --- AC9 ----------------------------------------------------------------------------------


def test_refine_source_palette_over_255_refused_before_spend(h):
    h.palette_world()
    src = h.generate(["--palette", "big256"])
    assert src.exit_code == EXIT_OK, (src.stdout, src.stderr)
    before = _names(h.out)
    calls_before = h.make_llm_calls
    a = h.refine(["--from", str(h.out / f"{SLUG}.json"), "--aseprite"])
    assert_refusal(h, a, "big256")
    b = h.refine(["--aseprite"], key=False)
    assert_refusal(h, b, "big256")
    assert _names(h.out) == before
    assert not list(h.out.glob("*-v2*"))
    assert h.make_llm_calls == calls_before
    ctl = h.refine(["--from", str(h.out / f"{SLUG}.json")])
    assert ctl.exit_code == EXIT_OK, (ctl.stdout, ctl.stderr)


# --- AC10 ---------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [[], ["--no-aseprite"]], ids=["default", "no-flag"])
def test_flag_off_output_set_and_stdout_unchanged(h, extra):
    result = h.generate(["-o", "out/x", *extra])
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    assert _names(h.out) == {"x.png", "x@8x.png", "x.json"}
    assert result.stdout == (
        f"Wrote {(h.out / 'x.png').resolve()} (16×16), "
        f"preview {(h.out / 'x@8x.png').resolve()}, "
        f"sidecar {(h.out / 'x.json').resolve()}\n"
    )
    sidecar = load_sidecar(h.out / "x.json")
    assert set(sidecar.outputs.model_dump()) == {"png", "preview", "scale"}
    assert "aseprite" not in (h.out / "x.json").read_text(encoding="utf-8")


def test_flag_off_stray_aseprite_is_not_a_collision(h):
    h.out.mkdir()
    stray = h.out / f"{SLUG}.aseprite"
    stray.write_bytes(b"stray\n")
    digest = _sha(stray)
    result = h.generate()
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    assert (h.out / f"{SLUG}.png").exists()
    assert not (h.out / f"{SLUG}_2.png").exists()
    assert _sha(stray) == digest


# --- AC13 ---------------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["generate", "refine"])
def test_aseprite_help_under_ci_colour_forcing(monkeypatch, verb):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, [verb, "--help"])
    assert result.exit_code == EXIT_OK
    text = _strip(result.stdout)
    assert "--aseprite" in text
    assert "--no-aseprite" in text
