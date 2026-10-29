"""CLI tests for ``pixelgen generate`` (PXG-11). Everything runs on FakeLLM, offline."""

import errno
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest
import typer.main
from fakes import FakeLLM
from PIL import Image
from typer.testing import CliRunner

import pixelgen.cli as cli
import pixelgen.pipeline as pipeline
from pixelgen.cli import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, app
from pixelgen.errors import (
    BackendUnavailableError,
    GridValidationError,
    LLMError,
    LLMOutputError,
    OutputExistsError,
    PaletteError,
    PresetError,
    SpecError,
)
from pixelgen.export import load_sidecar
from pixelgen.palettes import load_palette
from pixelgen.pixelate import grid_to_rgba, pixelate

runner = CliRunner()


def _grid(seed: int = 0, side: int = 16, colours: int = 16) -> list[list[int]]:
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5 + seed) % colours
            for c in range(side)
        ]
        for r in range(side)
    ]


def _bad_grid() -> dict:
    return {"grid": [[0] * 15 for _ in range(16)]}


class Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.llm: FakeLLM | None = None
        self.make_llm_calls = 0
        monkeypatch.chdir(tmp_path)

    def run(self, args, *script, key=True):
        self.llm = FakeLLM(*script)
        if key:
            self.mp.setenv("ANTHROPIC_API_KEY", "dummy-key")

        def fake_make():
            self.make_llm_calls += 1
            return self.llm

        self.mp.setattr(cli, "make_llm", fake_make)
        return runner.invoke(app, ["generate", *args])

    def png(self, name="a-red-apple"):
        return self.tmp / "out" / f"{name}.png"

    def sidecar(self, name="a-red-apple"):
        return load_sidecar(self.tmp / "out" / f"{name}.json")

    def pixels(self, name="a-red-apple"):
        with Image.open(self.png(name)) as img:
            assert img.mode == "RGBA"
            return np.array(img)


@pytest.fixture
def h(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


def assert_clean_failure(result, code, *fragments):
    assert result.exit_code == code, (result.stdout, result.stderr, result.exception)
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)
    lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, result.stderr
    assert lines[0].startswith("pixelgen: error:")
    for fragment in fragments:
        assert fragment in lines[0]
    return lines[0]


def _spec_none(sidecar):
    return sidecar.spec.model_copy(update={"dither": "none"})


# --- AC1 ------------------------------------------------------------------------------------


def test_flags_generate_declares_every_documented_flag():
    command = typer.main.get_command(app).commands["generate"]
    declared = {opt for p in command.params for opt in (*p.opts, *p.secondary_opts)}
    expected = {
        "--size", "--width", "--height", "--palette", "--preset", "--dither", "--outline",
        "--no-outline", "--background", "--view", "--seed", "--scale", "-o", "--output",
        "--backend", "--show-spec",
    }  # fmt: skip
    assert expected <= declared
    by_name = {p.name: p for p in command.params}
    assert by_name["scale"].default == 8
    assert by_name["backend"].default == "llm-grid"
    assert by_name["outline"].default is None


# --- AC2 / AC18 -----------------------------------------------------------------------------


def test_e2e_red_apple_writes_palette_only_png_preview_and_sidecar(h):
    grid = _grid()
    result = h.run(["a red apple", "--size", "16", "--palette", "pico8"], {}, {"grid": grid})
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    pal = load_palette("pico8")
    sidecar = h.sidecar()
    assert (sidecar.spec.width, sidecar.spec.height, sidecar.spec.palette) == (16, 16, "pico8")
    expected = pixelate(grid_to_rgba(np.array(grid), pal), _spec_none(sidecar), pal)
    got = h.pixels()
    assert got.shape == (16, 16, 4)
    assert (expected[..., 3] == 0).sum() >= 1
    assert len({tuple(p) for p in expected.reshape(-1, 4) if p[3]}) >= 2
    assert (got[..., 3] == 0).sum() == (expected[..., 3] == 0).sum()
    assert np.array_equal(got, expected)
    palette_set = {tuple(c) for c in pal}
    assert {tuple(p[:3]) for p in got.reshape(-1, 4) if p[3]} <= palette_set
    with Image.open(h.tmp / "out" / "a-red-apple@8x.png") as preview:
        assert preview.size == (128, 128)
    assert "a-red-apple.png" in result.stdout
    assert h.llm.call_count == 2


def test_dither_llm_grid_quantizes_with_none_and_sidecar_keeps_bayer(h):
    grid = _grid()
    result = h.run(
        ["a red apple", "--size", "16", "--palette", "pico8", "--dither", "bayer"],
        {},
        {"grid": grid},
    )
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    pal = load_palette("pico8")
    sidecar = h.sidecar()
    assert sidecar.spec.dither == "bayer"
    rgba = grid_to_rgba(np.array(grid), pal)
    with_bayer = pixelate(rgba, sidecar.spec, pal)
    with_none = pixelate(rgba, _spec_none(sidecar), pal)
    assert (with_bayer != with_none).any()  # non-vacuity: the grid is sensitive to dither
    assert np.array_equal(h.pixels(), with_none)


# --- AC3 / AC4 ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "inferred", "field", "expected"),
    [
        (["--preset", "gameboy", "--palette", "pico8"], {}, "palette", "pico8"),
        (["--preset", "gameboy"], {}, "palette", "gameboy"),
        (["--preset", "gameboy"], {"dither": "floyd-steinberg"}, "dither", "floyd-steinberg"),
        (
            ["--preset", "gameboy", "--dither", "none"],
            {"dither": "floyd-steinberg"},
            "dither",
            "none",
        ),
    ],
)
def test_precedence_flag_over_inferred_over_preset(h, flags, inferred, field, expected):
    grid = _grid(colours=4)  # valid for every palette used here, gameboy included
    result = h.run(["a red apple", "--size", "16", *flags], inferred, {"grid": grid})
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    assert getattr(h.sidecar().spec, field) == expected


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], True), (["--no-outline"], False), (["--preset", "gameboy", "--outline"], True)],
)
def test_outline_unset_is_not_false(h, flags, expected):
    result = h.run(["a red apple", "--size", "16", *flags], {}, {"grid": _grid(colours=4)})
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    assert h.sidecar().spec.outline is expected


# --- AC5 ------------------------------------------------------------------------------------


def test_dims_size_and_width_height_and_scale(h):
    assert h.run(["a red apple", "--size", "16"], {}, {"grid": _grid()}).exit_code == 0
    assert h.pixels().shape == (16, 16, 4)
    wide = [[0] * 24 for _ in range(16)]
    r = h.run(["wide", "--width", "24", "--height", "16", "-o", "out/wide"], {}, {"grid": wide})
    assert r.exit_code == 0, (r.stdout, r.stderr)
    assert h.pixels("wide").shape == (16, 24, 4)
    r = h.run(
        ["a red apple", "--size", "16", "--scale", "3", "-o", "out/s3"], {}, {"grid": _grid()}
    )
    assert r.exit_code == 0
    with Image.open(h.tmp / "out" / "s3@3x.png") as preview:
        assert preview.size == (48, 48)


@pytest.mark.parametrize("extra", [["--width", "20"], ["--height", "20"]])
def test_dims_size_conflict_is_a_usage_error_even_without_a_key(h, extra):
    result = h.run(["x", "--size", "16", *extra], key=False)
    assert_clean_failure(result, EXIT_USAGE, "--size")
    assert h.llm.call_count == 0 and h.make_llm_calls == 0
    assert not (h.tmp / "out").exists()


def test_dims_scale_zero_is_a_usage_error(h):
    result = h.run(["x", "--size", "16", "--scale", "0"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_USAGE, "--scale")
    assert h.llm.call_count == 0


# --- AC6 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "fragments"),
    [
        (["--palette", "nope"], ["nope", "pico8"]),
        (["--preset", "nope"], ["unknown preset", "gameboy"]),
        (["--dither", "sparkle"], ["invalid spec", "dither"]),
        (["--size", "7"], ["invalid spec", "width"]),
    ],
)
def test_usage_values_fail_with_exit_2_before_any_llm_call(h, flags, fragments):
    result = h.run(["x", *flags], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_USAGE, *fragments)
    assert h.llm.call_count == 0 and h.make_llm_calls == 0
    assert not (h.tmp / "out").exists()


# --- AC7 ------------------------------------------------------------------------------------


def test_show_spec_prints_the_table_and_still_generates(h):
    result = h.run(
        ["a red apple", "--size", "16", "--palette", "pico8", "--show-spec"], {}, {"grid": _grid()}
    )
    assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
    assert "Resolved spec" in result.stdout and "pico8" in result.stdout
    assert h.png().is_file()


def test_show_spec_absent_without_the_flag(h):
    result = h.run(["a red apple", "--size", "16"], {}, {"grid": _grid()})
    assert result.exit_code == 0
    assert "Resolved spec" not in result.stdout


def test_show_spec_is_printed_before_generation_fails(h):
    result = h.run(["x", "--size", "16", "--show-spec"], {}, _bad_grid(), _bad_grid(), _bad_grid())
    assert "Resolved spec" in result.stdout
    assert_clean_failure(result, EXIT_RUNTIME)


# --- AC8 ------------------------------------------------------------------------------------


def test_missing_key_exits_1_before_make_llm(h):
    result = h.run(["x", "--size", "16"], {}, {"grid": _grid()}, key=False)
    assert_clean_failure(result, EXIT_RUNTIME, "ANTHROPIC_API_KEY")
    assert h.make_llm_calls == 0 and h.llm.call_count == 0


def test_missing_key_blank_key_counts_as_missing(h, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    result = h.run(["x", "--size", "16"], {}, key=False)
    assert_clean_failure(result, EXIT_RUNTIME, "ANTHROPIC_API_KEY")
    assert h.make_llm_calls == 0


def test_missing_key_subprocess_has_no_traceback(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["PIXELGEN_HOME"] = str(tmp_path / "home")
    proc = subprocess.run(
        [sys.executable, "-c", "from pixelgen.cli import app; app(prog_name='pixelgen')",
         "generate", "x"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "pixelgen: error:" in proc.stderr and "ANTHROPIC_API_KEY" in proc.stderr
    assert "Traceback" not in proc.stderr and "Traceback" not in proc.stdout


# --- AC9 ------------------------------------------------------------------------------------


def test_retries_exhausted_is_a_one_line_runtime_failure(h):
    result = h.run(["x", "--size", "16"], {}, _bad_grid(), _bad_grid(), _bad_grid())
    assert_clean_failure(result, EXIT_RUNTIME, "after 3 attempts")
    assert h.llm.call_count == 4
    assert not (h.tmp / "out").exists()


# --- AC10 -----------------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [PaletteError, PresetError, SpecError])
def test_boundary_pre_validation_errors_exit_2(h, monkeypatch, exc):
    def boom(*a, **kw):
        raise exc("bad input")

    monkeypatch.setattr(pipeline, "build_spec", boom)
    result = h.run(["x", "--size", "16"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_USAGE, "bad input")
    assert h.llm.call_count == 0 and h.make_llm_calls == 0


@pytest.mark.parametrize(
    "exc",
    [
        PaletteError,
        PresetError,
        SpecError,
        LLMError,
        LLMOutputError,
        GridValidationError,
        BackendUnavailableError,
        OutputExistsError,
    ],
)
def test_boundary_post_phase_errors_exit_1(h, monkeypatch, exc):
    def boom(*a, **kw):
        raise exc("late failure")

    monkeypatch.setattr(pipeline, "run_generate", boom)
    result = h.run(["x", "--size", "16"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_RUNTIME, "late failure")


def test_boundary_collapses_whitespace_to_one_line(h, monkeypatch):
    def boom(*a, **kw):
        raise LLMError("first\nsecond")

    monkeypatch.setattr(pipeline, "run_generate", boom)
    result = h.run(["x", "--size", "16"], {})
    line = assert_clean_failure(result, EXIT_RUNTIME)
    assert line == "pixelgen: error: first second"


def test_boundary_empty_message_falls_back_to_default_message(h, monkeypatch):
    def boom(*a, **kw):
        raise LLMError("")

    monkeypatch.setattr(pipeline, "run_generate", boom)
    result = h.run(["x", "--size", "16"], {})
    assert_clean_failure(result, EXIT_RUNTIME, LLMError.default_message)


def test_boundary_exit_code_constants():
    assert (EXIT_OK, EXIT_RUNTIME, EXIT_USAGE) == (0, 1, 2)


def test_boundary_keeps_discovery_commands_working():
    for args in (["--version"], ["palettes"], ["presets"]):
        assert runner.invoke(app, args).exit_code == 0


def test_boundary_does_not_swallow_programming_errors(h, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("bug")

    monkeypatch.setattr(pipeline, "run_generate", boom)
    result = h.run(["x", "--size", "16"], {})
    assert isinstance(result.exception, RuntimeError)


# --- AC11 -----------------------------------------------------------------------------------


def test_diffusion_or_backend_diffusion_is_refused_before_any_llm_call(h):
    result = h.run(["x", "--size", "16", "--backend", "diffusion"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_RUNTIME, "diffusion")
    assert h.llm.call_count == 0
    assert not (h.tmp / "out").exists()


def test_diffusion_or_backend_unknown_backend_is_a_usage_error(h):
    result = h.run(["x", "--size", "16", "--backend", "nope"], {})
    assert_clean_failure(result, EXIT_USAGE, "llm-grid")
    assert h.llm.call_count == 0


# --- AC12 / AC13 ----------------------------------------------------------------------------


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_name_never_clobbers_an_earlier_result(h):
    assert h.run(["a red apple", "--size", "16"], {}, {"grid": _grid(0)}).exit_code == 0
    first = _sha(h.png())
    assert h.run(["a red apple", "--size", "16"], {}, {"grid": _grid(1)}).exit_code == 0
    assert _sha(h.png()) == first
    for suffix in (".png", "@8x.png", ".json"):
        assert (h.tmp / "out" / f"a-red-apple_2{suffix}").is_file()


def test_explicit_output_overwrites(h):
    args = ["a red apple", "--size", "16", "-o", "out/apple.png"]
    assert h.run(args, {}, {"grid": _grid(0)}).exit_code == 0
    first = _sha(h.png("apple"))
    result = h.run(args, {}, {"grid": _grid(1)})
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert _sha(h.png("apple")) != first
    assert not (h.tmp / "out" / "apple_2.png").exists()


def test_explicit_output_uppercase_png_does_not_clobber_lowercase(h):
    out = h.tmp / "out"
    out.mkdir()
    (out / "x.png").write_bytes(b"precious\n")
    before = hashlib.sha256(b"precious\n").hexdigest()
    result = h.run(["a red apple", "--size", "16", "-o", "out/x.PNG"], {}, {"grid": _grid(0)})
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert _sha(out / "x.png") == before
    with Image.open(out / "x.PNG.png") as img:
        assert img.mode == "RGBA"
        assert img.size == (16, 16)
    assert (out / "x.PNG@8x.png").is_file()
    assert (out / "x.PNG.json").is_file()
    assert not (out / "x@8x.png").exists()
    assert not (out / "x.json").exists()
    assert "x.PNG.png" in result.stdout
    assert h.llm.call_count == 2


def test_explicit_output_existing_directory_is_a_usage_error(h):
    (h.tmp / "somedir").mkdir()
    result = h.run(["x", "--size", "16", "-o", "somedir"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_USAGE)
    assert h.llm.call_count == 0


# --- X-PXG-38 fix-forward 1: pre-spend preflight and OSError boundary ----------------------


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_preflight_parent_file_default_name_refuses_before_spend(h):
    (h.tmp / "out").write_text("not a directory")
    before = _tree(h.tmp)
    result = h.run(["a red apple", "--size", "16"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_RUNTIME, "out")
    assert h.llm.call_count == 0
    assert h.make_llm_calls == 0
    assert _tree(h.tmp) == before


def test_preflight_parent_file_explicit_output_refuses_before_spend(h):
    (h.tmp / "afile").write_text("not a directory")
    before = _tree(h.tmp)
    result = h.run(["a red apple", "--size", "16", "-o", "afile/x.png"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_RUNTIME, "afile")
    assert "not a directory" in result.stderr
    assert h.llm.call_count == 0
    assert h.make_llm_calls == 0
    assert _tree(h.tmp) == before


def test_preflight_parent_file_subprocess_never_reaches_the_key_check(tmp_path):
    (tmp_path / "out").write_text("not a directory")
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["PIXELGEN_HOME"] = str(tmp_path / "home")
    proc = subprocess.run(
        [sys.executable, "-c", "from pixelgen.cli import app; app(prog_name='pixelgen')",
         "generate", "a red apple"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "pixelgen: error:" in proc.stderr and "out" in proc.stderr
    assert "ANTHROPIC_API_KEY" not in proc.stderr
    assert "Traceback" not in proc.stderr and "Traceback" not in proc.stdout


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda p: PermissionError(13, "Permission denied", p),
        lambda p: OSError(errno.ENAMETOOLONG, "File name too long", p),
    ],
)
def test_boundary_oserror_is_one_clean_line(h, monkeypatch, make_exc):
    def boom(*a, **kw):
        raise make_exc("out/target.png")

    monkeypatch.setattr(pipeline, "run_generate", boom)
    result = h.run(["x", "--size", "16"], {})
    assert isinstance(result.exception, SystemExit)
    line = assert_clean_failure(result, EXIT_RUNTIME, "out/target.png")
    assert make_exc("p").strerror in line


def test_boundary_oserror_real_filesystem_name_too_long(h):
    long_name = "a" * 300
    result = h.run(["x", "--size", "16", "-o", f"out/{long_name}.png"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_RUNTIME)


def test_boundary_bug_visible_runtime_error_is_not_swallowed(h, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("bug")

    monkeypatch.setattr(pipeline, "run_generate", boom)
    result = h.run(["x", "--size", "16"], {})
    assert isinstance(result.exception, RuntimeError)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_preflight_readonly_directory_refuses_before_spend(h):
    ro = h.tmp / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        result = h.run(["a red apple", "--size", "16", "-o", "ro/x.png"], {}, {"grid": _grid()})
    finally:
        ro.chmod(0o755)
    assert_clean_failure(result, EXIT_RUNTIME, "ro")
    assert h.llm.call_count == 0
    assert h.make_llm_calls == 0


def test_preflight_home_file_refuses_before_spend(h, monkeypatch):
    home = h.tmp / "homefile"
    home.write_text("not a directory")
    monkeypatch.setenv("PIXELGEN_HOME", str(home))
    result = h.run(["a red apple", "--size", "16"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_RUNTIME, str(home))
    assert h.llm.call_count == 0
    assert h.make_llm_calls == 0
    assert not (h.tmp / "out").exists()


def test_preflight_collision_exhaustion_refuses_before_spend(h, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_COLLISION_INDEX", 3)
    out = h.tmp / "out"
    out.mkdir()
    for name in ("a-red-apple", "a-red-apple_2", "a-red-apple_3"):
        (out / f"{name}.png").write_bytes(b"x")
    result = h.run(["a red apple", "--size", "16"], {}, {"grid": _grid()})
    assert_clean_failure(result, EXIT_RUNTIME, "no free output name")
    assert h.llm.call_count == 0
    assert h.make_llm_calls == 0


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_is_a_usage_error_before_the_key_check(h, prompt):
    result = h.run([prompt], {}, key=False)
    assert_clean_failure(result, EXIT_USAGE, "prompt is empty")
    assert h.make_llm_calls == 0


def test_diffusion_or_backend_refusal_needs_no_key_and_builds_no_client(h):
    result = h.run(["x", "--size", "16", "--backend", "diffusion"], {}, key=False)
    line = assert_clean_failure(result, EXIT_RUNTIME, "diffusion")
    assert "ANTHROPIC_API_KEY" not in line
    assert h.make_llm_calls == 0


def test_diffusion_or_backend_refusal_with_key_never_calls_make_llm(h):
    result = h.run(["x", "--size", "16", "--backend", "diffusion"], {})
    assert_clean_failure(result, EXIT_RUNTIME, "diffusion")
    assert h.make_llm_calls == 0
    assert h.llm.call_count == 0
