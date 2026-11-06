"""CLI tests for the raster base-image path (PXG-35, AC8-AC12).

A fake raster backend is registered under ``fake-raster`` and driven through the real CLI
(``CliRunner`` through ``cli.app``), offline. ``render_result`` is fetched inside the
tests that need it so a run against the base ``src/`` collects.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeRasterGenerator, raster_gradient
from PIL import Image
from test_cli_aseprite import Harness, _grid, assert_ase_matches_png

import pixelgen.backends.base
import pixelgen.pipeline as pipeline
from pixelgen.backends.base import GenerationResult, GenerationUsage
from pixelgen.export import load_sidecar
from pixelgen.palettes import load_palette
from pixelgen.pixelate import grid_to_rgba, rgba_to_grid

ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(__file__).resolve().parent / "aseprite_oracle.py"
KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)


def _sprite(spec):
    img = raster_gradient(600, 400, 4)
    img[:, :200, 3] = 0
    return img


def _opaque(spec):
    return raster_gradient()


@pytest.fixture
def rh(monkeypatch, tmp_path):
    monkeypatch.setitem(
        pixelgen.backends.base._REGISTRY, "fake-raster", "fakes:FakeRasterGenerator"
    )
    monkeypatch.setattr(FakeRasterGenerator, "log", [])
    monkeypatch.setattr(FakeRasterGenerator, "image_factory", _sprite)
    return Harness(monkeypatch, tmp_path)


def _run(rh, name, *flags, factory=None, aseprite=False):
    if factory is not None:
        rh.mp.setattr(FakeRasterGenerator, "image_factory", factory)
    args = ["generate", "a red apple", "--backend", "fake-raster", "-o", str(rh.out / name)]
    if aseprite:
        args.append("--aseprite")
    if not any(f in ("--size", "--width") for f in flags):
        args += ["--size", "32"]
    return rh.invoke([*args, *flags], {})


def _png(path):
    return np.array(Image.open(path).convert("RGBA"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _palette_exact(img, palette):
    assert img.dtype == np.uint8 and img.shape[2] == 4
    assert set(np.unique(img[..., 3]).tolist()) <= {0, 255}
    assert grid_to_rgba(rgba_to_grid(img, palette), palette).tobytes() == img.tobytes()


# --------------------------------------------------------------------------- AC8

AC8_ROWS = [
    ([], "none", (32, 32)),
    (["--dither", "floyd-steinberg"], "floyd-steinberg", (32, 32)),
    (["--dither", "bayer"], "bayer", (32, 32)),
    (["--size", "48"], "none", (48, 48)),
    (["--width", "40", "--height", "24"], "none", (40, 24)),
    (["--no-outline"], "none", (32, 32)),
    (["--background", "solid"], "none", (32, 32)),
    (["--no-outline", "--background", "solid"], "none", (32, 32)),
]


@pytest.mark.parametrize(
    "flags,dither,size", AC8_ROWS, ids=[" ".join(r[0]) or "default" for r in AC8_ROWS]
)
def test_cli_raster_writes_palette_exact_outputs(rh, tmp_path, flags, dither, size):
    result = _run(rh, "x", *flags)
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert {p.name for p in rh.out.iterdir()} == {"x.png", "x@8x.png", "x.json"}
    palette = load_palette("pico8")
    png = _png(rh.out / "x.png")
    assert png.shape == (size[1], size[0], 4)
    _palette_exact(png, palette)
    preview = _png(rh.out / "x@8x.png")
    assert preview.tobytes() == np.repeat(np.repeat(png, 8, axis=0), 8, axis=1).tobytes()

    sc = load_sidecar(rh.out / "x.json")
    assert sc.schema_version == 2
    assert sc.backend == "fake-raster"
    assert sc.spec.dither == dither

    redo = pipeline.render_grid(
        sc.spec,
        sc.grid,
        load_palette(sc.spec.palette),
        output=tmp_path / "rerender" / "y",
        scale=8,
        overwrite=False,
        backend="llm-grid",
        model=None,
        token_usage=sc.token_usage,
    )
    assert redo.image.tobytes() == png.tobytes()
    if not flags:
        grid_img = grid_to_rgba(np.asarray(sc.grid), palette)
        assert grid_img.tobytes() != png.tobytes()

    assert rh.llm.call_count == 1
    assert len(FakeRasterGenerator.log) == 1
    assert sc.token_usage.calls == 2


# --------------------------------------------------------------------------- AC9


def test_cli_raster_honours_dither(rh):
    hashes = []
    for i, dither in enumerate(("none", "floyd-steinberg", "bayer")):
        result = _run(rh, f"d{i}", "--background", "solid", "--dither", dither, factory=_opaque)
        assert result.exit_code == 0, (result.stdout, result.stderr)
        hashes.append(_sha(rh.out / f"d{i}.png"))
    assert len(set(hashes)) == 3


# --------------------------------------------------------------------------- AC10

AC10_ROWS = [
    ("default", [], "pico8"),
    ("fs", ["--dither", "floyd-steinberg"], "pico8"),
    ("bayer", ["--dither", "bayer"], "pico8"),
    ("outline", ["--outline"], "pico8"),
    ("nooutline", ["--no-outline"], "pico8"),
    ("solid", ["--background", "solid"], "pico8"),
    ("transparent", ["--background", "transparent"], "pico8"),
    ("size48", ["--size", "48"], "pico8"),
    ("gameboy", ["--palette", "gameboy"], "gameboy"),
    ("wh", ["--width", "40", "--height", "24"], "pico8"),
]


def test_cli_raster_aseprite_parity(rh, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARS}
    env["PIXELGEN_HOME"] = str(tmp_path / "oracle-home")
    for name, flags, palette in AC10_ROWS:
        result = _run(rh, name, *flags, aseprite=True)
        assert result.exit_code == 0, (name, result.stdout, result.stderr)
        png, ase = rh.out / f"{name}.png", rh.out / f"{name}.aseprite"
        proc = subprocess.run(
            [sys.executable, str(ORACLE), str(ase), "--png", str(png)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (name, proc.stdout, proc.stderr)
        assert_ase_matches_png(png, ase, palette)
    default = _sha(rh.out / "default.png")
    for name in ("nooutline", "solid", "bayer", "gameboy"):
        assert _sha(rh.out / f"{name}.png") != default, name


# --------------------------------------------------------------------------- AC11

OUTLINE = (0, 0, 0, 255)


def _ring(img):
    return int((img[:, :11] == OUTLINE).all(axis=2).sum())


def test_refine_from_raster_output(rh, monkeypatch):
    from pixelgen.backends.llm_grid import LLMGridGenerator

    result = _run(rh, "x", "--outline")
    assert result.exit_code == 0, (result.stdout, result.stderr)
    sc = load_sidecar(rh.out / "x.json")
    calls = []
    original = LLMGridGenerator.refine

    def spy(gen, previous_spec, previous_grid, palette, instruction):
        calls.append({"previous_grid": previous_grid})
        return original(gen, previous_spec, previous_grid, palette, instruction)

    monkeypatch.setattr(LLMGridGenerator, "refine", spy)
    refined = rh.invoke(
        ["refine", "x", "--from", str(rh.out / "x.json")], {"spec": {}, "grid": sc.grid}
    )
    assert refined.exit_code == 0, (refined.stdout, refined.stderr)
    v1, v2 = _png(rh.out / "x.png"), _png(rh.out / "x-v2.png")
    _palette_exact(v2, load_palette("pico8"))
    assert len(calls) == 1
    assert calls[0]["previous_grid"] == sc.grid
    assert len(FakeRasterGenerator.log) == 1
    assert load_sidecar(rh.out / "x-v2.json").backend == "llm-grid"
    assert _ring(v1) == 32
    assert _ring(v2) == _ring(v1)


# --------------------------------------------------------------------------- AC12

EXAMPLES = (
    "knight/knight",
    "knight/knight-v2",
    "red-apple/red-apple",
    "slime/slime",
)
EXAMPLES_SHA = "4d26f398b04ba4ee218edca24907d1b84e445d50d1641e79987f3f04c5868947"


def test_llm_grid_examples_replay_byte_identical(tmp_path):
    render_result = getattr(pipeline, "render_result", None)
    assert render_result is not None, "pipeline.render_result does not exist"
    committed = []
    for rel in EXAMPLES:
        stem = Path(rel).name
        base = ROOT / "examples" / rel
        sc = load_sidecar(base.with_suffix(".json"))
        w, h = len(sc.grid[0]), len(sc.grid)
        generated = GenerationResult(grid=sc.grid, generated_size=(w, h), usage=GenerationUsage())
        render_result(
            sc.spec,
            generated,
            load_palette(sc.spec.palette),
            output=tmp_path / stem,
            scale=sc.outputs.scale,
            overwrite=False,
            backend=sc.backend,
            model=sc.model,
            token_usage=sc.token_usage,
        )
        for name in (f"{stem}.png", f"{stem}@{sc.outputs.scale}x.png"):
            assert _sha(tmp_path / name) == _sha(base.parent / name), name
            committed.append(base.parent / name)
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in committed)).hexdigest()
    assert digest == EXAMPLES_SHA


C0 = (
    "cd6ecb82fb6c74d4b8b2096f16029d32163516ff6a0c79019badb1f9aea1ab1d",
    "277134b84cdb40c6b1059276332e79f8e942ba852ec04bc327373b171c9fd758",
)
CLI_ROWS = [
    ("C0", [], {}, C0),
    ("C1", ["--dither", "bayer"], {}, C0),
    ("C2", ["--dither", "floyd-steinberg"], {}, C0),
    (
        "C3",
        ["--no-outline"],
        {},
        (
            "49f97b90b7baa2b819cc6d2088ae1ef13298f4f055f5dd2e119d6dbfa6743984",
            "9050a2b6c3d99646d8d7aa5b35f8fa57ba8f40f4d2947c70708ecb36261f628b",
        ),
    ),
    (
        "C4",
        ["--background", "solid"],
        {},
        (
            "6e1743b891fe7ececf71d94773525c4c1af42b4caf5daea90d3ec8fd93009184",
            "e30f4a6f6ac2a6a43c63c71badd7ec5f8a9f27ded2dd7b5358c341ec1ef74fcc",
        ),
    ),
    (
        "C5",
        ["--palette", "grayscale4"],
        {"colours": 4},
        (
            "399f8b55a9a961d861209b265d3db288e394313fa5ca1c5a33bc3a59d6f600d8",
            "65d9958baee6af09b0e4517014700ae952a5b4bd76106a266ce0b8b57767d6fe",
        ),
    ),
    (
        "C6",
        ["--size", "48"],
        {"side": 32},
        (
            "ca0e2049c8db68786f0755063de044f795efb37667c3ca668f08d86c9ed11197",
            "9ae18490f01c173da3de50cdb887f3930201ae8f3c2bf41111b238c32183db14",
        ),
    ),
]


def test_llm_grid_cli_bytes_frozen(monkeypatch, tmp_path):
    assert getattr(pipeline, "render_result", None) is not None, "render_result does not exist"
    h = Harness(monkeypatch, tmp_path)
    for label, flags, grid_kw, (png_sha, preview_sha) in CLI_ROWS:
        args = ["generate", "a red apple"]
        if "--size" not in flags:
            args += ["--size", "16"]
        args += [*flags, "-o", str(h.out / label)]
        result = h.invoke(args, {}, {"grid": _grid(0, **grid_kw)})
        assert result.exit_code == 0, (label, result.stdout, result.stderr)
        assert _sha(h.out / f"{label}.png") == png_sha, label
        assert _sha(h.out / f"{label}@8x.png") == preview_sha, label
