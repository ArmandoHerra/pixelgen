"""PXG-31: the ``.aseprite`` encodes the image the PNG holds after ``pixelate()``.

In-process only (GR-018): ``CliRunner`` with FakeLLM, and the Aseprite binary through
``tests/aseprite_local.py`` for the local-only arm. ``_aseprite_indices`` is looked up inside the
one test that needs it, so a base-``src/`` run fails per test, not at collection (AC6).
"""

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from aseprite_local import require_aseprite, run_aseprite
from aseprite_oracle import parse_aseprite, to_sidecar_grid
from PIL import Image
from test_aseprite_writer import canonical_form
from test_cli_aseprite import Harness, _grid

import pixelgen.pipeline as pipeline
from pixelgen.cli import EXIT_OK
from pixelgen.errors import AsepriteError
from pixelgen.export import TokenUsage, export
from pixelgen.palettes import list_palettes, load_palette
from pixelgen.pixelate import grid_to_rgba, pixelate
from pixelgen.spec import ArtSpec

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_aseprite.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_aseprite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_aseprite"] = module  # dataclasses resolve their module through here
    spec.loader.exec_module(module)
    return module


va = _load_script()


@pytest.fixture
def h(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


def _png_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def _ase_rgba(path: Path, palette: np.ndarray) -> np.ndarray:
    grid = np.array(to_sidecar_grid(parse_aseprite(path.read_bytes())), dtype=np.int64)
    return grid_to_rgba(grid, palette)


def _differing(a: np.ndarray, b: np.ndarray) -> int:
    assert a.shape == b.shape
    both_clear = (a[..., 3] == 0) & (b[..., 3] == 0)
    return int(((a != b).any(axis=2) & ~both_clear).sum())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- AC1 ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outline", "background", "side"),
    [
        (True, "transparent", 16),
        (True, "solid", 16),
        (False, "transparent", 16),
        (False, "solid", 16),
        (True, "solid", 48),
    ],
    ids=[
        "outline-transparent",
        "outline-solid",
        "plain-transparent",
        "plain-solid",
        "outline-solid-48",
    ],
)
def test_render_grid_aseprite_matches_png(tmp_path, outline, background, side):
    palette = load_palette("pico8")
    spec = ArtSpec(
        subject="x",
        width=side,
        height=side,
        palette="pico8",
        outline=outline,
        background=background,
    )
    result = pipeline.render_grid(
        spec,
        _grid(1, side=side),
        palette,
        output=tmp_path / "x",
        scale=2,
        overwrite=False,
        backend="llm-grid",
        model=None,
        token_usage=TokenUsage(),
        aseprite=True,
    )
    differing = _differing(
        _png_rgba(result.export.png_path), _ase_rgba(result.export.aseprite_path, palette)
    )
    assert differing == 0, f"{differing} px differ between the PNG and the .aseprite"


# --- AC2 ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verb", "solid"),
    [("generate", False), ("generate", True), ("refine", False), ("refine", True)],
    ids=["generate-default", "generate-solid", "refine-default", "refine-solid"],
)
def test_cli_generate_and_refine_aseprite_match_png(h, verb, solid):
    palette = load_palette("pico8")
    bg = ["--background", "solid"] if solid else []
    if verb == "generate":
        result = h.generate(["-o", "out/x", "--aseprite", *bg])
        assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
        assert h.llm.call_count == 2
        stem = "x"
    else:
        first = h.generate(["-o", "out/x", *bg])
        assert first.exit_code == EXIT_OK, (first.stdout, first.stderr)
        result = h.refine(["--from", str(h.out / "x.json"), "--aseprite"])
        assert result.exit_code == EXIT_OK, (result.stdout, result.stderr)
        assert h.llm.call_count == 1
        stem = "x-v2"
    differing = _differing(
        _png_rgba(h.out / f"{stem}.png"), _ase_rgba(h.out / f"{stem}.aseprite", palette)
    )
    assert differing == 0, f"{differing} px differ between the PNG and the .aseprite"


# --- AC4 ----------------------------------------------------------------------------------


def test_real_aseprite_save_as_matches_png(tmp_path):
    exe = require_aseprite()
    palette = load_palette("pico8")
    result = pipeline.render_grid(
        ArtSpec(subject="x", width=16, height=16, palette="pico8"),
        _grid(1),
        palette,
        output=tmp_path / "x",
        scale=2,
        overwrite=False,
        backend="llm-grid",
        model=None,
        token_usage=TokenUsage(),
        aseprite=True,
    )
    run = run_aseprite(exe, ["-b", "x.aseprite", "--save-as", "x.ase.png"], root=tmp_path)
    assert (tmp_path / "x.ase.png").is_file(), (run.stdout, run.stderr)
    differing = _differing(_png_rgba(result.export.png_path), _png_rgba(tmp_path / "x.ase.png"))
    assert differing == 0, f"{differing} px differ between our PNG and Aseprite's"


# --- AC5 ----------------------------------------------------------------------------------

FROZEN_RGBA = {
    "outline": {
        "png": "a0afcc04555f27cf3ed579ee19382bb695b1ffc32498cfc3be3599fd984751b3",
        "preview": "b49f2e1818e782f9dc730ba26ec0925e30a0861b656f96d94ada5c609b78006d",
        "ase": "8becec7682426a6d01474922b6774fff40aeed01a37ee6258b69e7a3cf7d717d",
    },
    "plain": {
        "png": "e4b480db1372dd54e2fdef02be1dc7199bd86c4cac40fba135f0b061c1ffdece",
        "preview": "fbacee320cd576cf2aa7860d7eb0dcc28d69095de04ab97bebe933519286ec52",
        "ase": "7815a9a8d42d6c97e7224c531b0b45dcaed157ac77f3c58b7797481dce653202",
    },
}


def _export_pair(tmp_path, outline: bool):
    palette = load_palette("pico8")
    grid = _grid(1)
    spec = ArtSpec(subject="x", width=16, height=16, palette="pico8", outline=outline)
    image = pixelate(grid_to_rgba(np.asarray(grid), palette), spec, palette)
    paths, lasts = {}, {}
    for stem, flag in (("off", False), ("on", True)):
        paths[stem] = export(
            tmp_path / stem,
            image,
            spec=spec,
            grid=grid,
            palette_colors=palette,
            model=None,
            backend="llm-grid",
            token_usage=TokenUsage(),
            scale=8,
            now=NOW,
            aseprite=flag,
        )
        lasts[stem] = json.loads(paths[stem].last_path.read_text(encoding="utf-8"))
    return paths, lasts


def _decoded_sha(path: Path) -> str:
    return _sha(np.array(Image.open(path).convert("RGBA")).tobytes())


def _map_to_off(sidecar: dict, *, drop_aseprite: bool) -> dict:
    out = json.loads(json.dumps(sidecar))
    if drop_aseprite:
        out["outputs"].pop("aseprite")
    out["outputs"]["png"] = out["outputs"]["png"].replace("on", "off", 1)
    out["outputs"]["preview"] = out["outputs"]["preview"].replace("on", "off", 1)
    return out


@pytest.mark.parametrize("kind", ["outline", "plain"])
def test_flag_off_and_plain_bytes_frozen(tmp_path, kind):
    paths, lasts = _export_pair(tmp_path, outline=kind == "outline")
    off, on = paths["off"], paths["on"]
    pinned = FROZEN_RGBA[kind]
    assert _decoded_sha(off.png_path) == pinned["png"]
    assert _decoded_sha(off.preview_path) == pinned["preview"]
    assert off.png_path.read_bytes() == on.png_path.read_bytes()
    assert off.preview_path.read_bytes() == on.preview_path.read_bytes()
    side_off = json.loads(off.sidecar_path.read_text(encoding="utf-8"))
    side_on = json.loads(on.sidecar_path.read_text(encoding="utf-8"))
    assert _map_to_off(side_on, drop_aseprite=True) == side_off
    last_on = dict(lasts["on"])
    last_on["sidecar_path"] = str(off.sidecar_path)
    last_on["sidecar"] = _map_to_off(lasts["on"]["sidecar"], drop_aseprite=True)
    assert last_on == lasts["off"]
    assert _sha(canonical_form(on.aseprite_path.read_bytes())) == pinned["ase"]


VERIFY_PINS = {
    "pico8-16x16-index0": "1e53dc44f6326b55cf60b2f5cf6c5f715de1519670eb3ecdd7fd54012888c0b4",
    "gameboy-8x8-border": "bd51ff38c540b95d118f315817d2cab135d8d29d73aa4d1fe5a352041241fa9d",
    "nes-128x128-hashed": "6edfe8f73b5b1ea8813f66f3f922a065b659eb04ee3871e811fe211e1146e78f",
    "endesga32-17x33": "32d191d9ea959a0ed045d93cd76305431258b09ca77530669e2d919ab3dec22c",
    "grayscale4-16x16-solid": "a74cdf86be8764e5db48dd6d661a98e3a6cf066dfd9e43c39f6c9fdf9b80d7ce",
}


def test_verify_aseprite_pins_cover_every_case():
    assert set(VERIFY_PINS) == set(va.CASE_NAMES)


@pytest.mark.parametrize("name", sorted(VERIFY_PINS))
def test_verify_aseprite_cases_bytes_frozen(tmp_path, name):
    g, pal = va.build_case(name)
    spec = ArtSpec(subject="x", width=len(g[0]), height=len(g), palette="pico8")
    result = export(
        tmp_path / "c",
        grid_to_rgba(np.asarray(g), pal),
        spec=spec,
        grid=g,
        palette_colors=pal,
        model=None,
        backend="llm-grid",
        token_usage=TokenUsage(),
        scale=1,
        now=NOW,
        aseprite=True,
    )
    assert _sha(canonical_form(result.aseprite_path.read_bytes())) == VERIFY_PINS[name]


# --- AC6 ----------------------------------------------------------------------------------


def _small_export(tmp_path, rgba, palette, grid=None):
    spec = ArtSpec(subject="x", width=8, height=8, palette="pico8")
    return export(
        tmp_path / "t",
        rgba,
        spec=spec,
        grid=grid if grid is not None else [[2] * 8 for _ in range(8)],
        palette_colors=palette,
        model=None,
        backend="llm-grid",
        token_usage=TokenUsage(),
        scale=1,
        now=NOW,
        aseprite=True,
    )


@pytest.mark.parametrize(
    ("arm", "pixel", "message"),
    [
        ("partial-alpha", None, "cannot write .aseprite: pixel (1, 2) has partial alpha 128"),
        (
            "off-palette",
            (1, 2, 3, 255),
            "cannot write .aseprite: pixel (1, 2) is opaque and not in the palette",
        ),
    ],
    ids=["partial-alpha", "off-palette"],
)
def test_aseprite_indices_refuses_unmappable(tmp_path, arm, pixel, message):
    palette = load_palette("pico8")
    rgba = grid_to_rgba(np.full((8, 8), 2, dtype=np.int64), palette)
    rgba[2, 1] = pixel if pixel is not None else (*palette[2], 128)
    with pytest.raises(AsepriteError) as excinfo:
        _small_export(tmp_path, rgba, palette)
    assert message in str(excinfo.value)
    assert not list(tmp_path.glob("t*"))


DUP_PALETTE = np.array([(0, 0, 0), (9, 9, 9), (5, 5, 5), (9, 9, 9)], dtype=np.uint8)


def _dup_grid() -> list[list[int]]:
    grid = [[-1] * 8 for _ in range(8)]
    grid[0][:2] = [3, 1]
    grid[1][:2] = [2, 0]
    return grid


def test_aseprite_indices_keeps_raw_index_on_duplicate_palette(tmp_path):
    grid = _dup_grid()
    rgba = grid_to_rgba(np.asarray(grid), DUP_PALETTE)
    result = _small_export(tmp_path, rgba, DUP_PALETTE, grid)
    read = to_sidecar_grid(parse_aseprite(result.aseprite_path.read_bytes()))
    assert [row[:2] for row in read[:2]] == [[3, 1], [2, 0]]


def test_aseprite_indices_lowest_index_on_duplicate_palette(tmp_path):
    grid = _dup_grid()
    rgba = grid_to_rgba(np.asarray(grid), DUP_PALETTE)
    rgba[0, 2] = (9, 9, 9, 255)  # raw is -1 here, like an outline cell
    result = _small_export(tmp_path, rgba, DUP_PALETTE, grid)
    read = to_sidecar_grid(parse_aseprite(result.aseprite_path.read_bytes()))
    assert read[0][2] == 1


@pytest.mark.parametrize("palette_name", list_palettes())
def test_pixelate_output_always_encodable(palette_name):
    import pixelgen.export as export_mod  # function-local on purpose (AC6)

    _aseprite_indices = getattr(export_mod, "_aseprite_indices", None)
    assert _aseprite_indices is not None, "pixelgen.export._aseprite_indices is missing"

    palette = load_palette(palette_name)
    n = len(palette)
    for dither in ("none", "floyd-steinberg", "bayer"):
        for outline in (True, False):
            for background in ("transparent", "solid"):
                for side in (16, 48):
                    spec = ArtSpec(
                        subject="x",
                        width=side,
                        height=side,
                        palette=palette_name,
                        dither=dither,
                        outline=outline,
                        background=background,
                    )
                    g = np.asarray(_grid(1, side=side, colours=min(n, 16)))
                    img = pixelate(grid_to_rgba(g, palette), spec, palette)
                    idx = np.asarray(_aseprite_indices(img, g, palette))
                    alpha = img[..., 3]
                    assert set(np.unique(alpha)) <= {0, 255}
                    assert (idx[alpha == 0] == -1).all()
                    opaque = alpha == 255
                    assert ((idx[opaque] >= 0) & (idx[opaque] < n)).all()
                    assert (palette[idx[opaque]] == img[opaque][:, :3]).all()
                    raw_ok = opaque & (g >= 0) & (g < n)
                    raw_ok[raw_ok] = (palette[g[raw_ok]] == img[raw_ok][:, :3]).all(axis=1)
                    assert (idx[raw_ok] == g[raw_ok]).all()
