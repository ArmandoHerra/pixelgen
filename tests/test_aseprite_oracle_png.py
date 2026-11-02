"""PXG-34: the oracle's `--png` pixel-parity mode.

The stdlib PNG decoder is checked against Pillow (reference reader, tests only), every PNG filter
type, and every refusal; the comparator against the alpha-0 rule, row-major first-difference
and size lines; and the whole mode end to end over real `pixelgen generate/refine --aseprite`
outputs, read by the oracle in a `-I` child process that imports no product code.
"""

import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import aseprite_oracle as ao
import numpy as np
import pytest
from PIL import Image
from test_aseprite_oracle import FIXTURES, INDEXED, MADE, ORACLE_PATH, ROOT, _synth
from test_cli_aseprite import Harness, _grid

INDEXED_PNG = INDEXED.parent / "indexed_01.png"
OPERATOR = FIXTURES / "operator"
OPERATOR_CASES = [
    "endesga32-17x33",
    "gameboy-8x8-border",
    "grayscale4-16x16-solid",
    "nes-128x128-hashed",
    "pico8-16x16-index0",
]


def _decode_png(data):
    """Fetch the new symbol by name so a base without it fails by assertion, not AttributeError."""
    fn = getattr(ao, "decode_png", None)
    assert callable(fn), "aseprite_oracle.decode_png is missing"
    return fn(data)


# --------------------------------------------------------------------------- test-private encoder


def _chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body))


def _filter_row(ftype: int, cur: bytes, prev: bytes, bpp: int) -> bytes:
    out = bytearray(len(cur))
    for i, value in enumerate(cur):
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if ftype == 0:
            pred = 0
        elif ftype == 1:
            pred = a
        elif ftype == 2:
            pred = b
        elif ftype == 3:
            pred = (a + b) // 2
        else:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
        out[i] = (value - pred) & 0xFF
    return bytes(out)


def _png(
    rgba,
    *,
    filter_type=0,
    idat_parts=1,
    ctype=6,
    depth=8,
    interlace=0,
    plte=None,
    raw=None,
    iend=True,
    signature=b"\x89PNG\r\n\x1a\n",
):
    """Encode an (h, w, 4) uint8 array as RGBA (or raw scanline data for refusal cases)."""
    arr = np.asarray(rgba, dtype=np.uint8)
    h, w = arr.shape[:2]
    if raw is None:
        rows, prev = [], bytes(w * 4)
        for y in range(h):
            cur = arr[y].tobytes()
            f = (y % 5) if filter_type == "mixed" else filter_type
            rows.append(bytes([f]) + _filter_row(f, cur, prev, 4))
            prev = cur
        raw = b"".join(rows)
    z = zlib.compress(raw)
    size = -(-len(z) // idat_parts)
    parts = [z[i : i + size] for i in range(0, len(z), size)] or [z]
    out = signature + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, interlace))
    if plte is not None:
        out += _chunk(b"PLTE", plte)
    out += b"".join(_chunk(b"IDAT", part) for part in parts)
    if iend:
        out += _chunk(b"IEND", b"")
    return out


def _source(w, h, seed=34):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 4), dtype=np.uint8)


def _as_rows(arr):
    return [[tuple(int(c) for c in px) for px in row] for row in arr]


# --------------------------------------------------------------------------- AC1

VENDORED_PNGS = sorted(FIXTURES.glob("**/*.png"))


def test_decode_png_matches_pillow_set_is_ten():
    assert len(VENDORED_PNGS) == 10


@pytest.mark.parametrize("path", VENDORED_PNGS, ids=lambda p: str(p.relative_to(FIXTURES)))
def test_decode_png_matches_pillow(path):
    w, h, rows = _decode_png(path.read_bytes())
    ref = np.array(Image.open(path).convert("RGBA"))
    assert (w, h) == (ref.shape[1], ref.shape[0])
    assert rows == _as_rows(ref)


@pytest.mark.parametrize("mode", ["rgb", "rgba", "p-trns"])
def test_decode_png_matches_pillow_synthetic(mode, tmp_path):
    rng = np.random.default_rng(34)
    path = tmp_path / "s.png"
    if mode == "rgb":
        Image.fromarray(rng.integers(0, 256, (23, 37, 3), dtype=np.uint8)).save(path)
    elif mode == "rgba":
        Image.fromarray(rng.integers(0, 256, (23, 37, 4), dtype=np.uint8)).save(path)
    else:
        # more than 16 palette entries so Pillow writes 8-bit indices, not packed 4-bit ones
        img = Image.fromarray(rng.integers(0, 20, (23, 37), dtype=np.uint8), "P")
        img.putpalette([c for i in range(20) for c in (i * 12, 255 - i * 9, i * 7)])
        img.save(path, transparency=bytes([0, 128, 200]))
    w, h, rows = _decode_png(path.read_bytes())
    ref = np.array(Image.open(path).convert("RGBA"))
    assert (w, h) == (ref.shape[1], ref.shape[0])
    assert rows == _as_rows(ref)


# --------------------------------------------------------------------------- AC2


@pytest.mark.parametrize("ftype", [0, 1, 2, 3, 4, "mixed"], ids=str)
def test_decode_png_filter_type(ftype):
    src = _source(9, 7)
    w, h, rows = _decode_png(_png(src, filter_type=ftype, idat_parts=3))
    assert (w, h) == (9, 7)
    assert rows == _as_rows(src)


def test_decode_png_filter_type_paeth_ties():
    # Values 0..3 make pa == pc and pb == pc ties common; the PNG tie order is a, b, c.
    src = np.random.default_rng(5).integers(0, 4, size=(16, 16, 4), dtype=np.uint8)
    w, h, rows = _decode_png(_png(src, filter_type=4))
    assert (w, h) == (16, 16)
    assert rows == _as_rows(src)


# --------------------------------------------------------------------------- AC3


def _refusal_png(case, tmp_path):
    src = _source(4, 3)
    good_raw = b"".join(b"\x00" + src[y].tobytes() for y in range(3))
    if case == "signature":
        return _png(src, signature=b"\x89PNG\r\n\x1a\x0b")
    if case == "crc":
        data = bytearray(_png(src))
        data[-1] ^= 0xFF  # IEND's own CRC; every pixel byte is still intact
        return bytes(data)
    if case == "no-iend":
        return _png(src, iend=False)
    if case == "interlace":
        return _png(src, interlace=1)
    if case == "depth16":
        return _png(src, depth=16)
    if case == "grey":
        return _png(src, ctype=0, raw=good_raw)
    if case == "grey-alpha":
        return _png(src, ctype=4, raw=good_raw)
    if case == "index-past-plte":
        raw = b"".join(b"\x00" + bytes([0, 1, 2, 3]) for _ in range(3))
        return _png(src, ctype=3, plte=b"\x00\x00\x00\xff\xff\xff", raw=raw)
    raise AssertionError(case)


REFUSALS = [
    "signature",
    "crc",
    "no-iend",
    "interlace",
    "depth16",
    "grey",
    "grey-alpha",
    "index-past-plte",
    "missing-file",
]


@pytest.mark.parametrize("case", REFUSALS)
def test_png_refusals(case, tmp_path, capsys):
    path = tmp_path / "bad.png"
    if case != "missing-file":
        path.write_bytes(_refusal_png(case, tmp_path))
    rc = ao.main([str(INDEXED), "--png", str(path)])
    out, err = capsys.readouterr()
    assert rc == 1
    assert out == ""
    assert len(err.splitlines()) == 1
    prefix = "aseprite_oracle: error: " + ("" if case == "missing-file" else "png: ")
    assert err.startswith(prefix)
    assert "Traceback" not in err


# --------------------------------------------------------------------------- AC4

PAIRS = [("third_party/indexed.aseprite", "third_party/indexed_01.png")]
PAIRS += [
    (f"aseprite_made/{c}.aseprite", f"aseprite_made/{c}.png")
    for c in ["black0", "grid128", "nonsquare", "offset_cel"]
]
PAIRS += [
    (f"operator/{c}/ours.aseprite", f"operator/{c}/aseprite-export.png") for c in OPERATOR_CASES
]


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p[0])
def test_png_parity_real_pairs(pair, capsys):
    ase, png = FIXTURES / pair[0], FIXTURES / pair[1]
    rc = ao.main([str(ase), "--png", str(png)])
    out, err = capsys.readouterr()
    w, h, _ = _decode_png(png.read_bytes())
    assert (rc, err) == (0, "")
    assert len(out.splitlines()) == 1
    assert out.startswith("OK ")
    assert out.rstrip("\n").endswith(f"; png {png} matches ({w * h} pixels)")


def test_png_parity_real_pairs_set_is_ten():
    assert len(PAIRS) == 10
    assert MADE.is_dir()


# --------------------------------------------------------------------------- AC5/6/7

RED = (255, 0, 0, 255)
BLACK = (0, 0, 0, 255)
CLEAR = (0, 0, 0, 0)
PALETTE = [RED, BLACK, CLEAR]


def _file(w, h, pixels):
    """A `w`x`h` indexed file; `pixels` maps (x, y) -> palette index (default 0 = red)."""
    idx = bytearray(w * h)
    for (x, y), v in pixels.items():
        idx[y * w + x] = v
    return _synth(w, h, PALETTE, 2, [(0, 0, 0, w, h, bytes(idx))])


def _png_of(w, h, pixels, base=RED):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :] = base
    for (x, y), rgba in pixels.items():
        arr[y, x] = rgba
    return arr


def _run(tmp_path, capsys, ase_bytes, png_bytes):
    a, p = tmp_path / "f.aseprite", tmp_path / "f.png"
    a.write_bytes(ase_bytes)
    p.write_bytes(png_bytes)
    rc = ao.main([str(a), "--png", str(p)])
    out, err = capsys.readouterr()
    return rc, out.splitlines(), err


ALPHA_ARMS = {
    "both-clear-rgb-differs": ({(1, 1): 2}, (17, 34, 51, 0), 0),
    "clear-vs-opaque-black": ({(1, 1): 2}, (0, 0, 0, 255), 1),
    "opaque-black-vs-clear": ({(1, 1): 1}, (0, 0, 0, 0), 1),
    "partial-alpha": ({}, (255, 0, 0, 254), 1),
}


@pytest.mark.parametrize("arm", list(ALPHA_ARMS))
def test_png_alpha_rule(arm, tmp_path, capsys):
    file_px, png_rgba, want_rc = ALPHA_ARMS[arm]
    ase = _file(4, 3, file_px)
    png = _png(_png_of(4, 3, {(1, 1): png_rgba}))
    rc, lines, err = _run(tmp_path, capsys, ase, png)
    assert (rc, err) == (want_rc, "")
    if want_rc == 0:
        assert lines[0].startswith("OK ")
    else:
        assert len(lines) == 1 and lines[0].startswith("MISMATCH png x=1 y=1: ")
    if arm == "clear-vs-opaque-black":
        assert "file (0, 0, 0, 0) != png (0, 0, 0, 255)" in lines[0]


def test_png_first_differing_pixel_row_major(tmp_path, capsys):
    ase = _file(8, 6, {})
    png = _png(_png_of(8, 6, {(5, 0): BLACK, (0, 3): BLACK}))
    rc, lines, err = _run(tmp_path, capsys, ase, png)
    assert (rc, err) == (1, "")
    assert lines == [
        "MISMATCH png x=5 y=0: file (255, 0, 0, 255) != png (0, 0, 0, 255) (2 of 48 pixels differ)"
    ]


def test_png_first_differing_single_pixel(tmp_path, capsys):
    ase = _file(8, 6, {})
    png = _png(_png_of(8, 6, {(3, 2): BLACK}))
    rc, lines, _ = _run(tmp_path, capsys, ase, png)
    assert rc == 1 and len(lines) == 1
    assert lines[0].startswith("MISMATCH png x=3 y=2: ")
    assert lines[0].endswith("(1 of 48 pixels differ)")


def test_png_size_mismatch_wider(tmp_path, capsys):
    rc, lines, _ = _run(tmp_path, capsys, _file(8, 6, {}), _png(_png_of(9, 6, {})))
    assert rc == 1
    assert lines == ["MISMATCH png size: file 8x6 != png 9x6"]


def test_png_size_mismatch_shorter(tmp_path, capsys):
    rc, lines, _ = _run(tmp_path, capsys, _file(8, 6, {}), _png(_png_of(8, 5, {})))
    assert rc == 1
    assert lines == ["MISMATCH png size: file 8x6 != png 8x5"]


# --------------------------------------------------------------------------- AC8

ORACLE_ARMS = {
    "generate-default": ([], None),
    "generate-no-outline": (["--no-outline"], None),
    "generate-solid": (["--background", "solid"], None),
    "generate-48": (["--size", "48"], _grid(1, side=32)),
    "generate-gameboy": (["--palette", "gameboy"], _grid(1, colours=4)),
    "refine-default": ([], None),
}


@pytest.fixture
def h(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


API_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)


def _oracle_child(ase: Path, png: Path):
    # The Harness sets a dummy API key; the stdlib-only `-I` oracle child needs none, and the
    # test guard refuses an unguarded child that inherits one.
    env = {k: v for k, v in os.environ.items() if k not in API_KEY_VARS}
    return subprocess.run(
        [sys.executable, "-I", str(ORACLE_PATH), str(ase), "--png", str(png)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.parametrize("arm", list(ORACLE_ARMS))
def test_cli_outputs_png_parity(arm, h):
    extra, grid = ORACLE_ARMS[arm]
    script = ({}, {"grid": grid if grid is not None else _grid(1)})
    if arm == "refine-default":
        first = h.generate(["-o", "out/x"])
        assert first.exit_code == 0, (first.stdout, first.stderr)
        result = h.refine(["--from", str(h.out / "x.json"), "--aseprite"])
        assert result.exit_code == 0, (result.stdout, result.stderr)
        assert h.llm.call_count == 1
        stem = "x-v2"
    else:
        result = h.generate([*extra, "-o", "out/x", "--aseprite"], *script)
        assert result.exit_code == 0, (result.stdout, result.stderr)
        assert h.llm.call_count == 2
        stem = "x"
    ase, png = h.out / f"{stem}.aseprite", h.out / f"{stem}.png"
    run = _oracle_child(ase, png)
    assert run.returncode == 0, (run.stdout, run.stderr)
    assert run.stdout.startswith("OK ") and "; png " in run.stdout

    parsed = ao.parse_aseprite(ase.read_bytes())
    w, hgt = parsed.width, parsed.height
    preview = _oracle_child(ase, h.out / f"{stem}@8x.png")
    assert preview.returncode == 1
    assert preview.stdout.splitlines() == [
        f"MISMATCH png size: file {w}x{hgt} != png {8 * w}x{8 * hgt}"
    ]


# --------------------------------------------------------------------------- AC9


def test_oracle_png_path_loads_no_numpy_pil_or_pixelgen():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import aseprite_oracle as ao; "
        "rc = ao.main([sys.argv[2], '--png', sys.argv[3]]); assert rc == 0, rc; "
        "print([m for m in ('numpy', 'PIL', 'pixelgen') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT / "tests"), str(INDEXED), str(INDEXED_PNG)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "[]"


def test_oracle_png_public_names():
    assert callable(getattr(ao, "decode_png", None))
    assert callable(getattr(ao, "compare_with_png", None))


def test_oracle_without_png_output_unchanged(capsys):
    rc = ao.main([str(INDEXED)])
    out, err = capsys.readouterr()
    assert (rc, err) == (0, "")
    assert out == (
        f"OK {INDEXED}: 64x64 depth 8, 4 frame(s), transparent_index 1, 73 palette entries\n"
    )
