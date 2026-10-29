"""Tests for pixelgen.palettes (PXG-03).

Sources (fetched 2026-10-28 with ``curl -L https://lospec.com/palette-list/<slug>.hex``;
the sha256 is of the raw downloaded file):

- pico8:      https://lospec.com/palette-list/pico-8.hex
              5814ea274522a65f52dc8f923e55db19dba57f14aed6d9d5e602e9ea4c32b3eb
- gameboy:    https://lospec.com/palette-list/nintendo-gameboy-bgb.hex
              07736f333d7cd161246239cb8a0430b6c7e8dd378099c12454f5fbfe5e9ee47e
- nes:        https://lospec.com/palette-list/nintendo-entertainment-system.hex
              66791a24cdc15d4a584c3073f97850674e3e9428c30a99de059874bff07d856e
              The shipped file is Lospec's 55 entries minus fcfcfc ($30, the hardware
              duplicate of $20 white f8f8f8), per X-PXG-25: 54 colours. To rebuild it,
              normalize the download (strip, lowercase, drop blanks) and remove fcfcfc.
- endesga32:  https://lospec.com/palette-list/endesga-32.hex
              56243bfe40a4c0ade168af091ee69269ab84d9ea201d023d7d94bdf5f10eae2f
- grayscale4: https://lospec.com/palette-list/2-bit-grayscale.hex
              1093434349c887472b7a0dbc982addc2bb081dd0807967e5bbd3a1068097559d
"""

import hashlib
import importlib.resources
import re

import numpy
import pytest

from pixelgen import palettes
from pixelgen.errors import PaletteError
from pixelgen.palettes import list_palettes, load_palette, parse_hex

EXPECTED_COUNTS = {"pico8": 16, "gameboy": 4, "nes": 54, "endesga32": 32, "grayscale4": 4}
NAMES = ["endesga32", "gameboy", "grayscale4", "nes", "pico8"]

PICO8 = [
    (0x00, 0x00, 0x00),
    (0x1D, 0x2B, 0x53),
    (0x7E, 0x25, 0x53),
    (0x00, 0x87, 0x51),
    (0xAB, 0x52, 0x36),
    (0x5F, 0x57, 0x4F),
    (0xC2, 0xC3, 0xC7),
    (0xFF, 0xF1, 0xE8),
    (0xFF, 0x00, 0x4D),
    (0xFF, 0xA3, 0x00),
    (0xFF, 0xEC, 0x27),
    (0x00, 0xE4, 0x36),
    (0x29, 0xAD, 0xFF),
    (0x83, 0x76, 0x9C),
    (0xFF, 0x77, 0xA8),
    (0xFF, 0xCC, 0xAA),
]
GAMEBOY = [(0x08, 0x18, 0x20), (0x34, 0x68, 0x56), (0x88, 0xC0, 0x70), (0xE0, 0xF8, 0xD0)]
VALUE_SHA256 = {
    "nes": "b35a7ad6eff68dd1286a421a64107d4e68bb7756589a076dae8235a038ca2517",
    "endesga32": "6e56dbf66a8e92789eb29a99a0fbfa70b39fdfca2f73273b5b58adeff0394f56",
}


def _digest(arr) -> str:
    text = "\n".join(f"{r:02x}{g:02x}{b:02x}" for r, g, b in arr)
    return hashlib.sha256(text.encode()).hexdigest()


def _luma(row) -> float:
    r, g, b = (float(v) for v in row)
    return 0.299 * r + 0.587 * g + 0.114 * b


def test_list_palettes_returns_the_five_sorted():
    result = list_palettes()
    assert isinstance(result, list)
    assert result == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_shipped_palette_shape_and_count(name):
    arr = load_palette(name)
    assert isinstance(arr, numpy.ndarray)
    assert arr.dtype == numpy.uint8
    assert arr.ndim == 2
    assert arr.shape[1] == 3
    assert arr.shape[0] == EXPECTED_COUNTS[name]


def test_pico8_values_are_canonical():
    assert load_palette("pico8").tolist() == [list(c) for c in PICO8]


def test_gameboy_values_are_pinned():
    assert load_palette("gameboy").tolist() == [list(c) for c in GAMEBOY]


def test_grayscale4_luminance_strictly_increases():
    arr = load_palette("grayscale4")
    assert all(r == g == b for r, g, b in arr.tolist())
    lumas = [_luma(row) for row in arr]
    assert all(a < b for a, b in zip(lumas, lumas[1:], strict=False))


@pytest.mark.parametrize("name", sorted(VALUE_SHA256))
def test_value_digest(name):
    arr = load_palette(name)
    assert _digest(arr.tolist()) == VALUE_SHA256[name]
    if name == "nes":
        assert [0xFC, 0xFC, 0xFC] not in arr.tolist()


def test_parse_tolerates_case_blanks_comments_crlf_bom():
    text = "﻿; c\r\n\r\n  FF0000 \r\n00ff00\r\n;x\r\n0000Ff\r\n"
    arr = parse_hex(text)
    assert arr.dtype == numpy.uint8
    assert arr.tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


@pytest.mark.parametrize("bad", ["12345", "1234567", "gg0000", "#ff0000", "0xff0000", "ff 00 00"])
def test_parse_rejects_malformed_line(bad):
    with pytest.raises(PaletteError) as exc:
        parse_hex(f"000000\n{bad}\n")
    assert "line 2" in str(exc.value)
    assert "expected RRGGBB" in str(exc.value)


@pytest.mark.parametrize("text", ["", "\n\n", "; only a comment\n"])
def test_parse_rejects_empty(text):
    with pytest.raises(PaletteError, match="no colours"):
        parse_hex(text)


def test_load_rejects_non_utf8(monkeypatch, tmp_path):
    (tmp_path / "bad.hex").write_bytes(b"\xff\xfe\x00")
    monkeypatch.setattr(palettes, "_palette_dir", lambda: tmp_path)
    with pytest.raises(PaletteError, match="not valid UTF-8"):
        load_palette("bad")


def test_parse_rejects_duplicates_case_insensitively():
    with pytest.raises(PaletteError) as exc:
        parse_hex("ff0000\n00ff00\nFF0000\n")
    msg = str(exc.value)
    assert "duplicate colour ff0000" in msg
    assert "line 3" in msg
    assert "line 1" in msg


def test_parse_caps_at_256_colours():
    rows = [f"{i:06x}" for i in range(257)]
    assert parse_hex("\n".join(rows[:256])).shape == (256, 3)
    with pytest.raises(PaletteError, match="maximum of 256"):
        parse_hex("\n".join(rows))


@pytest.mark.parametrize("name", ["nope", "../pico8", "pico8.hex", "PICO8", ""])
def test_unknown_name_lists_available(name):
    with pytest.raises(PaletteError) as exc:
        load_palette(name)
    msg = str(exc.value)
    assert "\n" not in msg
    assert repr(name) in msg
    assert "available: endesga32, gameboy, grayscale4, nes, pico8" in msg


def test_dropped_file_is_discovered_without_code_change(monkeypatch, tmp_path):
    (tmp_path / "zz.hex").write_text("123456\n")
    (tmp_path / "aa.hex").write_text("abcdef\n")
    (tmp_path / "notes.txt").write_text("hello\n")
    monkeypatch.setattr(palettes, "_palette_dir", lambda: tmp_path)
    assert list_palettes() == ["aa", "zz"]
    assert load_palette("zz").tolist() == [[0x12, 0x34, 0x56]]


@pytest.mark.parametrize("name", NAMES)
def test_shipped_file_is_plain_lowercase_rrggbb(name):
    path = importlib.resources.files("pixelgen") / "palettes" / f"{name}.hex"
    data = path.read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"\n")
    lines = data.decode("utf-8").split("\n")[:-1]
    assert lines
    assert all(re.fullmatch(r"[0-9a-f]{6}", line) for line in lines)
