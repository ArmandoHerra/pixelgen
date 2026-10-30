"""PXG-24: the stdlib `.aseprite` oracle, proven against bytes Aseprite itself wrote.

Anchors: three files vendored from asefile (MIT) and four cases made by the operator's local
Aseprite from `generate.lua`. Self-authored synthetic bytes (`_synth`) are used only where no real
file can carry the case.
"""

import ast
import dataclasses
import hashlib
import json
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import aseprite_oracle as ao
import pytest
from aseprite_local import require_aseprite
from PIL import Image
from regen_aseprite_made import CASES, aseprite_version, regenerate

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "aseprite"
THIRD_PARTY = FIXTURES / "third_party"
MADE = FIXTURES / "aseprite_made"
ORACLE_PATH = ROOT / "tests" / "aseprite_oracle.py"
INDEXED = THIRD_PARTY / "indexed.aseprite"
PALETTE = THIRD_PARTY / "palette.aseprite"
ROW = re.compile(r"^\| `([^`]+)` \| ([0-9a-f]{64}) \|$", re.MULTILINE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> dict[str, str]:
    return dict(ROW.findall(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- byte helpers


def _chunk(ctype: int, payload: bytes) -> bytes:
    return struct.pack("<IH", 6 + len(payload), ctype) + payload


def _synth(
    width,
    height,
    palette_rgba,
    transparent_index,
    cels,
    *,
    layers=1,
    cel_type=2,
    trailing=b"",
    extra_chunks=(),
):
    """Assemble a one-frame indexed file. `cels` is [(layer, x, y, w, h, pixel bytes)]."""
    n = len(palette_rgba)
    pal = struct.pack("<III8x", n, 0, n - 1) + b"".join(
        struct.pack("<HBBBB", 0, *rgba) for rgba in palette_rgba
    )
    chunks = [_chunk(0x2019, pal), *extra_chunks]
    for i in range(layers):
        name = f"L{i}".encode()
        chunks.append(
            _chunk(
                0x2004,
                struct.pack("<HHHHHHB3x", 3, 0, 0, 0, 0, 0, 255)
                + struct.pack("<H", len(name))
                + name,
            )
        )
    for layer, x, y, w, h, pixels in cels:
        data = zlib.compress(pixels) if cel_type == 2 else pixels
        body = struct.pack("<HhhBHh5x", layer, x, y, 255, cel_type, 0)
        chunks.append(_chunk(0x2005, body + struct.pack("<HH", w, h) + data + trailing))
    frame_size = 16 + sum(len(c) for c in chunks)
    frame = struct.pack("<IHHH2xI", frame_size, 0xF1FA, len(chunks), 100, len(chunks)) + b"".join(
        chunks
    )
    head = bytearray(128)
    struct.pack_into("<IHHHHHIH", head, 0, 128 + len(frame), 0xA5E0, 1, width, height, 8, 1, 100)
    struct.pack_into("<B", head, 28, transparent_index)
    struct.pack_into("<HBB", head, 32, n, 1, 1)
    return bytes(head) + frame


def _walk(data: bytes) -> list[dict]:
    """Locate every chunk by walking the real bytes (never hard-coded offsets)."""
    out = []
    (nframes,) = struct.unpack_from("<H", data, 6)
    off = 128
    for fi in range(nframes):
        (fsize,) = struct.unpack_from("<I", data, off)
        pos = off + 16
        while pos < off + fsize:
            csize, ctype = struct.unpack_from("<IH", data, pos)
            out.append({"frame": fi, "frame_off": off, "off": pos, "size": csize, "type": ctype})
            pos += csize
        off += fsize
    return out


def _first(data: bytes, ctype: int, frame: int = 0) -> dict:
    return next(c for c in _walk(data) if c["type"] == ctype and c["frame"] == frame)


def _patch(data: bytes, fmt: str, offset: int, *values) -> bytes:
    buf = bytearray(data)
    struct.pack_into(fmt, buf, offset, *values)
    return bytes(buf)


def _fix_size(data: bytes) -> bytes:
    return _patch(data, "<I", 0, len(data))


def _compare_rgba(ours, ref_rgba: bytes, w: int, h: int) -> list:
    """Research 6.3 rule: where the reference alpha is 0 ours must be 0 too, else exact RGBA."""
    bad = []
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 4
            ref = tuple(ref_rgba[i : i + 4])
            got = ours[y][x]
            same = got[3] == 0 if ref[3] == 0 else got == ref
            if not same:
                bad.append((x, y, got, ref))
    return bad


def _png_rgba(path: Path) -> bytes:
    return Image.open(path).convert("RGBA").tobytes()


# --------------------------------------------------------------------------- AC1, AC2


def test_third_party_fixtures_match_source_sha256():
    expected = {
        "indexed.aseprite": "a068ac5fe641df53601c14c14850275bb94ca4407c2f5032cc688e917c9a109f",
        "indexed_01.png": "bf0b707888b34addfb69b07e32760fef3303b0dd5f0bd29d074c9c189dc1a620",
        "palette.aseprite": "d8b904e12a76241f4470de8e639b86961ebfdaa9f1c06dfcc2a4f3e02c1a946e",
        "LICENSE": "f33b372b56866e612afc83c26781ff9102282b37ebf628ecec33d8e72dbbc1ed",
    }
    assert {p.name for p in THIRD_PARTY.iterdir()} == {*expected, "SOURCE.md"}
    source = (THIRD_PARTY / "SOURCE.md").read_text(encoding="utf-8")
    assert "https://github.com/alpine-alpaca/asefile" in source
    assert "2274c354cea6764f85597252a0d2228e64709348" in source
    rows = _rows(THIRD_PARTY / "SOURCE.md")
    assert set(rows) == set(expected)
    for name, digest in rows.items():
        assert _sha(THIRD_PARTY / name) == digest == expected[name], name


def test_third_party_licence_is_upstream_mit():
    licence = THIRD_PARTY / "LICENSE"
    assert _sha(licence) == "f33b372b56866e612afc83c26781ff9102282b37ebf628ecec33d8e72dbbc1ed"
    text = licence.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "The MIT License (MIT)"
    assert "Copyright (c) 2020 Alpine Alpaca Games" in text
    assert "MIT" in (THIRD_PARTY / "SOURCE.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- AC3


def test_oracle_public_interface():
    import inspect

    assert issubclass(ao.AseFormatError, Exception)
    for name in (
        "parse_aseprite",
        "composite_indices",
        "to_sidecar_grid",
        "composite_rgba",
        "compare_with_sidecar",
        "main",
    ):
        assert callable(getattr(ao, name)), name
    assert inspect.signature(ao.composite_indices).parameters["frame"].default == 0
    assert dataclasses.is_dataclass(ao.AseFile)
    fields = {f.name for f in dataclasses.fields(ao.AseFile)}
    assert {
        "width",
        "height",
        "color_depth",
        "transparent_index",
        "palette",
        "frames",
        "layers",
        "cels",
    } <= fields
    assert "chunk_types" in {f.name for f in dataclasses.fields(ao.AseFrame)}


# --------------------------------------------------------------------------- AC4 to AC7


def test_l3a_indexed_frame0_matches_reference_png():
    ase = ao.parse_aseprite(INDEXED.read_bytes())
    ours = ao.composite_rgba(ase, 0)
    bad = _compare_rgba(ours, _png_rgba(THIRD_PARTY / "indexed_01.png"), 64, 64)
    assert bad == [], bad[:5]
    visible = sum(v != ase.transparent_index for row in ao.composite_indices(ase, 0) for v in row)
    assert visible == 693


def test_indexed_fixture_structure():
    ase = ao.parse_aseprite(INDEXED.read_bytes())
    assert (ase.width, ase.height, ase.color_depth) == (64, 64, 8)
    assert ase.transparent_index == 1
    assert len(ase.frames) == 4
    assert len(ase.palette) == 73
    assert ase.palette[0] == (0, 0, 0, 0)
    assert ase.palette[1] == (46, 34, 47, 255)
    assert ase.palette[72] == (0, 0, 0, 83)
    assert ase.old_palette is not None
    assert ase.frames[0].chunk_types == [
        0x2007,
        0x2019,
        0x0004,
        0x2004,
        0x2004,
        0x2004,
        0x2005,
        0x2005,
        0x2005,
    ]
    for frame in ase.frames[1:]:
        assert frame.chunk_types == [0x2005] * 3
    assert [layer.name for layer in ase.layers] == ["Layer 1", "Layer 2", "Layer 3"]
    assert all(layer.layer_type == 0 and layer.flags == 3 for layer in ase.layers)
    frame0 = [(c.layer, c.x, c.y, c.width, c.height, c.cel_type) for c in ase.cels if c.frame == 0]
    assert frame0 == [(0, 3, 21, 31, 28, 2), (1, 20, 8, 31, 39, 2), (2, 7, 5, 45, 46, 2)]


def test_linked_cel_resolves_to_source_frame():
    ase = ao.parse_aseprite(INDEXED.read_bytes())

    def cel(frame, layer):
        return next(c for c in ase.cels if c.frame == frame and c.layer == layer)

    for layer in (1, 2):
        linked, source = cel(3, layer), cel(1, layer)
        assert linked.cel_type == 1 and linked.linked_frame == 1
        assert (linked.x, linked.y, linked.width, linked.height, linked.pixels) == (
            source.x,
            source.y,
            source.width,
            source.height,
            source.pixels,
        )
    grid = ao.composite_indices(ase, 3)
    assert len(grid) == 64 and all(len(row) == 64 for row in grid)


def test_palette_fixture_matches_asefile_assertions():
    ase = ao.parse_aseprite(PALETTE.read_bytes())
    assert len(ase.palette) == 85
    assert ase.palette[0] == (46, 34, 47, 255)
    assert ase.palette[71] == (0, 0, 0, 83)
    assert ase.transparent_index == 0
    assert (ase.width, ase.height, len(ase.frames)) == (16, 16, 1)
    assert [(c.layer, c.x, c.y, c.width, c.height, c.cel_type) for c in ase.cels] == [
        (0, 3, 2, 12, 10, 2)
    ]
    grid = ao.composite_indices(ase, 0)
    outside = [
        grid[y][x] for y in range(16) for x in range(16) if not (3 <= x < 15 and 2 <= y < 12)
    ]
    assert len(outside) == 16 * 16 - 12 * 10
    assert set(outside) == {0}


@pytest.mark.parametrize("path", [PALETTE, INDEXED], ids=["palette", "indexed"])
def test_transparent_index_is_alpha0_even_when_palette_entry_opaque(path):
    ase = ao.parse_aseprite(path.read_bytes())
    t = ase.transparent_index
    assert ase.palette[t][3] == 255  # the palette entry itself is opaque
    indices = ao.composite_indices(ase, 0)
    rgba = ao.composite_rgba(ase, 0)
    cells = [(y, x) for y, row in enumerate(indices) for x, v in enumerate(row) if v == t]
    assert cells
    assert all(rgba[y][x] == (0, 0, 0, 0) for y, x in cells)


# --------------------------------------------------------------------------- AC8


GREY = [(i * 10, i * 10, i * 10, 255) for i in range(6)]


def test_raw_and_zlib_cels_composite_identically():
    pixels = bytes([0, 1, 2, 3, 4, 0, 1, 2, 3])
    cels = [(0, 1, 1, 3, 3, pixels)]
    raw = ao.parse_aseprite(_synth(5, 5, GREY, 5, cels, cel_type=0))
    comp = ao.parse_aseprite(_synth(5, 5, GREY, 5, cels, cel_type=2))
    assert ao.composite_indices(raw) == ao.composite_indices(comp)
    assert ao.composite_indices(raw)[1][1:4] == [0, 1, 2]
    assert ao.composite_indices(raw)[0] == [5] * 5


def test_offset_cels_are_clipped_to_canvas():
    cels = [
        (0, -2, -1, 3, 2, bytes([1, 1, 1, 2, 2, 2])),
        (1, 7, 5, 3, 2, bytes([3, 3, 3, 4, 4, 4])),
    ]
    ase = ao.parse_aseprite(_synth(8, 6, GREY, 5, cels, layers=2))
    expected = [[5] * 8 for _ in range(6)]
    expected[0][0] = 2
    expected[5][7] = 3
    assert ao.composite_indices(ase) == expected


def test_transparent_cel_pixels_do_not_overwrite():
    cels = [(0, 0, 0, 4, 3, bytes([1] * 12)), (1, 0, 0, 2, 2, bytes([5, 2, 3, 5]))]
    ase = ao.parse_aseprite(_synth(4, 3, GREY, 5, cels, layers=2))
    grid = ao.composite_indices(ase)
    assert grid[0][:2] == [1, 2]
    assert grid[1][:2] == [3, 1]


# --------------------------------------------------------------------------- AC9


def _mut_file_size(d):
    return _patch(d, "<I", 0, len(d) + 1)


def _mut_header_magic(d):
    return _patch(d, "<H", 4, 0)


def _mut_frame_magic(d):
    return _patch(d, "<H", _first(d, 0x2007)["frame_off"] + 4, 0)


def _mut_header_frame_count(d):
    return _patch(d, "<H", 6, 5)


def _mut_old_zero(d):
    return _patch(d, "<H", _first(d, 0x2007)["frame_off"] + 6, 0)


def _mut_old_off_by_one(d):
    return _patch(d, "<H", _first(d, 0x2007)["frame_off"] + 6, 8)


def _mut_chunk_under_6(d):
    return _patch(d, "<I", _first(d, 0x2007)["off"], 5)


def _mut_chunk_overruns(d):
    return _patch(d, "<I", _first(d, 0x2007)["off"], 0x7FFFFFFF)


def _mut_trailing(d):
    return _fix_size(d + b"\0")


def _mut_cel_zlib(d):
    c = _first(d, 0x2005)
    buf = bytearray(d)
    buf[c["off"] + 6 + 20 + (c["size"] - 26) // 2] ^= 0xFF
    return bytes(buf)


def _mut_cel_type(d):
    return _patch(d, "<H", _first(d, 0x2005)["off"] + 6 + 7, 3)


def _mut_layer_type(d):
    return _patch(d, "<H", _first(d, 0x2004)["off"] + 6 + 2, 1)


def _mut_background(d):
    off = _first(d, 0x2004)["off"] + 6
    (flags,) = struct.unpack_from("<H", d, off)
    return _patch(d, "<H", off, flags | 8)


def _mut_no_palette(d):
    return _patch(d, "<H", _first(d, 0x2019)["off"] + 4, 0x7FFF)


def _mut_depth32(d):
    return _patch(d, "<H", 12, 32)


MUTATIONS = {
    "file-size": _mut_file_size,
    "header-magic": _mut_header_magic,
    "frame-magic": _mut_frame_magic,
    "header-frame-count": _mut_header_frame_count,
    "old-new-count-mismatch": _mut_old_zero,
    "old-count-off-by-one": _mut_old_off_by_one,
    "chunk-size-under-6": _mut_chunk_under_6,
    "chunk-overruns-frame": _mut_chunk_overruns,
    "trailing-bytes": _mut_trailing,
    "truncated-100": lambda d: d[:100],
    "truncated-1000": lambda d: d[:1000],
    "cel-zlib-corrupt": _mut_cel_zlib,
    "unsupported-cel-type": _mut_cel_type,
    "unsupported-layer-type": _mut_layer_type,
    "background-layer": _mut_background,
    "no-palette": _mut_no_palette,
    "depth-32": _mut_depth32,
}


@pytest.mark.parametrize("case", sorted(MUTATIONS))
def test_structural_defect_raises_ase_format_error(case):
    mutated = MUTATIONS[case](INDEXED.read_bytes())
    with pytest.raises(ao.AseFormatError):
        ao.parse_aseprite(mutated)


def test_structural_defect_cel_pixel_length():
    short = [(0, 0, 0, 2, 2, bytes([1, 1, 1]))]
    with pytest.raises(ao.AseFormatError):
        ao.parse_aseprite(_synth(4, 4, GREY, 5, short))


def test_structural_defect_cel_trailing_data():
    cels = [(0, 0, 0, 2, 2, bytes([1, 1, 1, 1]))]
    with pytest.raises(ao.AseFormatError):
        ao.parse_aseprite(_synth(4, 4, GREY, 5, cels, trailing=b"\x00\x01"))


def test_structural_defect_raw_cel_overruns_chunk():
    cels = [(0, 0, 0, 3, 3, bytes([1] * 8))]
    with pytest.raises(ao.AseFormatError):
        ao.parse_aseprite(_synth(4, 4, GREY, 5, cels, cel_type=0))


def test_fixture_frame_counts_accepted():
    for path, counts in ((INDEXED, [9, 3, 3, 3]), (PALETTE, [5])):
        ase = ao.parse_aseprite(path.read_bytes())
        got = [(f.old_chunk_count, f.new_chunk_count, len(f.chunk_types)) for f in ase.frames]
        assert got == [(n, n, n) for n in counts]


@pytest.mark.parametrize("frame", [4, -1])
def test_composite_rejects_frame_out_of_range(frame):
    ase = ao.parse_aseprite(INDEXED.read_bytes())
    with pytest.raises(ao.AseFormatError):
        ao.composite_indices(ase, frame)


def test_composite_rejects_non_indexed_depth():
    ase = ao.parse_aseprite(INDEXED.read_bytes())
    rgba = dataclasses.replace(ase, color_depth=32)
    with pytest.raises(ao.AseFormatError):
        ao.composite_indices(rgba, 0)


def test_old_palette_count_zero_means_256():
    packet = struct.pack("<H", 1) + bytes([0, 0]) + bytes(range(256)) * 3
    ase = ao.parse_aseprite(
        _synth(2, 2, GREY, 5, [(0, 0, 0, 2, 2, bytes(4))], extra_chunks=[_chunk(0x0004, packet)])
    )
    assert ase.old_palette is not None and len(ase.old_palette) == 256


# --------------------------------------------------------------------------- AC10, AC11


def test_unknown_chunk_types_are_walked_by_size():
    data = INDEXED.read_bytes()
    base = ao.composite_indices(ao.parse_aseprite(data), 0)
    mutated = _patch(data, "<H", _first(data, 0x2007)["off"] + 4, 0x7FF0)
    ase = ao.parse_aseprite(mutated)
    assert ase.frames[0].chunk_types[0] == 0x7FF0
    assert ao.composite_indices(ase, 0) == base


def test_to_sidecar_grid_maps_only_transparent_index():
    palette = [(0, 0, 0, 255), (9, 9, 9, 255), (5, 5, 5, 255), (0, 0, 0, 0)]
    ase = ao.parse_aseprite(_synth(3, 2, palette, 3, [(0, 0, 0, 3, 2, bytes([0, 2, 3, 3, 0, 2]))]))
    assert ao.to_sidecar_grid(ase) == [[0, 2, -1], [-1, 0, 2]]
    real = ao.parse_aseprite(INDEXED.read_bytes())
    flat = [v for row in ao.to_sidecar_grid(real) for v in row]
    assert flat.count(-1) == 64 * 64 - 693 == 3403
    assert min(flat) == -1


# --------------------------------------------------------------------------- AC12


def test_oracle_imports_stdlib_only():
    tree = ast.parse(ORACLE_PATH.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__":
            pytest.fail("oracle calls __import__")
    assert modules
    assert modules <= set(sys.stdlib_module_names), sorted(modules - set(sys.stdlib_module_names))
    assert not modules & {"pixelgen", "numpy", "PIL", "importlib"}


def test_oracle_runtime_loads_no_numpy_pil_or_pixelgen():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import aseprite_oracle as ao; "
        "a = ao.parse_aseprite(open(sys.argv[2], 'rb').read()); ao.composite_indices(a, 0); "
        "print([m for m in ('numpy', 'PIL', 'pixelgen') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT / "tests"), str(INDEXED)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# --------------------------------------------------------------------------- AC13


COLORS = ["000000", "ff0000", "00ff00", "0000ff"]


def _shaped(pixels, w, h, *, transparent=4):
    palette = [(int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255) for c in COLORS]
    palette.append((0, 0, 0, 0))
    return _synth(w, h, palette, transparent, [(0, 0, 0, w, h, bytes(pixels))])


def _grid_of(pixels, w, h, t=4):
    return [
        [-1 if pixels[y * w + x] == t else pixels[y * w + x] for x in range(w)] for y in range(h)
    ]


def _sidecar(w, h, pixels, colors=COLORS):
    return {
        "spec": {"width": w, "height": h},
        "palette": {"colors": list(colors)},
        "grid": _grid_of(pixels, w, h),
    }


PIX = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]  # 4 x 3
W, H = 4, 3


def _run_main(capsys, *argv):
    rc = ao.main([str(a) for a in argv])
    out = capsys.readouterr()
    return rc, out.out, out.err


def _write_pair(tmp_path, data, sidecar):
    f, s = tmp_path / "f.aseprite", tmp_path / "s.json"
    f.write_bytes(data)
    s.write_text(json.dumps(sidecar), encoding="utf-8")
    return f, s


def test_cli_ok_on_real_fixture(capsys):
    rc, out, err = _run_main(capsys, INDEXED)
    assert rc == 0 and err == ""
    lines = out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("OK ") and "64x64" in lines[0]


def test_cli_sidecar_match_returns_zero(tmp_path, capsys):
    f, s = _write_pair(tmp_path, _shaped(PIX, W, H), _sidecar(W, H, PIX))
    rc, out, err = _run_main(capsys, f, "--sidecar", s)
    assert (rc, err) == (0, "") and out.startswith("OK ")


def _case_size(tmp_path):
    side = _sidecar(W, H, PIX)
    side["spec"]["width"] = 9
    return _shaped(PIX, W, H), side


def _case_palette(tmp_path):
    return _shaped(PIX, W, H), _sidecar(W, H, PIX, ["000000", "ff0001", "00ff00", "0000ff"])


def _case_transparent(tmp_path):
    pixels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    side = {
        "spec": {"width": W, "height": H},
        "palette": {"colors": COLORS},
        "grid": _grid_of(pixels, W, H, t=99),
    }
    return _shaped(pixels, W, H, transparent=3), side


def _case_grid(tmp_path):
    side = _sidecar(W, H, PIX)
    side["grid"][0][0] = 3
    side["grid"][2][3] = 0
    return _shaped(PIX, W, H), side


@pytest.mark.parametrize(
    ("check", "builder"),
    [
        ("size", _case_size),
        ("palette", _case_palette),
        ("transparent_index", _case_transparent),
        ("grid", _case_grid),
    ],
)
def test_cli_reports_only_the_failing_check(check, builder, tmp_path, capsys):
    data, side = builder(tmp_path)
    f, s = _write_pair(tmp_path, data, side)
    rc, out, err = _run_main(capsys, f, "--sidecar", s)
    lines = out.splitlines()
    assert rc == 1 and err == ""
    assert lines and all(line.startswith(f"MISMATCH {check}") for line in lines), lines
    if check == "grid":
        assert sum(line.startswith("MISMATCH grid x=") for line in lines) == 2
        assert lines[-1] == f"MISMATCH grid: 2 of {W * H} cells differ"


def test_cli_reports_grid_shape_mismatch(tmp_path, capsys):
    side = _sidecar(W, H, PIX)
    side["grid"] = side["grid"][:2]
    f, s = _write_pair(tmp_path, _shaped(PIX, W, H), side)
    rc, out, _ = _run_main(capsys, f, "--sidecar", s)
    assert rc == 1 and out.splitlines() == [f"MISMATCH grid: shape {H}x{W} != sidecar 2x{W}"]


def test_cli_failures_print_one_error_line_and_no_traceback(tmp_path, capsys):
    truncated = tmp_path / "t.aseprite"
    truncated.write_bytes(INDEXED.read_bytes()[:100])
    notjson = tmp_path / "bad.json"
    notjson.write_text("{nope", encoding="utf-8")
    nokey = tmp_path / "nokey.json"
    nokey.write_text("{}", encoding="utf-8")
    for argv in (
        [truncated],
        [tmp_path / "missing.aseprite"],
        [INDEXED, "--sidecar", notjson],
        [INDEXED, "--sidecar", nokey],
    ):
        rc, out, err = _run_main(capsys, *argv)
        assert rc == 1 and out == ""
        assert len(err.splitlines()) == 1 and err.startswith("aseprite_oracle: error: "), err
        assert "Traceback" not in err


def test_cli_runs_as_standalone_script(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(ORACLE_PATH), str(INDEXED)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK ") and "64x64" in result.stdout
    bad = subprocess.run(
        [sys.executable, "-I", str(ORACLE_PATH), str(tmp_path / "nope")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1 and bad.stderr.startswith("aseprite_oracle: error: ")
    assert "Traceback" not in bad.stderr


# --------------------------------------------------------------------------- AC17 to AC19, AC21


def _grid128_colors():
    return [(0, 0, 0)] + [
        ((16 * i) % 256, (37 * i) % 256, (255 - 11 * i) % 256) for i in range(1, 16)
    ]


FOUR = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
MADE_CASES = {
    "black0": (8, 8, FOUR, lambda x, y: (x + 2 * y) % 5, (0, 0, 8, 8)),
    "nonsquare": (17, 33, FOUR, lambda x, y: (3 * x + y) % 5, (0, 0, 17, 33)),
    "grid128": (128, 128, _grid128_colors(), lambda x, y: (7 * x + 13 * y) % 17, (0, 0, 128, 128)),
    "offset_cel": (32, 24, FOUR, lambda u, v: (u + v) % 4, (5, 7, 10, 6)),
}


def _formula_grid(case):
    w, h, colors, f, (cx, cy, cw, ch) = MADE_CASES[case]
    n = len(colors)
    grid = [[n] * w for _ in range(h)]
    for v in range(ch):
        for u in range(cw):
            grid[cy + v][cx + u] = f(u, v)
    return grid


def test_made_cases_table_matches_driver_cases():
    assert tuple(MADE_CASES) == CASES


def test_aseprite_made_fixtures_match_provenance_sha256():
    expected = {f"{c}{s}" for c in CASES for s in (".aseprite", ".png")} | {"generate.lua"}
    assert {p.name for p in MADE.iterdir()} == expected | {"PROVENANCE.md"}
    assert len(list(MADE.iterdir())) == 10
    text = (MADE / "PROVENANCE.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^- aseprite --version: (Aseprite \S+)$", text, re.MULTILINE)) == 1
    rows = _rows(MADE / "PROVENANCE.md")
    assert set(rows) == expected
    for name, digest in rows.items():
        assert _sha(MADE / name) == digest, name


@pytest.mark.parametrize("case", CASES)
def test_aseprite_made_structure_matches_generator(case):
    w, h, colors, _, cel = MADE_CASES[case]
    n = len(colors)
    ase = ao.parse_aseprite((MADE / f"{case}.aseprite").read_bytes())
    assert (ase.width, ase.height, ase.color_depth) == (w, h, 8)
    assert ase.transparent_index == n and len(ase.frames) == 1
    assert ase.frames[0].chunk_types == [0x2007, 0x2019, 0x2004, 0x2005]
    assert ase.old_palette is None
    assert ase.palette[:n] == [(*c, 255) for c in colors]
    assert len(ase.palette) == n + 1 and ase.palette[n][3] == 0
    assert len(ase.layers) == 1
    layer = ase.layers[0]
    assert layer.layer_type == 0 and layer.flags & 1 and not layer.flags & 8
    assert [(c.x, c.y, c.width, c.height) for c in ase.cels if c.frame == 0] == [cel]
    grid = _formula_grid(case)
    assert ao.composite_indices(ase, 0) == grid
    if case == "offset_cel":
        assert sum(v == n for row in grid for v in row) == 768 - 60
    sidecar = {
        "spec": {"width": w, "height": h},
        "palette": {"colors": [bytes(c).hex() for c in colors]},
        "grid": [[-1 if v == n else v for v in row] for row in grid],
    }
    assert ao.compare_with_sidecar(ase, sidecar) == []


@pytest.mark.parametrize("case", CASES)
def test_aseprite_made_composite_matches_aseprite_png(case):
    w, h, colors, _, _ = MADE_CASES[case]
    ase = ao.parse_aseprite((MADE / f"{case}.aseprite").read_bytes())
    ours = ao.composite_rgba(ase, 0)
    ref = _png_rgba(MADE / f"{case}.png")
    bad = _compare_rgba(ours, ref, w, h)
    assert bad == [], bad[:5]
    if case == "black0":
        black = sum(tuple(ref[i : i + 4]) == (0, 0, 0, 255) for i in range(0, len(ref), 4))
        zeros = sum(v == 0 for row in ao.composite_indices(ase, 0) for v in row)
        assert black == zeros == 13


def test_aseprite_made_regenerates_identically(tmp_path):
    exe = require_aseprite()
    out = tmp_path / "out"
    root = tmp_path / "root"
    regenerate(exe, out, root=root)
    committed = (MADE / "PROVENANCE.md").read_text(encoding="utf-8")
    committed_version = re.search(r"^- aseprite --version: (Aseprite \S+)$", committed, re.M)[1]
    version = aseprite_version(exe, root=root)
    who = f"run {version}, committed {committed_version}"
    for case in CASES:
        new = ao.parse_aseprite((out / f"{case}.aseprite").read_bytes())
        old = ao.parse_aseprite((MADE / f"{case}.aseprite").read_bytes())
        assert (new.width, new.height, new.transparent_index) == (
            old.width,
            old.height,
            old.transparent_index,
        ), f"{case} header differs: {who}"
        assert new.palette == old.palette, f"{case} palette differs: {who}"
        assert [(c.x, c.y, c.width, c.height) for c in new.cels if c.frame == 0] == [
            (c.x, c.y, c.width, c.height) for c in old.cels if c.frame == 0
        ], f"{case} cel geometry differs: {who}"
        assert ao.composite_indices(new, 0) == ao.composite_indices(old, 0), (
            f"{case} composite differs: {who}"
        )
        assert _png_rgba(out / f"{case}.png") == _png_rgba(MADE / f"{case}.png"), (
            f"{case} PNG differs: {who}"
        )
    if version == committed_version:
        regenerated = (out / "PROVENANCE.md").read_text(encoding="utf-8")
        assert regenerated == committed, (
            f"PROVENANCE.md differs on the same version {version}: the binary or its output "
            "changed (compare the 'aseprite binary sha256' line and the file rows)"
        )
    else:
        print(f"decoded comparison only: {who}")
    assert (root / "xdg" / "config" / "aseprite").is_dir()
