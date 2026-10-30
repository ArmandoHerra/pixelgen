"""PXG-26: the stdlib `.aseprite` writer, checked byte by byte and through the independent oracle.

Raw fields are read with `struct.unpack_from` at the offsets of the research section 3 byte
contract, written out here and never imported from the writer. Semantic checks go through
`aseprite_oracle`. The AC20 cases also run the operator's real Aseprite, locally only.
"""

import ast
import hashlib
import struct
import sys
import zlib
from pathlib import Path

import aseprite_oracle as ao
import numpy
import pytest
from aseprite_local import require_aseprite, run_aseprite
from PIL import Image

from pixelgen import errors
from pixelgen.aseprite import MAX_ASEPRITE_COLOURS, encode_aseprite
from pixelgen.errors import AsepriteError, PixelgenError
from pixelgen.palettes import list_palettes, load_palette
from pixelgen.pixelate import grid_to_rgba

ROOT = Path(__file__).resolve().parent.parent
WRITER_PATH = ROOT / "src" / "pixelgen" / "aseprite.py"
THIS_FILE = Path(__file__).resolve()

PALETTES = list_palettes()
CHUNK_ORDER = [0x2007, 0x0004, 0x2019, 0x2004, 0x2005]


# ---------------------------------------------------------------------------------- helpers


def grid16() -> numpy.ndarray:
    return numpy.random.default_rng(26).integers(-1, 16, size=(16, 16))


def pico8() -> numpy.ndarray:
    return load_palette("pico8")


def synthetic_255() -> numpy.ndarray:
    return numpy.array([[i, (37 * i) % 256, 255 - i] for i in range(255)], dtype=numpy.uint8)


def walk_chunks(data: bytes) -> list[tuple[int, int, int]]:
    """(offset, size, type) for each chunk of the single frame, from offset 144."""
    chunks = []
    pos = 144
    while pos < len(data):
        size, ctype = struct.unpack_from("<IH", data, pos)
        chunks.append((pos, size, ctype))
        pos += size
    return chunks


def chunk_of(data: bytes, ctype: int) -> bytes:
    found = [data[o : o + s] for o, s, t in walk_chunks(data) if t == ctype]
    assert len(found) == 1
    return found[0]


def canonical_form(data: bytes) -> bytes:
    """Length DWORDs zeroed and the cel's zlib stream replaced by its decompressed bytes."""
    out = bytearray(data[:144])
    out[0:4] = b"\x00" * 4
    out[128:132] = b"\x00" * 4
    for offset, size, ctype in walk_chunks(data):
        chunk = data[offset : offset + size]
        if ctype == 0x2005:
            out += b"\x00" * 4 + chunk[4:26] + zlib.decompress(chunk[26:])
        else:
            out += chunk
    return bytes(out)


def marked_grid(rng: numpy.random.Generator, n: int, height: int, width: int) -> numpy.ndarray:
    """Random cells plus an asymmetric corner marker and a transparent centre."""
    grid = rng.integers(-1, n, size=(height, width))
    grid[0, 0] = 0
    grid[0, width - 1] = n - 1
    grid[height - 1, 0] = n - 1
    grid[height - 1, width - 1] = n - 1
    grid[height // 2, width // 2] = -1
    return grid


def seed_for(palette_index: int, width: int, height: int) -> int:
    return 1_000_000 * palette_index + 1000 * width + height


def assert_roundtrip(grid: numpy.ndarray, palette: numpy.ndarray, *, seed: object = None) -> None:
    n = palette.shape[0]
    data = encode_aseprite(grid, palette)
    try:
        ase = ao.parse_aseprite(data)
        got = ao.to_sidecar_grid(ase)
    except ao.AseFormatError as exc:  # pragma: no cover - failure path
        pytest.fail(f"oracle rejected the bytes (seed {seed}): {exc}")
    assert got == grid.tolist(), f"grid mismatch (seed {seed})"
    assert [c[:3] for c in ase.palette[:n]] == [tuple(int(v) for v in c) for c in palette]
    rgba = numpy.array(ao.composite_rgba(ase), dtype=numpy.uint8)
    want = grid_to_rgba(numpy.asarray(grid, dtype=numpy.int64), palette)
    assert (rgba[..., 3] == want[..., 3]).all(), f"alpha mismatch (seed {seed})"
    opaque = want[..., 3] == 255
    assert (rgba[opaque][:, :3] == want[opaque][:, :3]).all(), f"rgb mismatch (seed {seed})"


# ---------------------------------------------------------------------------------- AC1


def test_aseprite_error_is_a_one_line_pixelgen_error():
    assert issubclass(AsepriteError, PixelgenError)
    message = str(AsepriteError())
    assert message and "\n" not in message
    others = {
        cls.default_message
        for cls in vars(errors).values()
        if isinstance(cls, type) and issubclass(cls, PixelgenError) and cls is not AsepriteError
    }
    assert message not in others


def test_max_colours_is_255():
    assert MAX_ASEPRITE_COLOURS == 255


# ---------------------------------------------------------------------------------- AC2


def test_header_file_size_equals_length():
    data = encode_aseprite(grid16(), pico8())
    assert struct.unpack_from("<I", data, 0)[0] == len(data)


def test_header_magic_frames_size_depth():
    data = encode_aseprite(grid16(), pico8())
    assert struct.unpack_from("<H", data, 4)[0] == 0xA5E0
    assert struct.unpack_from("<H", data, 6)[0] == 1
    assert struct.unpack_from("<HH", data, 8) == (16, 16)
    assert struct.unpack_from("<H", data, 12)[0] == 8


def test_header_flags_speed_ratio_grid_reserved():
    data = encode_aseprite(grid16(), pico8())
    assert struct.unpack_from("<I", data, 14)[0] & 1
    assert struct.unpack_from("<H", data, 18)[0] == 100
    assert struct.unpack_from("<II", data, 20) == (0, 0)
    assert data[29:32] == b"\x00\x00\x00"
    assert struct.unpack_from("<BB", data, 34) == (1, 1)
    assert struct.unpack_from("<hhHH", data, 36) == (0, 0, 16, 16)
    assert data[44:128] == b"\x00" * 84


def test_header_palette_count_and_transparent_index():
    data = encode_aseprite(grid16(), pico8())
    assert data[28] == 16
    assert struct.unpack_from("<H", data, 32)[0] == 17


# ---------------------------------------------------------------------------------- AC3


def test_frame_header_magic_size_and_counts():
    data = encode_aseprite(grid16(), pico8())
    size, magic, old_count, duration = struct.unpack_from("<IHHH", data, 128)
    reserved = data[138:140]
    (new_count,) = struct.unpack_from("<I", data, 140)
    assert size == len(data) - 128
    assert magic == 0xF1FA
    assert duration == 100
    assert reserved == b"\x00\x00"
    assert (old_count, new_count, len(walk_chunks(data))) == (5, 5, 5)


def test_chunks_tile_the_frame_exactly():
    data = encode_aseprite(grid16(), pico8())
    (frame_size,) = struct.unpack_from("<I", data, 128)
    chunks = walk_chunks(data)
    assert sum(size for _, size, _ in chunks) == frame_size - 16
    assert chunks[-1][0] + chunks[-1][1] == len(data)
    assert all(size >= 6 for _, size, _ in chunks)
    assert [t for _, _, t in chunks] == CHUNK_ORDER


# ---------------------------------------------------------------------------------- AC4


def test_palette_chunk_entries():
    palette = pico8()
    n = palette.shape[0]
    data = encode_aseprite(grid16(), palette)
    chunk = chunk_of(data, 0x2019)
    assert len(chunk) == 6 + 20 + 6 * (n + 1)
    size, first, last = struct.unpack_from("<III", chunk, 6)
    assert (size, first, last) == (n + 1, 0, n)
    entries = [struct.unpack_from("<HBBBB", chunk, 26 + 6 * i) for i in range(n + 1)]
    for i in range(n):
        assert entries[i] == (0, *[int(v) for v in palette[i]], 255)
    assert entries[n][4] == 0
    ase = ao.parse_aseprite(data)
    want = [(*[int(v) for v in c], 255) for c in palette] + [(0, 0, 0, 0)]
    assert ase.palette == want
    assert ase.transparent_index == n


def test_transparent_slot_reserved_without_minus_one():
    palette = pico8()
    data = encode_aseprite(numpy.zeros((8, 8), dtype=int), palette)
    ase = ao.parse_aseprite(data)
    assert ase.ncolors == 17
    assert ase.transparent_index == 16
    assert ase.palette[16][3] == 0


# ---------------------------------------------------------------------------------- AC5


def old_palette_triples(data: bytes) -> tuple[int, int, int, list[tuple[int, int, int]]]:
    chunk = chunk_of(data, 0x0004)
    packets, skip, count = struct.unpack_from("<HBB", chunk, 6)
    real = count or 256
    triples = [struct.unpack_from("<BBB", chunk, 10 + 3 * i) for i in range(real)]
    assert len(chunk) == 10 + 3 * real
    return packets, skip, count, triples


def test_old_palette_chunk_matches_new():
    palette = pico8()
    data = encode_aseprite(grid16(), palette)
    packets, skip, count, triples = old_palette_triples(data)
    assert (packets, skip, count) == (1, 0, 17)
    new = chunk_of(data, 0x2019)
    new_rgb = [struct.unpack_from("<BBB", new, 26 + 6 * i + 2) for i in range(17)]
    assert triples == new_rgb


def test_old_palette_count_byte_at_255_colours():
    palette = synthetic_255()
    data = encode_aseprite(numpy.zeros((4, 4), dtype=int), palette)
    packets, skip, count, triples = old_palette_triples(data)
    assert (packets, skip, count) == (1, 0, 0)
    assert len(triples) == 256
    assert triples[:255] == [tuple(int(v) for v in c) for c in palette]
    assert [t for _, _, t in walk_chunks(data)] == CHUNK_ORDER


# ---------------------------------------------------------------------------------- AC6


def test_colour_profile_chunk_is_srgb():
    data = encode_aseprite(grid16(), pico8())
    chunk = chunk_of(data, 0x2007)
    assert len(chunk) == 22
    assert struct.unpack_from("<HHi", chunk, 6) == (1, 0, 0)
    assert chunk[14:] == b"\x00" * 8
    assert walk_chunks(data)[0][2] == 0x2007


def test_chunk_order():
    data = encode_aseprite(grid16(), pico8())
    assert [t for _, _, t in walk_chunks(data)] == CHUNK_ORDER


# ---------------------------------------------------------------------------------- AC7


def test_layer_chunk():
    data = encode_aseprite(grid16(), pico8())
    chunk = chunk_of(data, 0x2004)
    assert len(chunk) == 32
    flags, ltype, child, _w, _h, blend = struct.unpack_from("<HHHHHH", chunk, 6)
    assert flags == 3 and flags & 8 == 0
    assert (ltype, child, blend) == (0, 0, 0)
    assert chunk[18] == 255
    assert chunk[22:] == b"\x08\x00pixelgen"


# ---------------------------------------------------------------------------------- AC8


def test_cel_chunk_header():
    data = encode_aseprite(grid16(), pico8())
    chunk = chunk_of(data, 0x2005)
    layer, x, y, opacity, ctype, z = struct.unpack_from("<HhhBHh", chunk, 6)
    assert (layer, x, y, opacity, ctype, z) == (0, 0, 0, 255, 2, 0)
    assert chunk[17:22] == b"\x00" * 5
    assert struct.unpack_from("<HH", chunk, 22) == (16, 16)


def test_cel_payload_is_exact_zlib_index_bytes():
    grid = grid16()
    data = encode_aseprite(grid, pico8())
    chunk = chunk_of(data, 0x2005)
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(chunk[26:])
    assert decoder.eof and decoder.unused_data == b""
    assert len(pixels) == 16 * 16
    for y in range(16):
        for x in range(16):
            want = 16 if grid[y, x] < 0 else grid[y, x]
            assert pixels[y * 16 + x] == want


def test_grid_input_not_mutated():
    grid, palette = grid16(), pico8()
    grid_copy, palette_copy = grid.copy(), palette.copy()
    encode_aseprite(grid, palette)
    assert (grid == grid_copy).all() and (palette == palette_copy).all()
    assert (grid == -1).any()


# ---------------------------------------------------------------------------------- AC9


@pytest.mark.parametrize("name", PALETTES)
def test_transparent_index_never_hides_a_real_colour(name):
    palette = load_palette(name)
    n = palette.shape[0]
    grid = numpy.full((16, 16), -1, dtype=int)
    grid[1:-1, 1:-1] = 0
    ase = ao.parse_aseprite(encode_aseprite(grid, palette))
    used = {int(v) for v in grid.ravel() if v >= 0}
    assert ase.transparent_index == n
    assert ase.transparent_index not in used
    canvas = ao.composite_indices(ase)
    assert all(canvas[y][x] == 0 for y in range(1, 15) for x in range(1, 15))
    assert ase.palette[0][3] == 255
    if name == "pico8":
        assert ase.palette[canvas[8][8]] == (0, 0, 0, 255)


# ---------------------------------------------------------------------------------- AC10


def _square_params():
    return [
        pytest.param(s, id=f"pico8-{s}x{s}-s{seed_for(PALETTES.index('pico8'), s, s)}")
        for s in range(8, 129)
    ]


@pytest.mark.parametrize("size", _square_params())
def test_roundtrip_square_sizes(size):
    pi = PALETTES.index("pico8")
    seed = seed_for(pi, size, size)
    grid = marked_grid(numpy.random.default_rng(seed), 16, size, size)
    assert_roundtrip(grid, load_palette("pico8"), seed=seed)


def _by_size_params():
    return [
        pytest.param(name, s, id=f"{name}-{s}x{s}-s{seed_for(i, s, s)}")
        for i, name in enumerate(PALETTES)
        for s in (8, 9, 16, 31, 32, 64, 127, 128)
    ]


@pytest.mark.parametrize(("name", "size"), _by_size_params())
def test_roundtrip_palettes_by_size(name, size):
    palette = load_palette(name)
    seed = seed_for(PALETTES.index(name), size, size)
    grid = marked_grid(numpy.random.default_rng(seed), palette.shape[0], size, size)
    assert_roundtrip(grid, palette, seed=seed)


def _non_square_params():
    return [
        pytest.param(name, w, h, id=f"{name}-{w}x{h}-s{seed_for(i, w, h)}")
        for i, name in enumerate(PALETTES)
        for w, h in ((8, 128), (128, 8), (17, 33), (33, 17))
    ]


@pytest.mark.parametrize(("name", "width", "height"), _non_square_params())
def test_roundtrip_non_square(name, width, height):
    palette = load_palette(name)
    seed = seed_for(PALETTES.index(name), width, height)
    grid = marked_grid(numpy.random.default_rng(seed), palette.shape[0], height, width)
    assert_roundtrip(grid, palette, seed=seed)


SPECIAL_KINDS = ("all-minus1", "no-minus1", "all-0", "all-last", "checker")


def _special_grid(kind: str, n: int, seed: int) -> numpy.ndarray:
    if kind == "all-minus1":
        return numpy.full((16, 16), -1, dtype=int)
    if kind == "no-minus1":
        return numpy.random.default_rng(seed).integers(0, n, size=(16, 16))
    if kind == "all-0":
        return numpy.zeros((16, 16), dtype=int)
    if kind == "all-last":
        return numpy.full((16, 16), n - 1, dtype=int)
    yy, xx = numpy.indices((16, 16))
    return numpy.where((yy + xx) % 2 == 0, 0, -1)


def _special_params():
    return [
        pytest.param(name, kind, id=f"{name}-{kind}-s{seed_for(i, 16, 16)}")
        for i, name in enumerate(PALETTES)
        for kind in SPECIAL_KINDS
    ]


@pytest.mark.parametrize(("name", "kind"), _special_params())
def test_roundtrip_special_grids(name, kind):
    palette = load_palette(name)
    seed = seed_for(PALETTES.index(name), 16, 16)
    assert_roundtrip(_special_grid(kind, palette.shape[0], seed), palette, seed=seed)


# ---------------------------------------------------------------------------------- AC11


@pytest.mark.parametrize("count", [255, 256, 300])
def test_palette_limit_255_ok_256_rejected(count):
    palette = numpy.array([[i % 256, (i * 7) % 256, (i * 13) % 256] for i in range(count)])
    palette = palette.astype(numpy.uint8)
    grid = numpy.arange(64).reshape(8, 8) % count - 1
    if count == 255:
        data = encode_aseprite(grid, palette)
        ase = ao.parse_aseprite(data)
        assert ase.ncolors == 256 and ase.transparent_index == 255
        assert ao.to_sidecar_grid(ase) == grid.tolist()
        return
    with pytest.raises(AsepriteError) as info:
        encode_aseprite(grid, palette)
    text = str(info.value)
    assert "\n" not in text and "at most 255" in text and str(count) in text


# ---------------------------------------------------------------------------------- AC12

_P = pico8()
_G = numpy.zeros((4, 4), dtype=int)
_PALETTE_LINE = "palette_colors must be a uint8 array of shape (N, 3) with N >= 1"
_GRID_LINE = "grid must be a 2-D integer array"

INVALID_CASES = {
    "palette-float": (_G, _P.astype(float), _PALETTE_LINE),
    "palette-n-by-4": (_G, numpy.zeros((4, 4), dtype=numpy.uint8), _PALETTE_LINE),
    "palette-empty": (_G, numpy.zeros((0, 3), dtype=numpy.uint8), _PALETTE_LINE),
    "grid-1d": (numpy.zeros(4, dtype=int), _P, _GRID_LINE),
    "grid-3d": (numpy.zeros((2, 2, 2), dtype=int), _P, _GRID_LINE),
    "grid-float": (numpy.zeros((4, 4), dtype=float), _P, _GRID_LINE),
    "grid-bool": (numpy.zeros((4, 4), dtype=bool), _P, _GRID_LINE),
    "grid-ragged": ([[0, 1], [0]], _P, _GRID_LINE),
    "value--2": (numpy.full((4, 4), -2), _P, "grid index -2 outside -1..15"),
    "value-N": (numpy.full((4, 4), 16), _P, "grid index 16 outside -1..15"),
}


@pytest.mark.parametrize("case", list(INVALID_CASES))
def test_invalid_inputs_raise_aseprite_error(case):
    grid, palette, line = INVALID_CASES[case]
    with pytest.raises(AsepriteError) as info:
        encode_aseprite(grid, palette)
    assert str(info.value) == line


def test_valid_list_of_lists_grid_is_accepted():
    data = encode_aseprite([[0, 1, -1], [2, 3, 4]], _P)
    assert ao.to_sidecar_grid(ao.parse_aseprite(data)) == [[0, 1, -1], [2, 3, 4]]


# ---------------------------------------------------------------------------------- AC13

ACCEPTED_SIZES = [(1, 1), (1, 65535), (65535, 1)]
REJECTED_SIZES = [(0, 8), (8, 0), (1, 65536), (65536, 1)]


@pytest.mark.parametrize(("height", "width"), ACCEPTED_SIZES, ids=lambda v: str(v))
def test_canvas_size_bounds_accepted(height, width):
    grid = numpy.zeros((height, width), dtype=numpy.int8)
    ase = ao.parse_aseprite(encode_aseprite(grid, _P))
    assert (ase.width, ase.height) == (width, height)
    assert ao.to_sidecar_grid(ase) == grid.tolist()


@pytest.mark.parametrize(("height", "width"), REJECTED_SIZES, ids=lambda v: str(v))
def test_canvas_size_bounds_rejected(height, width):
    grid = numpy.zeros((height, width), dtype=numpy.int8)
    with pytest.raises(AsepriteError) as info:
        encode_aseprite(grid, _P)
    assert str(info.value) == (
        f"grid size {width}x{height} is outside the .aseprite range 1x1..65535x65535"
    )


# ---------------------------------------------------------------------------------- AC14


def test_same_input_same_bytes():
    palette, grid = pico8(), grid16()
    first = encode_aseprite(grid, palette)
    assert encode_aseprite(grid, palette) == first
    assert encode_aseprite(grid.astype(numpy.int16), palette) == first
    assert encode_aseprite(grid.tolist(), palette) == first


PINNED = {
    "pico8-16x16": "353e3a5c3169b618f8b82e2999bc2d54c163de4e203ca4c184e952ecbb2558b1",
    "n255-8x8": "03ba011b626bcd78f2a72f660d715d2bb849e0b154631cbcc9b3e7c88cea7de9",
}


def _canonical_inputs():
    return {
        "pico8-16x16": (grid16(), pico8()),
        "n255-8x8": (numpy.arange(64).reshape(8, 8) - 1, synthetic_255()),
    }


@pytest.mark.parametrize("key", list(PINNED))
def test_canonical_form_hash_is_pinned(key):
    grid, palette = _canonical_inputs()[key]
    digest = hashlib.sha256(canonical_form(encode_aseprite(grid, palette))).hexdigest()
    assert digest == PINNED[key]


# ---------------------------------------------------------------------------------- AC16


def _imports(tree: ast.AST) -> list[tuple[str, list[str]]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", [a.name for a in node.names]))
    return found


def test_writer_imports_stay_pure():
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    banned = {"time", "datetime", "random", "secrets", "uuid", "os", "pathlib", "io", "tempfile"}
    for module, _names in _imports(tree):
        top = module.split(".")[0]
        if top == "pixelgen":
            assert module == "pixelgen.errors", f"forbidden pixelgen import: {module}"
            continue
        assert top == "numpy" or top in sys.stdlib_module_names, f"non-stdlib import: {module}"
        assert top not in banned, f"forbidden import: {module}"
    calls = [
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "open" not in calls


def test_writer_tests_use_public_api_only():
    tree = ast.parse(THIS_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pixelgen.aseprite":
            assert not any(a.name.startswith("_") for a in node.names)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert not (node.value.id == "aseprite" and node.attr.startswith("_"))


# ---------------------------------------------------------------------------------- AC20

_DUMP_LUA = """\
local spr = app.activeSprite
print("size " .. spr.width .. " " .. spr.height)
print("indexed " .. tostring(spr.colorMode == ColorMode.INDEXED))
print("transparent " .. spr.transparentColor)
local pal = spr.palettes[1]
print("ncolors " .. #pal)
for i = 0, #pal - 1 do
  local c = pal:getColor(i)
  print("pal " .. i .. " " .. c.red .. " " .. c.green .. " " .. c.blue .. " " .. c.alpha)
end
local img = Image(spr.spec)
img:drawSprite(spr, 1)
for y = 0, spr.height - 1 do
  local row = {}
  for x = 0, spr.width - 1 do
    row[#row + 1] = img:getPixel(x, y)
  end
  print("row " .. y .. " " .. table.concat(row, " "))
end
"""


def _case_black0_border() -> tuple[numpy.ndarray, numpy.ndarray]:
    grid = numpy.zeros((12, 12), dtype=int)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = -1
    return grid, load_palette("pico8")


def _case_pico8_17x33() -> tuple[numpy.ndarray, numpy.ndarray]:
    grid = marked_grid(numpy.random.default_rng(17033), 16, 17, 33)
    grid[1, 0:17] = numpy.arange(-1, 16)
    return grid, load_palette("pico8")


def _case_n255() -> tuple[numpy.ndarray, numpy.ndarray]:
    return numpy.arange(256).reshape(16, 16) - 1, synthetic_255()


def _case_gameboy() -> tuple[numpy.ndarray, numpy.ndarray]:
    yy, xx = numpy.indices((4, 9))
    return (yy + xx) % 4, load_palette("gameboy")


REAL_CASES = {
    "pico8-black0-border-12x12": _case_black0_border,
    "pico8-17x33": _case_pico8_17x33,
    "n255-16x16": _case_n255,
    "gameboy-no-minus1-4x9": _case_gameboy,
}


def _parse_dump(text: str) -> dict:
    out: dict = {"pal": {}, "row": {}}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key, rest = parts[0], [int(p) if p.lstrip("-").isdigit() else p for p in parts[1:]]
        if key == "pal":
            out["pal"][rest[0]] = tuple(rest[1:])
        elif key == "row":
            out["row"][rest[0]] = rest[1:]
        else:
            out[key] = rest
    return out


@pytest.mark.parametrize("case", list(REAL_CASES))
def test_real_aseprite_round_trip(case, tmp_path):
    exe = require_aseprite()
    grid, palette = REAL_CASES[case]()
    n = palette.shape[0]
    height, width = grid.shape
    ase_path = tmp_path / f"{case}.aseprite"
    png_path = tmp_path / f"{case}.png"
    lua_path = tmp_path / "pxg26_dump.lua"
    ase_path.write_bytes(encode_aseprite(grid, palette))
    lua_path.write_text(_DUMP_LUA, encoding="utf-8")
    root = tmp_path / "ase-root"

    saved = run_aseprite(exe, ["-b", str(ase_path), "--save-as", str(png_path)], root=root)
    dumped = run_aseprite(exe, ["-b", str(ase_path), "--script", str(lua_path)], root=root)
    assert saved.returncode == 0
    assert dumped.returncode == 0

    dump = _parse_dump(dumped.stdout)
    assert dump["size"] == [width, height]
    assert dump["indexed"] == ["true"]
    assert dump["transparent"] == [n]
    assert dump["ncolors"] == [n + 1]
    for i in range(n):
        assert dump["pal"][i] == (*[int(v) for v in palette[i]], 255), f"pal {i}"
    assert dump["pal"][n] == (0, 0, 0, 0)
    for y in range(height):
        want = [n if v < 0 else int(v) for v in grid[y]]
        assert dump["row"][y] == want, f"row {y}"

    rgba = numpy.array(Image.open(png_path).convert("RGBA"))
    assert rgba.shape == (height, width, 4)
    transparent = grid < 0
    assert ((rgba[..., 3] == 0) == transparent).all()
    assert (rgba[~transparent][:, 3] == 255).all()
    assert (rgba[~transparent][:, :3] == palette[grid[~transparent]]).all()
