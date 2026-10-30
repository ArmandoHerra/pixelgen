"""Pure stdlib writer for minimal indexed `.aseprite` bytes (PXG-26).

One frame, one normal layer named ``pixelgen``, one full-canvas zlib cel, depth 8. The transparent
slot is appended after the palette at index N and is always reserved, so a real colour 0 is never
hidden. The module reads no clock, touches no file and imports only ``pixelgen.errors``.
"""

import struct
import zlib
from typing import Final

import numpy

from pixelgen.errors import AsepriteError

MAX_ASEPRITE_COLOURS: Final[int] = 255

_ZLIB_LEVEL: Final[int] = 9
_MAX_SIDE: Final[int] = 65535


def _chunk(chunk_type: int, body: bytes) -> bytes:
    return struct.pack("<IH", 6 + len(body), chunk_type) + body


def _profile_chunk() -> bytes:
    return _chunk(0x2007, struct.pack("<HHi8x", 1, 0, 0))


def _old_palette_chunk(rgb: numpy.ndarray) -> bytes:
    body = struct.pack("<HBB", 1, 0, len(rgb) % 256) + rgb.tobytes()
    return _chunk(0x0004, body)


def _palette_chunk(rgb: numpy.ndarray) -> bytes:
    count = len(rgb)
    body = struct.pack("<III8x", count, 0, count - 1)
    for r, g, b, a in _entries(rgb):
        body += struct.pack("<HBBBB", 0, r, g, b, a)
    return _chunk(0x2019, body)


def _entries(rgb: numpy.ndarray) -> list[tuple[int, int, int, int]]:
    """Entries 0..N-1 opaque, the last (transparent) entry alpha 0."""
    entries = [(int(r), int(g), int(b), 255) for r, g, b in rgb]
    entries[-1] = (0, 0, 0, 0)
    return entries


def _layer_chunk() -> bytes:
    name = b"pixelgen"
    body = struct.pack("<HHHHHHB3xH", 3, 0, 0, 0, 0, 0, 255, len(name)) + name
    return _chunk(0x2004, body)


def _cel_chunk(indices: numpy.ndarray) -> bytes:
    height, width = indices.shape
    payload = zlib.compress(indices.tobytes(order="C"), _ZLIB_LEVEL)
    body = struct.pack("<HhhBHh5xHH", 0, 0, 0, 255, 2, 0, width, height) + payload
    return _chunk(0x2005, body)


def _check_palette(palette_colors: object) -> numpy.ndarray:
    if (
        not isinstance(palette_colors, numpy.ndarray)
        or palette_colors.dtype != numpy.uint8
        or palette_colors.ndim != 2
        or palette_colors.shape[1] != 3
        or palette_colors.shape[0] < 1
    ):
        raise AsepriteError("palette_colors must be a uint8 array of shape (N, 3) with N >= 1")
    if palette_colors.shape[0] > MAX_ASEPRITE_COLOURS:
        raise AsepriteError(
            f".aseprite output supports at most {MAX_ASEPRITE_COLOURS} palette colours "
            f"(one index is reserved for transparency); this palette has "
            f"{palette_colors.shape[0]}"
        )
    return palette_colors


def _check_grid(grid: object, n: int) -> numpy.ndarray:
    try:
        cells = numpy.asarray(grid)
    except ValueError as exc:
        raise AsepriteError("grid must be a 2-D integer array") from exc
    if cells.ndim != 2 or cells.dtype.kind not in "iu":
        raise AsepriteError("grid must be a 2-D integer array")
    height, width = cells.shape
    if not (1 <= width <= _MAX_SIDE and 1 <= height <= _MAX_SIDE):
        raise AsepriteError(
            f"grid size {width}x{height} is outside the .aseprite range 1x1..65535x65535"
        )
    low, high = int(cells.min()), int(cells.max())
    if low < -1 or high > n - 1:
        bad = low if low < -1 else high
        raise AsepriteError(f"grid index {bad} outside -1..{n - 1}")
    return cells


def encode_aseprite(grid, palette_colors) -> bytes:
    """Encode a ``-1..N-1`` index grid and an ``(N, 3)`` uint8 palette as `.aseprite` bytes."""
    palette = _check_palette(palette_colors)
    n = palette.shape[0]
    cells = _check_grid(grid, n)
    height, width = cells.shape
    indices = numpy.ascontiguousarray(numpy.where(cells < 0, n, cells).astype(numpy.uint8))
    full = numpy.vstack([palette, numpy.zeros((1, 3), dtype=numpy.uint8)])
    chunks = [
        _profile_chunk(),
        _old_palette_chunk(full),
        _palette_chunk(full),
        _layer_chunk(),
        _cel_chunk(indices),
    ]
    frame_body = b"".join(chunks)
    frame = struct.pack("<IHHH2xI", 16 + len(frame_body), 0xF1FA, len(chunks), 100, len(chunks))
    size = 128 + len(frame) + len(frame_body)
    header = struct.pack(
        "<IHHHHHIHIIB3xHBBhhHH84x",
        size,
        0xA5E0,
        1,
        width,
        height,
        8,
        1,
        100,
        0,
        0,
        n,
        n + 1,
        1,
        1,
        0,
        0,
        16,
        16,
    )
    return header + frame + frame_body
