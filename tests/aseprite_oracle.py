"""Independent `.aseprite` test oracle (PXG-24).

A general reader for indexed, normal-layer Aseprite files. It uses only the standard library
(struct, zlib, dataclasses, json, argparse) and never touches product code, so it can check bytes
written by `pixelgen.aseprite` without sharing a misreading of the file spec with it. It is proven
against real Aseprite bytes vendored under `tests/fixtures/aseprite/`.

Usage as a script (zero-spend artefact probe):

    python tests/aseprite_oracle.py FILE [--sidecar JSON]

Exit 0 on a clean parse (and sidecar match), 1 on a mismatch or any failure, 2 on a usage error.
"""

import argparse
import json
import struct
import sys
import zlib
from dataclasses import dataclass

HEADER_SIZE = 128
FRAME_HEADER_SIZE = 16
CHUNK_HEADER_SIZE = 6
FILE_MAGIC = 0xA5E0
FRAME_MAGIC = 0xF1FA

CHUNK_OLD_PALETTE = 0x0004
CHUNK_LAYER = 0x2004
CHUNK_CEL = 0x2005
CHUNK_PALETTE = 0x2019

LAYER_VISIBLE = 1
LAYER_BACKGROUND = 8


class AseFormatError(Exception):
    """The only exception this module raises on malformed input."""


@dataclass(frozen=True)
class AseLayer:
    flags: int
    layer_type: int
    child_level: int
    blend_mode: int
    opacity: int
    name: str


@dataclass(frozen=True)
class AseCel:
    frame: int
    layer: int
    x: int
    y: int
    opacity: int
    cel_type: int
    z_index: int
    width: int
    height: int
    pixels: bytes
    linked_frame: int | None


@dataclass(frozen=True)
class AseFrame:
    duration_ms: int
    old_chunk_count: int
    new_chunk_count: int
    chunk_types: list[int]


@dataclass(frozen=True)
class AseFile:
    file_size: int
    width: int
    height: int
    color_depth: int
    flags: int
    speed: int
    transparent_index: int
    ncolors: int
    pixel_width: int
    pixel_height: int
    palette: list[tuple[int, int, int, int]]
    old_palette: list[tuple[int, int, int]] | None
    frames: list[AseFrame]
    layers: list[AseLayer]
    cels: list[AseCel]


def _unpack(fmt: str, data: bytes, offset: int, what: str) -> tuple:
    try:
        return struct.unpack_from(fmt, data, offset)
    except struct.error as exc:
        raise AseFormatError(f"truncated {what} at offset {offset}") from exc


def _read_string(data: bytes, offset: int, end: int, what: str) -> tuple[str, int]:
    (length,) = _unpack("<H", data, offset, what)
    start = offset + 2
    if start + length > end:
        raise AseFormatError(f"{what} string overruns its chunk")
    try:
        return data[start : start + length].decode("utf-8"), start + length
    except UnicodeDecodeError as exc:
        raise AseFormatError(f"{what} string is not valid UTF-8") from exc


def _decode_zlib(blob: bytes, expected: int) -> bytes:
    decoder = zlib.decompressobj()
    try:
        out = decoder.decompress(blob, expected + 1)
    except zlib.error as exc:
        raise AseFormatError("cel zlib stream is corrupt") from exc
    if len(out) != expected:
        raise AseFormatError(f"cel pixel length {len(out)} != w*h*bpp {expected}")
    if not decoder.eof:
        raise AseFormatError("cel zlib stream is truncated")
    if decoder.unused_data or decoder.unconsumed_tail:
        raise AseFormatError("cel zlib stream is followed by trailing data")
    return out


def _parse_palette(
    data: bytes, start: int, end: int, palette: list[tuple[int, int, int, int]]
) -> None:
    new_size, first, last = _unpack("<III", data, start, "palette chunk")
    if first > last or last >= new_size:
        raise AseFormatError(f"palette range {first}..{last} invalid for size {new_size}")
    if new_size > 0x10000:
        raise AseFormatError(f"palette size {new_size} is unreasonable")
    if len(palette) < new_size:
        palette.extend([(0, 0, 0, 0)] * (new_size - len(palette)))
    del palette[new_size:]
    pos = start + 20
    for index in range(first, last + 1):
        (flags,) = _unpack("<H", data, pos, "palette entry")
        r, g, b, a = _unpack("<BBBB", data, pos + 2, "palette entry")
        pos += 6
        if flags & 1:
            _, pos = _read_string(data, pos, end, "palette entry name")
        if pos > end:
            raise AseFormatError("palette entry overruns its chunk")
        palette[index] = (r, g, b, a)
    if pos != end:
        raise AseFormatError("palette chunk has trailing bytes")


def _parse_old_palette(data: bytes, start: int, end: int) -> list[tuple[int, int, int]]:
    (packets,) = _unpack("<H", data, start, "old palette chunk")
    pos = start + 2
    colors: list[tuple[int, int, int]] = []
    for _ in range(packets):
        skip, count = _unpack("<BB", data, pos, "old palette packet")
        pos += 2
        count = count or 256
        if pos + 3 * count > end:
            raise AseFormatError("old palette packet overruns its chunk")
        colors.extend([(0, 0, 0)] * skip)
        for _ in range(count):
            colors.append(_unpack("<BBB", data, pos, "old palette colour"))
            pos += 3
    if pos != end:
        raise AseFormatError("old palette chunk has trailing bytes")
    return colors


def _parse_layer(data: bytes, start: int, end: int) -> AseLayer:
    flags, layer_type, child_level, _w, _h, blend = _unpack("<HHHHHH", data, start, "layer chunk")
    (opacity,) = _unpack("<B", data, start + 12, "layer chunk")
    name, _ = _read_string(data, start + 16, end, "layer name")
    if layer_type != 0:
        raise AseFormatError(f"unsupported layer type {layer_type}")
    if flags & LAYER_BACKGROUND:
        raise AseFormatError("unsupported layer: background layer")
    return AseLayer(flags, layer_type, child_level, blend, opacity, name)


def _parse_cel(
    data: bytes,
    start: int,
    end: int,
    frame: int,
    layers: list[AseLayer],
    cels: list[AseCel],
    bpp: int,
) -> AseCel:
    layer, x, y, opacity, cel_type, z_index = _unpack("<HhhBHh", data, start, "cel chunk")
    body = start + 16  # 2+2+2+1+2+2 fixed fields, then 5 reserved bytes
    if body > end:
        raise AseFormatError("cel chunk is shorter than its fixed fields")
    if layer >= len(layers):
        raise AseFormatError(f"cel names layer {layer} but only {len(layers)} exist")
    if any(c.frame == frame and c.layer == layer for c in cels):
        raise AseFormatError(f"frame {frame} has two cels on layer {layer}")
    if cel_type in (0, 2):
        width, height = _unpack("<HH", data, body, "cel size")
        pixels_at = body + 4
        expected = width * height * bpp
        if cel_type == 0:
            if pixels_at + expected > end:
                raise AseFormatError("raw cel pixels overrun their chunk")
            pixels = data[pixels_at : pixels_at + expected]
        else:
            pixels = _decode_zlib(data[pixels_at:end], expected)
        return AseCel(frame, layer, x, y, opacity, cel_type, z_index, width, height, pixels, None)
    if cel_type == 1:
        (linked,) = _unpack("<H", data, body, "linked cel")
        target = next((c for c in cels if c.frame == linked and c.layer == layer), None)
        if target is None:
            raise AseFormatError(f"linked cel points at missing frame {linked} layer {layer}")
        return AseCel(
            frame,
            layer,
            target.x,
            target.y,
            opacity,
            1,
            z_index,
            target.width,
            target.height,
            target.pixels,
            linked,
        )
    raise AseFormatError(f"unsupported cel type {cel_type}")


def _parse_frame(
    data: bytes,
    offset: int,
    index: int,
    ctx: dict,
) -> tuple[AseFrame, int]:
    if offset + FRAME_HEADER_SIZE > len(data):
        raise AseFormatError(f"frame {index} header overruns the file")
    size, magic, old_count, duration, _pad, new_count = _unpack(
        "<IHHH2sI", data, offset, "frame header"
    )
    if magic != FRAME_MAGIC:
        raise AseFormatError(f"frame {index} magic 0x{magic:04X} != 0xF1FA")
    if size < FRAME_HEADER_SIZE:
        raise AseFormatError(f"frame {index} size {size} < {FRAME_HEADER_SIZE}")
    end = offset + size
    if end > len(data):
        raise AseFormatError(f"frame {index} overruns the file")
    chunk_types: list[int] = []
    pos = offset + FRAME_HEADER_SIZE
    while pos < end:
        if pos + CHUNK_HEADER_SIZE > end:
            raise AseFormatError(f"frame {index} has a partial chunk header")
        csize, ctype = _unpack("<IH", data, pos, "chunk header")
        if csize < CHUNK_HEADER_SIZE:
            raise AseFormatError(f"frame {index} chunk 0x{ctype:04X} size {csize} < 6")
        if pos + csize > end:
            raise AseFormatError(f"frame {index} chunk 0x{ctype:04X} overruns the frame")
        body, cend = pos + CHUNK_HEADER_SIZE, pos + csize
        chunk_types.append(ctype)
        if ctype == CHUNK_PALETTE:
            _parse_palette(data, body, cend, ctx["palette"])
        elif ctype == CHUNK_OLD_PALETTE:
            if ctx["old_palette"] is None:
                ctx["old_palette"] = _parse_old_palette(data, body, cend)
        elif ctype == CHUNK_LAYER:
            ctx["layers"].append(_parse_layer(data, body, cend))
        elif ctype == CHUNK_CEL:
            ctx["cels"].append(
                _parse_cel(data, body, cend, index, ctx["layers"], ctx["cels"], ctx["bpp"])
            )
        pos = cend
    if new_count:
        if old_count != min(new_count, 0xFFFF):
            raise AseFormatError(
                f"frame {index} old_chunk_count {old_count} != min(new {new_count}, 0xFFFF)"
            )
        expected = new_count
    else:
        expected = old_count
    if len(chunk_types) != expected:
        raise AseFormatError(
            f"frame {index} holds {len(chunk_types)} chunks, header says {expected}"
        )
    return AseFrame(duration, old_count, new_count, chunk_types), end


def parse_aseprite(data: bytes) -> AseFile:
    """Parse `.aseprite` bytes; raise `AseFormatError` on any structural inconsistency."""
    if len(data) < HEADER_SIZE:
        raise AseFormatError(f"file is {len(data)} bytes, shorter than the 128-byte header")
    file_size, magic, nframes, width, height, depth, flags, speed = _unpack(
        "<IHHHHHIH", data, 0, "header"
    )
    (transparent_index,) = _unpack("<B", data, 28, "header")
    ncolors, pixel_width, pixel_height = _unpack("<HBB", data, 32, "header")
    if file_size != len(data):
        raise AseFormatError(f"file_size {file_size} != len {len(data)}")
    if magic != FILE_MAGIC:
        raise AseFormatError(f"header magic 0x{magic:04X} != 0xA5E0")
    if depth not in (8, 16, 32):
        raise AseFormatError(f"color_depth {depth} is not 8, 16 or 32")
    ctx: dict = {"palette": [], "old_palette": None, "layers": [], "cels": [], "bpp": depth // 8}
    frames: list[AseFrame] = []
    offset = HEADER_SIZE
    for index in range(nframes):
        frame, offset = _parse_frame(data, offset, index, ctx)
        frames.append(frame)
    if offset != len(data):
        raise AseFormatError(f"{len(data) - offset} trailing bytes after the last frame")
    if not ctx["palette"]:
        raise AseFormatError("no 0x2019 palette chunk")
    return AseFile(
        file_size=file_size,
        width=width,
        height=height,
        color_depth=depth,
        flags=flags,
        speed=speed,
        transparent_index=transparent_index,
        ncolors=ncolors,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        palette=ctx["palette"],
        old_palette=ctx["old_palette"],
        frames=frames,
        layers=ctx["layers"],
        cels=ctx["cels"],
    )


def composite_indices(ase: AseFile, frame: int = 0) -> list[list[int]]:
    """Composite one frame to H rows of W palette indices (indexed files only)."""
    if ase.color_depth != 8:
        raise AseFormatError(f"composite_indices needs color_depth 8, got {ase.color_depth}")
    if not 0 <= frame < len(ase.frames):
        raise AseFormatError(f"frame {frame} out of range 0..{len(ase.frames) - 1}")
    transparent = ase.transparent_index
    canvas = [[transparent] * ase.width for _ in range(ase.height)]
    for layer_index, layer in enumerate(ase.layers):
        if not layer.flags & LAYER_VISIBLE:
            continue
        for cel in ase.cels:
            if cel.frame != frame or cel.layer != layer_index:
                continue
            for v in range(cel.height):
                cy = cel.y + v
                if not 0 <= cy < ase.height:
                    continue
                row = cel.pixels[v * cel.width : (v + 1) * cel.width]
                for u, value in enumerate(row):
                    cx = cel.x + u
                    if 0 <= cx < ase.width and value != transparent:
                        canvas[cy][cx] = value
    return canvas


def to_sidecar_grid(ase: AseFile) -> list[list[int]]:
    """Frame-0 indices with every transparent-index cell mapped to -1."""
    t = ase.transparent_index
    return [[-1 if v == t else v for v in row] for row in composite_indices(ase, 0)]


def composite_rgba(ase: AseFile, frame: int = 0) -> list[list[tuple[int, int, int, int]]]:
    """Map the composited indices through the palette; the transparent index is (0, 0, 0, 0)."""
    t = ase.transparent_index
    out = []
    for row in composite_indices(ase, frame):
        rgba_row = []
        for v in row:
            if v == t:
                rgba_row.append((0, 0, 0, 0))
            elif v < len(ase.palette):
                rgba_row.append(ase.palette[v])
            else:
                raise AseFormatError(f"index {v} outside the {len(ase.palette)}-entry palette")
        out.append(rgba_row)
    return out


def compare_with_sidecar(ase: AseFile, sidecar: dict) -> list[str]:
    """Return mismatch lines (empty means match). Raises KeyError/TypeError/ValueError on a
    malformed sidecar."""
    spec = sidecar["spec"]
    want_w, want_h = spec["width"], spec["height"]
    colors = sidecar["palette"]["colors"]
    grid = sidecar["grid"]
    n = len(colors)
    lines: list[str] = []

    if (ase.width, ase.height) != (want_w, want_h):
        lines.append(f"MISMATCH size: file {ase.width}x{ase.height} != sidecar {want_w}x{want_h}")

    if len(ase.palette) != n + 1:
        lines.append(f"MISMATCH palette: length {len(ase.palette)} != {n + 1}")
    bad: list[str] = []
    for i in range(min(n + 1, len(ase.palette))):
        have = ase.palette[i]
        if i < n:
            want = (int(colors[i][0:2], 16), int(colors[i][2:4], 16), int(colors[i][4:6], 16), 255)
            if have != want:
                bad.append(f"MISMATCH palette entry {i}: file {have} != sidecar {want}")
        elif have[3] != 0:
            bad.append(f"MISMATCH palette entry {i}: file {have} is not transparent (alpha 0)")
    lines.extend(bad[:5])
    if bad:
        lines.append(f"MISMATCH palette: {len(bad)} entries differ")

    if ase.transparent_index != n:
        lines.append(f"MISMATCH transparent_index: file {ase.transparent_index} != {n}")

    have_grid = to_sidecar_grid(ase)
    if len(grid) != ase.height or any(len(r) != ase.width for r in grid):
        h2 = len(grid)
        w2 = len(grid[0]) if grid else 0
        lines.append(f"MISMATCH grid: shape {ase.height}x{ase.width} != sidecar {h2}x{w2}")
    else:
        diffs = [
            (x, y, a, b)
            for y, (row, want_row) in enumerate(zip(have_grid, grid, strict=True))
            for x, (a, b) in enumerate(zip(row, want_row, strict=True))
            if a != b
        ]
        for x, y, a, b in diffs[:5]:
            lines.append(f"MISMATCH grid x={x} y={y}: file {a} != sidecar {b}")
        if diffs:
            lines.append(f"MISMATCH grid: {len(diffs)} of {ase.width * ase.height} cells differ")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aseprite_oracle", description=__doc__.splitlines()[0])
    parser.add_argument("file")
    parser.add_argument("--sidecar", default=None)
    args = parser.parse_args(argv)
    try:
        with open(args.file, "rb") as fh:
            ase = parse_aseprite(fh.read())
        composite_indices(ase, 0)
        mismatches: list[str] = []
        if args.sidecar is not None:
            with open(args.sidecar, encoding="utf-8") as fh:
                sidecar = json.load(fh)
            mismatches = compare_with_sidecar(ase, sidecar)
    except (AseFormatError, OSError) as exc:
        print(f"aseprite_oracle: error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"aseprite_oracle: error: sidecar is unusable: {exc!r}", file=sys.stderr)
        return 1
    if mismatches:
        for line in mismatches:
            print(line)
        return 1
    print(
        f"OK {args.file}: {ase.width}x{ase.height} depth {ase.color_depth}, "
        f"{len(ase.frames)} frame(s), transparent_index {ase.transparent_index}, "
        f"{len(ase.palette)} palette entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
