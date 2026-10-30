"""Advisory LibreSprite compatibility helper (PXG-28).

``make <dir>`` writes a small case set with the shipped ``.aseprite`` writer plus the native PNG
for each case. ``compare <dir>`` pixel-compares each native PNG against the PNG LibreSprite
rendered from the same ``.aseprite`` file, under the alpha-0 rule: where the native pixel is fully
transparent the render must be fully transparent too (its RGB is ignored); elsewhere RGBA must be
equal. Nothing here reads the pixelgen home, the network or the LLM layer.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from pixelgen.aseprite import encode_aseprite
from pixelgen.palettes import load_palette
from pixelgen.pixelate import grid_to_rgba

# (case name, palette name, (height, width))
CASES: tuple[tuple[str, str, tuple[int, int]], ...] = (
    ("pico8-16x16", "pico8", (16, 16)),
    ("endesga32-17x33", "endesga32", (33, 17)),
    ("grayscale4-8x8-solid", "grayscale4", (8, 8)),
)

_MAX_REPORTED = 5


def build_case(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(grid, palette)`` for a case: grid uses ``-1`` for transparent cells."""
    for case, palette_name, (height, width) in CASES:
        if case != name:
            continue
        palette = load_palette(palette_name)
        n = palette.shape[0]
        if name == "pico8-16x16":
            grid = np.full((height, width), -1, dtype=np.int64)
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    grid[y, x] = ((y - 1) * (width - 2) + (x - 1)) % n
        elif name == "endesga32-17x33":
            grid = np.random.default_rng(28).integers(-1, n, (height, width))
        else:
            rows, cols = np.indices((height, width))
            grid = (rows + cols) % n
        return grid.astype(np.int64), palette
    raise KeyError(name)


def make(out_dir: Path) -> list[str]:
    """Write ``<case>.aseprite`` and the native ``<case>.png`` for every case."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, _, _ in CASES:
        grid, palette = build_case(name)
        (out_dir / f"{name}.aseprite").write_bytes(encode_aseprite(grid, palette))
        Image.fromarray(grid_to_rgba(grid, palette)).save(out_dir / f"{name}.png")
        lines.append(f"aseprite-compat: wrote {name}")
    return lines


def compare_images(ours: Image.Image, theirs: Image.Image) -> list[str]:
    """Alpha-0 compare. Empty list means equal; else up to 5 diff lines plus a total."""
    a = ours.convert("RGBA")
    b = theirs.convert("RGBA")
    if a.size != b.size:
        return [f"size ours={a.size[0]}x{a.size[1]} theirs={b.size[0]}x{b.size[1]}"]
    pa = np.asarray(a)
    pb = np.asarray(b)
    transparent = pa[..., 3] == 0
    bad = np.where(transparent, pb[..., 3] != 0, np.any(pa != pb, axis=-1))
    count = int(bad.sum())
    if count == 0:
        return []
    lines = []
    for y, x in np.argwhere(bad)[:_MAX_REPORTED]:
        ours_px = tuple(int(v) for v in pa[y, x])
        theirs_px = tuple(int(v) for v in pb[y, x])
        lines.append(f"x={int(x)} y={int(y)} ours={ours_px} theirs={theirs_px}")
    lines.append(f"{count} pixels differ")
    return lines


def compare(directory: Path) -> int:
    """Compare every case; return 1 if any case mismatched or is missing its render."""
    directory = Path(directory)
    failed = False
    for source in sorted(directory.glob("*.aseprite")):
        stem = source.stem
        native = directory / f"{stem}.png"
        render = directory / f"{stem}.libresprite.png"
        if not render.is_file():
            problems = [f"missing render {render.name}"]
        else:
            with Image.open(native) as ours, Image.open(render) as theirs:
                problems = compare_images(ours, theirs)
        if problems:
            failed = True
            print(f"aseprite-compat: {stem}: MISMATCH")
            for line in problems:
                print(f"  {line}")
        else:
            print(f"aseprite-compat: {stem}: ok")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aseprite_compat")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("make", "compare"):
        sub.add_parser(name).add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "make":
        for line in make(args.directory):
            print(line)
        return 0
    return compare(args.directory)


if __name__ == "__main__":
    sys.exit(main())
