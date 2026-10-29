"""Load, list and validate Lospec-style ``.hex`` palettes shipped in the package."""

import importlib.resources
import re
from importlib.resources.abc import Traversable

import numpy

from pixelgen.errors import PaletteError

_MAX_COLOURS = 256
_RRGGBB = re.compile(r"[0-9A-Fa-f]{6}")


def _palette_dir() -> Traversable:
    # Not files("pixelgen.palettes"): the data dir has no __init__.py (it would shadow this module).
    return importlib.resources.files("pixelgen") / "palettes"


def list_palettes() -> list[str]:
    return sorted(
        entry.name[:-4]
        for entry in _palette_dir().iterdir()
        if entry.is_file() and entry.name.endswith(".hex")
    )


def parse_hex(text: str, *, source: str = "<string>") -> numpy.ndarray:
    """Parse one ``RRGGBB`` per line; ``;`` comments and blank lines are skipped."""
    text = text.removeprefix("﻿")
    rows: list[tuple[int, int, int]] = []
    seen: dict[str, int] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if not _RRGGBB.fullmatch(line):
            raise PaletteError(f"{source}: line {number}: expected RRGGBB, got {line!r}")
        lowered = line.lower()
        if lowered in seen:
            raise PaletteError(
                f"{source}: line {number}: duplicate colour {lowered} "
                f"(first seen on line {seen[lowered]})"
            )
        seen[lowered] = number
        rows.append((int(lowered[0:2], 16), int(lowered[2:4], 16), int(lowered[4:6], 16)))
    if not rows:
        raise PaletteError(f"{source}: no colours")
    if len(rows) > _MAX_COLOURS:
        raise PaletteError(f"{source}: {len(rows)} colours exceeds the maximum of {_MAX_COLOURS}")
    return numpy.array(rows, dtype=numpy.uint8).reshape(-1, 3)


def load_palette(name: str) -> numpy.ndarray:
    available = list_palettes()
    if name not in available:
        raise PaletteError(f"unknown palette {name!r}; available: {', '.join(available)}")
    data = (_palette_dir() / f"{name}.hex").read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise PaletteError(f"{name}.hex: not valid UTF-8") from None
    return parse_hex(text, source=f"{name}.hex")
