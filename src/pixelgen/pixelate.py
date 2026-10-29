"""Deterministic pixelation pipeline: pure NumPy, no RNG (PXG-04).

Every image is an RGBA ``uint8`` array of shape ``(H, W, 4)``. ``pixelate`` composes the
steps; its output has exactly ``spec.height x spec.width`` pixels, each either a palette
colour (alpha 255) or canonical transparent ``(0, 0, 0, 0)``.
"""

from __future__ import annotations

import numpy as np

from pixelgen.errors import SpecError
from pixelgen.spec import ArtSpec

TRANSPARENT = (0, 0, 0, 0)
ALPHA_THRESHOLD = 128  # alpha < 128 is transparent, >= 128 is opaque
BAYER_4X4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.int64)
BAYER_SPREAD = 64  # offset = ((M + 0.5) / 16 - 0.5) * BAYER_SPREAD == 4 * M - 30

_DITHERS = ("none", "floyd-steinberg", "bayer")


# --------------------------------------------------------------------------- validation


def _check_rgba(img: np.ndarray, name: str = "img") -> None:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 numpy array")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"{name} must have shape (H, W, 4), got {img.shape}")
    if img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError(f"{name} must not be empty")


def _check_palette(palette: np.ndarray) -> None:
    if not isinstance(palette, np.ndarray) or palette.dtype != np.uint8:
        raise ValueError("palette must be a uint8 numpy array")
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must have shape (N, 3), got {palette.shape}")
    if palette.shape[0] < 1:
        raise ValueError("palette must not be empty")


def _out(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.uint8)


# --------------------------------------------------------------------------- luminance


def _luma(palette: np.ndarray) -> np.ndarray:
    p = palette.astype(np.int64)
    return 299 * p[:, 0] + 587 * p[:, 1] + 114 * p[:, 2]


def darkest_index(palette: np.ndarray) -> int:
    """Index of the darkest colour (Rec. 601 integer luma); ties go to the lowest index."""
    _check_palette(palette)
    return int(np.argmin(_luma(palette)))


def lightest_index(palette: np.ndarray) -> int:
    """Index of the lightest colour (Rec. 601 integer luma); ties go to the lowest index."""
    _check_palette(palette)
    return int(np.argmax(_luma(palette)))


# --------------------------------------------------------------------------- downscale


def downscale(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbour, centre-sampled resize to ``h x w``; alpha is copied untouched."""
    _check_rgba(img)
    if w < 1 or h < 1:
        raise ValueError(f"target size must be >= 1x1, got {w}x{h}")
    src_h, src_w = img.shape[:2]
    ys = ((2 * np.arange(h, dtype=np.int64) + 1) * src_h) // (2 * h)
    xs = ((2 * np.arange(w, dtype=np.int64) + 1) * src_w) // (2 * w)
    return _out(img[ys[:, None], xs[None, :]])


# --------------------------------------------------------------------------- quantize


def _nearest(colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Palette index of the nearest colour (squared RGB distance, lowest index on ties)."""
    c = colors.astype(np.int64)
    p = palette.astype(np.int64)
    d = ((c[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


def quantize(img: np.ndarray, palette: np.ndarray, dither: str = "none") -> np.ndarray:
    """Map every opaque pixel to the palette; binarize alpha (transparent -> (0,0,0,0))."""
    _check_rgba(img)
    _check_palette(palette)
    if dither not in _DITHERS:
        raise ValueError(f"unknown dither {dither!r}; expected one of {_DITHERS}")
    h, w = img.shape[:2]
    opaque = img[..., 3] >= ALPHA_THRESHOLD
    out = np.zeros((h, w, 4), dtype=np.uint8)

    if dither == "floyd-steinberg":
        idx = _floyd_steinberg(img[..., :3], opaque, palette)
        out[opaque, :3] = palette[idx[opaque]]
    else:
        rgb = img[..., :3].astype(np.int64)
        if dither == "bayer":
            m = BAYER_4X4[np.arange(h)[:, None] % 4, np.arange(w)[None, :] % 4]
            offset = 4 * m - 30  # == ((M + 0.5) / 16 - 0.5) * BAYER_SPREAD
            rgb = np.clip(rgb + offset[..., None], 0, 255)
        idx = np.zeros((h, w), dtype=np.int64)
        if opaque.any():
            idx[opaque] = _nearest(rgb[opaque], palette)
        out[opaque, :3] = palette[idx[opaque]]
    out[opaque, 3] = 255
    return _out(out)


def _floyd_steinberg(rgb: np.ndarray, opaque: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = opaque.shape
    buf = rgb.astype(np.float64)
    pal_f = palette.astype(np.float64)
    idx = np.zeros((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            if not opaque[y, x]:
                continue
            cur = np.clip(buf[y, x], 0.0, 255.0)
            k = int(np.argmin(((pal_f - cur) ** 2).sum(axis=1)))
            idx[y, x] = k
            err = cur - pal_f[k]
            if x + 1 < w and opaque[y, x + 1]:
                buf[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x - 1 >= 0 and opaque[y + 1, x - 1]:
                    buf[y + 1, x - 1] += err * (3.0 / 16.0)
                if opaque[y + 1, x]:
                    buf[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w and opaque[y + 1, x + 1]:
                    buf[y + 1, x + 1] += err * (1.0 / 16.0)
    return idx


# --------------------------------------------------------------------------- post-steps


def _pack(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.uint32)
    return (a[..., 0] << 24) | (a[..., 1] << 16) | (a[..., 2] << 8) | a[..., 3]


def cleanup(img: np.ndarray) -> np.ndarray:
    """Replace isolated interior pixels (differing from all 4 neighbours) by the majority.

    Single pass, every decision read from the input; ties go to N, W, E, S order.
    """
    _check_rgba(img)
    out = img.copy()
    h, w = img.shape[:2]
    if h < 3 or w < 3:
        return _out(out)
    packed = _pack(img)
    c = packed[1:-1, 1:-1]
    nbrs = [packed[:-2, 1:-1], packed[1:-1, :-2], packed[1:-1, 2:], packed[2:, 1:-1]]  # N W E S
    isolated = np.ones(c.shape, dtype=bool)
    for n in nbrs:
        isolated &= c != n
    counts = np.stack(
        [sum((nbrs[j] == nbrs[i]).astype(np.int64) for j in range(4)) for i in range(4)]
    )
    best = np.argmax(counts, axis=0)  # first of the tied maxima, in N, W, E, S order
    nbr_rgba = [img[:-2, 1:-1], img[1:-1, :-2], img[1:-1, 2:], img[2:, 1:-1]]
    chosen = np.choose(best[..., None], nbr_rgba)
    inner = out[1:-1, 1:-1]
    inner[isolated] = chosen[isolated]
    return _out(out)


def add_outline(img: np.ndarray, color) -> np.ndarray:
    """Outward 4-connected outline: transparent pixels beside an opaque one get ``color``."""
    _check_rgba(img)
    rgb = np.asarray(color)
    if rgb.shape != (3,):
        raise ValueError("color must be an RGB triple")
    out = img.copy()
    opaque = img[..., 3] >= ALPHA_THRESHOLD
    near = np.zeros_like(opaque)
    near[1:, :] |= opaque[:-1, :]
    near[:-1, :] |= opaque[1:, :]
    near[:, 1:] |= opaque[:, :-1]
    near[:, :-1] |= opaque[:, 1:]
    ring = near & ~opaque
    out[ring, :3] = rgb.astype(np.uint8)
    out[ring, 3] = 255
    return _out(out)


def apply_background(img: np.ndarray, spec: ArtSpec, palette: np.ndarray) -> np.ndarray:
    """Fill alpha-0 pixels with a palette colour when ``spec.background == "solid"``."""
    _check_rgba(img)
    _check_palette(palette)
    out = img.copy()
    if spec.background != "solid":
        return _out(out)
    i = spec.background_color_index
    if i is None:
        i = lightest_index(palette)
    n = palette.shape[0]
    if i < 0 or i >= n:
        raise SpecError(f"background_color_index {i} is out of range for a palette of {n} colors.")
    hole = out[..., 3] == 0
    out[hole, :3] = palette[i]
    out[hole, 3] = 255
    return _out(out)


# --------------------------------------------------------------------------- grid <-> rgba


def grid_to_rgba(grid: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Index grid (``-1`` = transparent) to RGBA via the palette."""
    _check_palette(palette)
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.dtype.kind not in "iu":
        raise ValueError("grid must be a 2-D integer numpy array")
    if grid.size == 0:
        raise ValueError("grid must not be empty")
    n = palette.shape[0]
    if grid.min() < -1 or grid.max() >= n:
        raise ValueError(f"grid values must be in -1..{n - 1}")
    out = np.zeros(grid.shape + (4,), dtype=np.uint8)
    opaque = grid >= 0
    out[opaque, :3] = palette[grid[opaque]]
    out[opaque, 3] = 255
    return _out(out)


def rgba_to_grid(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Inverse of ``grid_to_rgba``; off-palette opaque pixels raise ``ValueError``."""
    _check_rgba(img)
    _check_palette(palette)
    h, w = img.shape[:2]
    grid = np.full((h, w), -1, dtype=np.int64)
    opaque = img[..., 3] >= ALPHA_THRESHOLD
    if opaque.any():
        match = (img[opaque][:, None, :3] == palette[None, :, :]).all(axis=2)
        if not match.any(axis=1).all():
            raise ValueError("image contains an opaque pixel that is not in the palette")
        grid[opaque] = np.argmax(match, axis=1)  # first match: lowest index
    return grid


# --------------------------------------------------------------------------- composition

_cleanup_step = cleanup


def pixelate(
    img: np.ndarray, spec: ArtSpec, palette: np.ndarray, *, cleanup: bool | None = None
) -> np.ndarray:
    """The only sanctioned sequencing: downscale, quantize, cleanup, outline, background."""
    _check_palette(palette)
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            raise ValueError("img must be a uint8 numpy array")
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    out = downscale(img, spec.width, spec.height)
    out = quantize(out, palette, spec.dither)
    if cleanup is None:
        cleanup = spec.width > 32 or spec.height > 32
    if cleanup:
        out = _cleanup_step(out)
    if spec.outline:
        out = add_outline(out, palette[darkest_index(palette)])
    return apply_background(out, spec, palette)
