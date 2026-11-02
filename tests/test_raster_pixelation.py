"""Unit tests for the raster pixelation path (PXG-35, AC1-AC7).

``block_downscale`` and ``pixelate_raster`` are fetched inside each test body, so a run
against the base ``src/`` collects and goes red by assertion, not by ImportError.
Expected values are written out here from the spec's D2 edge formula, never imported.
"""

import ast
import dataclasses
import itertools
import time
from pathlib import Path

import numpy as np
import pytest
from fakes import raster_gradient, raster_noise

import pixelgen.pixelate as pixelate_mod
from pixelgen.backends.base import GenerationResult, GenerationUsage
from pixelgen.palettes import load_palette
from pixelgen.pixelate import downscale, grid_to_rgba, rgba_to_grid
from pixelgen.spec import ArtSpec

DITHERS = ("none", "floyd-steinberg", "bayer")


def _fetch(name):
    fn = getattr(pixelate_mod, name, None)
    assert fn is not None, f"pixelgen.pixelate.{name} does not exist"
    return fn


def make_spec(w=32, h=32, **kw):
    return ArtSpec(subject="x", width=w, height=h, **kw)


def assert_palette_exact(img, palette, h, w):
    assert img.dtype == np.uint8
    assert img.shape == (h, w, 4)
    assert set(np.unique(img[..., 3]).tolist()) <= {0, 255}
    clear = img[..., 3] == 0
    assert not img[clear].any(), "alpha-0 pixels must be (0, 0, 0, 0)"
    rebuilt = grid_to_rgba(rgba_to_grid(img, palette), palette)
    assert rebuilt.tobytes() == img.tobytes()


# --------------------------------------------------------------------------- AC1


def test_generation_result_exactly_one_payload():
    assert "image" in {f.name for f in dataclasses.fields(GenerationResult)}
    usage = GenerationUsage()
    GenerationResult(grid=[[0]], generated_size=(1, 1), usage=usage)
    GenerationResult(image=np.zeros((4, 4, 3), np.uint8), generated_size=(4, 4), usage=usage)
    GenerationResult(image=np.zeros((4, 4, 4), np.uint8), generated_size=(4, 4), usage=usage)
    with pytest.raises(ValueError, match="exactly one of grid or image"):
        GenerationResult(
            grid=[[0]], image=np.zeros((4, 4, 3), np.uint8), generated_size=(4, 4), usage=usage
        )
    with pytest.raises(ValueError, match="exactly one of grid or image"):
        GenerationResult(generated_size=(4, 4), usage=usage)
    for bad in (
        np.zeros((4, 4, 3), np.float64),
        np.zeros((4, 4), np.uint8),
        np.zeros((4, 4, 2), np.uint8),
        np.zeros((0, 4, 3), np.uint8),
    ):
        with pytest.raises(ValueError, match="uint8 array of shape"):
            GenerationResult(image=bad, generated_size=(4, 4), usage=usage)


# --------------------------------------------------------------------------- AC2


def _axis_keys(src, n):
    """(source index -> key, output index -> expected key) from the D2 edge formula."""
    if src >= n:
        key_of_src = [None] * src
        for i in range(n):
            lo = (i * src) // n
            hi = max(lo + 1, ((i + 1) * src) // n)
            for s in range(lo, hi):
                key_of_src[s] = i
        assert None not in key_of_src
        return key_of_src, list(range(n))
    return list(range(src)), [(i * src) // n for i in range(n)]


@pytest.mark.parametrize(
    "src_w,src_h,w,h",
    [
        (512, 512, 32, 32),
        (500, 300, 32, 32),
        (17, 9, 5, 4),
        (512, 8, 32, 32),
        (8, 8, 32, 32),
        (32, 32, 32, 32),
    ],
)
def test_block_downscale_partition_and_edges(src_w, src_h, w, h):
    block_downscale = _fetch("block_downscale")
    ky_src, ky_out = _axis_keys(src_h, h)
    kx_src, kx_out = _axis_keys(src_w, w)

    def colour(ky, kx):
        return (ky * 8, kx * 8, 77, 255)

    img = np.zeros((src_h, src_w, 4), np.uint8)
    for y in range(src_h):
        for x in range(src_w):
            img[y, x] = colour(ky_src[y], kx_src[x])
    out = block_downscale(img, w, h)
    assert out.shape == (h, w, 4)
    for i in range(h):
        for j in range(w):
            assert tuple(out[i, j]) == colour(ky_out[i], kx_out[j]), (i, j)


def test_block_downscale_partition_and_edges_equal_size_alpha():
    block_downscale = _fetch("block_downscale")
    img = raster_noise(3, 32, 32, 4)
    out = block_downscale(img, 32, 32)
    opaque = img[..., 3] >= 128
    assert (out[opaque][:, :3] == img[opaque][:, :3]).all()
    assert (out[opaque][:, 3] == 255).all()
    assert not out[~opaque].any()


# --------------------------------------------------------------------------- AC3


def test_block_downscale_votes_majority_not_centre():
    img = np.empty((512, 512, 4), np.uint8)
    img[:] = (200, 30, 30, 255)
    img[8::16, 8::16] = (30, 30, 200, 255)
    control = downscale(img, 32, 32)
    assert (control == (30, 30, 200, 255)).all(axis=2).sum() == 1024
    block_downscale = _fetch("block_downscale")
    out = block_downscale(img, 32, 32)
    assert (out == (200, 30, 30, 255)).all(axis=2).sum() == 1024


# --------------------------------------------------------------------------- AC4


def _block(pixels):
    arr = np.array(pixels, np.uint8).reshape(4, 4, 4)
    return arr


def test_block_downscale_binned_vote_ties_and_alpha():
    block_downscale = _fetch("block_downscale")
    one = lambda px: tuple(block_downscale(_block(px), 1, 1)[0, 0])  # noqa: E731

    # (a) a 9-pixel bin beats 7 exact-colour pixels; the block takes the bin's mean.
    px = [(100, 100, 100, 255)] * 7 + [(192 + k, 50, 50, 255) for k in range(9)]
    assert one(px) == (196, 50, 50, 255)
    # (b) a tie between bins goes to the lower bin key.
    px = [(32, 32, 32, 255)] * 8 + [(16, 16, 16, 255)] * 8
    assert one(px) == (16, 16, 16, 255)
    # (c) transparent needs strictly more votes; alpha 127 is transparent, 128 opaque.
    red, clear = (255, 0, 0, 255), (9, 9, 9, 0)
    assert one([clear] * 9 + [red] * 7) == (0, 0, 0, 0)
    assert one([clear] * 8 + [red] * 8) == (255, 0, 0, 255)
    assert one([(255, 0, 0, 127)] * 16) == (0, 0, 0, 0)
    assert one([(255, 0, 0, 128)] * 16) == (255, 0, 0, 255)
    # (d) half-transparent raster
    big = np.zeros((512, 512, 4), np.uint8)
    big[:, 256:] = (255, 0, 0, 255)
    out = block_downscale(big, 32, 32)
    assert (out[:, :16] == 0).all()
    assert (out[:, 16:] == (255, 0, 0, 255)).all()


# --------------------------------------------------------------------------- AC5


def test_pixelate_raster_palette_exact_invariant():
    pixelate_raster = _fetch("pixelate_raster")
    import pixelgen.export as export_mod

    inputs = [
        raster_noise(20261101),
        raster_noise(20261101, channels=4),
        raster_gradient(),
    ]
    for pal_name in ("pico8", "gameboy"):
        palette = load_palette(pal_name)
        for w, h in ((32, 32), (16, 16), (128, 128), (33, 17)):
            for dither in DITHERS:
                for outline, background in itertools.product(
                    (True, False), ("transparent", "solid")
                ):
                    spec = make_spec(
                        w,
                        h,
                        dither=dither,
                        outline=outline,
                        background=background,
                        palette=pal_name,
                    )
                    for img in inputs:
                        out = pixelate_raster(img, spec, palette)
                        assert_palette_exact(out, palette, h, w)
                        grid = rgba_to_grid(out, palette).tolist()
                        export_mod._aseprite_indices(out, grid, palette)


# --------------------------------------------------------------------------- AC6


def test_pixelate_raster_gradient_dithers_pairwise_different():
    """Fractions of differing pixels, 32x32 grayscale4 gradient (drafting prototype):
    none/floyd-steinberg 0.2354, none/bayer 0.1719, floyd-steinberg/bayer 0.2744."""
    pixelate_raster = _fetch("pixelate_raster")
    palette = load_palette("grayscale4")
    outs = {
        d: pixelate_raster(
            raster_gradient(), make_spec(dither=d, outline=False, background="solid"), palette
        )
        for d in DITHERS
    }
    for a, b in (("none", "floyd-steinberg"), ("none", "bayer"), ("floyd-steinberg", "bayer")):
        frac = float((outs[a] != outs[b]).any(axis=2).mean())
        assert frac >= 0.05, (a, b, frac)


# --------------------------------------------------------------------------- AC7


def test_raster_path_pure_and_deterministic():
    block_downscale = _fetch("block_downscale")
    pixelate_raster = _fetch("pixelate_raster")
    palette = load_palette("pico8")
    img = raster_noise(7, 256, 256, 4)
    before = img.copy()
    bd = block_downscale(img, 32, 32)
    pr = pixelate_raster(img, make_spec(dither="bayer"), palette)
    assert img.tobytes() == before.tobytes()
    assert not np.shares_memory(bd, img)
    assert not np.shares_memory(pr, img)
    assert block_downscale(img, 32, 32).tobytes() == bd.tobytes()
    assert pixelate_raster(img, make_spec(dither="bayer"), palette).tobytes() == pr.tobytes()

    tree = ast.parse(Path(pixelate_mod.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "random" not in names

    big = raster_noise(7, 1024, 1024)
    for dither in DITHERS:
        start = time.perf_counter()
        pixelate_raster(big, make_spec(128, 128, dither=dither), palette)
        elapsed = time.perf_counter() - start
        assert elapsed < 10, f"{dither}: {elapsed:.2f}s"
