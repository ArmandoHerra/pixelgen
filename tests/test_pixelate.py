"""Tests for the deterministic pixelation pipeline (PXG-04)."""

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from pixelgen import pixelate as px
from pixelgen.errors import SpecError
from pixelgen.pixelate import (
    add_outline,
    apply_background,
    cleanup,
    darkest_index,
    downscale,
    grid_to_rgba,
    lightest_index,
    pixelate,
    quantize,
    rgba_to_grid,
)
from pixelgen.spec import ArtSpec

DITHERS = ("none", "floyd-steinberg", "bayer")
BW = np.array([(0, 0, 0), (255, 255, 255)], dtype=np.uint8)
CLEAR = np.array([0, 0, 0, 0], dtype=np.uint8)


def pal(*colors):
    return np.array(colors, dtype=np.uint8)


def gray_img(rows):
    """Opaque gray image from a 2-D list of gray levels."""
    g = np.array(rows, dtype=np.uint8)
    alpha = np.full(g.shape, 255, dtype=np.uint8)
    return np.stack([g, g, g, alpha], axis=-1)


def solid(h, w, rgba):
    img = np.empty((h, w, 4), dtype=np.uint8)
    img[:] = rgba
    return img


def make_spec(w=32, h=32, **kw):
    return ArtSpec(subject="x", width=w, height=h, **kw)


def random_rgba(rng, h, w, partial_alpha=True):
    img = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
    if not partial_alpha:
        img[..., 3] = 255
    return img


def random_palette(rng, n):
    return rng.integers(0, 256, size=(n, 3), dtype=np.uint8)


def gradient():
    return gray_img([[x * 255 // 63 for x in range(64)]] * 16)


GRAD_PAL = pal((0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255))


def assert_rgba(out, shape):
    assert out.dtype == np.uint8
    assert out.shape == shape
    assert out.flags["C_CONTIGUOUS"]


def test_array_contract():
    img = random_rgba(np.random.default_rng(1), 6, 5)
    p = random_palette(np.random.default_rng(2), 4)
    grid = np.zeros((6, 5), dtype=np.int64)
    spec = make_spec(8, 8)
    assert_rgba(downscale(img, 3, 2), (2, 3, 4))
    for d in DITHERS:
        assert_rgba(quantize(img, p, d), (6, 5, 4))
    assert_rgba(cleanup(img), (6, 5, 4))
    assert_rgba(add_outline(img, (1, 2, 3)), (6, 5, 4))
    assert_rgba(apply_background(img, spec, p), (6, 5, 4))
    assert_rgba(grid_to_rgba(grid, p), (6, 5, 4))
    assert_rgba(pixelate(img, spec, p), (8, 8, 4))

    with pytest.raises(ValueError):
        quantize(img.astype(np.float64), p)
    with pytest.raises(ValueError):
        quantize(img[..., :2], p)
    with pytest.raises(ValueError):
        quantize(img, np.zeros((3, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        quantize(img, np.zeros((0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        quantize(img, p, "atkinson")


def _case_downscale(img, p, spec):
    return downscale(img, 3, 2)


def _case_quantize(mode):
    return lambda img, p, spec: quantize(img, p, mode)


def _case_cleanup(img, p, spec):
    return cleanup(img)


def _case_outline(img, p, spec):
    return add_outline(img, (9, 9, 9))


def _case_background(img, p, spec):
    return apply_background(img, make_spec(8, 8, background="solid"), p)


def _case_pixelate(img, p, spec):
    return pixelate(img, make_spec(8, 8, background="solid"), p)


@pytest.mark.parametrize(
    "fn",
    [
        _case_downscale,
        _case_quantize("none"),
        _case_quantize("floyd-steinberg"),
        _case_quantize("bayer"),
        _case_cleanup,
        _case_outline,
        _case_background,
        _case_pixelate,
    ],
)
def test_inputs_not_mutated(fn):
    rng = np.random.default_rng(7)
    img = random_rgba(rng, 9, 9)
    p = random_palette(rng, 5)
    img_before, p_before = img.copy(), p.copy()
    out = fn(img, p, None)
    assert np.array_equal(img, img_before)
    assert np.array_equal(p, p_before)
    assert not np.shares_memory(out, img)
    assert not np.shares_memory(out, p)


def test_inputs_not_mutated_grid_to_rgba():
    p = random_palette(np.random.default_rng(3), 4)
    grid = np.array([[0, 1, -1], [2, 3, 0]], dtype=np.int64)
    g_before, p_before = grid.copy(), p.copy()
    out = grid_to_rgba(grid, p)
    assert np.array_equal(grid, g_before)
    assert np.array_equal(p, p_before)
    assert not np.shares_memory(out, grid)
    assert not np.shares_memory(out, p)


def test_downscale_nearest():
    img = np.zeros((4, 4, 4), dtype=np.uint8)
    for y in range(4):
        for x in range(4):
            img[y, x] = (y * 16 + x, y, x, 255)
    out = downscale(img, 2, 2)
    for oy, ox, (sy, sx) in [(0, 0, (1, 1)), (0, 1, (1, 3)), (1, 0, (3, 1)), (1, 1, (3, 3))]:
        assert np.array_equal(out[oy, ox], img[sy, sx])
    small = np.array([[[1, 2, 3, 77], [4, 5, 6, 255]], [[7, 8, 9, 0], [10, 11, 12, 128]]], np.uint8)
    up = downscale(small, 4, 4)
    for y in range(4):
        for x in range(4):
            assert np.array_equal(up[y, x], small[y // 2, x // 2])
    same = downscale(img, 4, 4)
    assert np.array_equal(same, img)
    assert not np.shares_memory(same, img)
    assert downscale(small, 1, 1).shape == (1, 1, 4)
    assert downscale(small, 2, 2)[0, 0, 3] == 77
    with pytest.raises(ValueError):
        downscale(img, 0, 2)


def test_quantize_none_nearest_and_ties():
    img = gray_img([[127, 128]])
    out = quantize(img, BW, "none")
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 255
    one = gray_img([[1]])
    a = quantize(one, pal((0, 0, 0), (2, 2, 2)), "none")
    b = quantize(one, pal((2, 2, 2), (0, 0, 0)), "none")
    assert tuple(a[0, 0]) == (0, 0, 0, 255)
    assert tuple(b[0, 0]) == (2, 2, 2, 255)


@pytest.mark.parametrize("mode", DITHERS)
def test_quantize_alpha_semantics(mode):
    img = np.array([[[200, 200, 200, 127], [200, 200, 200, 128]]], dtype=np.uint8)
    out = quantize(img, BW, mode)
    assert tuple(out[0, 0]) == (0, 0, 0, 0)
    assert out[0, 1, 3] == 255 and tuple(out[0, 1, :3]) in {(0, 0, 0), (255, 255, 255)}
    rng = np.random.default_rng(11)
    big = random_rgba(rng, 12, 12)
    out = quantize(big, GRAD_PAL, mode)
    assert set(np.unique(out[..., 3])) <= {0, 255}
    assert np.array_equal(out[..., 3] == 0, big[..., 3] < 128)


def test_bayer_pattern_exact():
    expected = {
        128: [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
        110: [[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]],
        132: [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]],
    }
    for level, mask in expected.items():
        out = quantize(gray_img([[level] * 4] * 4), BW, "bayer")
        assert (out[..., 0] == 255).astype(int).tolist() == mask, level
        assert np.all(out[..., 3] == 255)


def test_floyd_steinberg_hand_examples():
    a = gray_img([[100, 100]])
    fs = quantize(a, BW, "floyd-steinberg")
    assert [tuple(p) for p in fs[0]] == [(0, 0, 0, 255), (255, 255, 255, 255)]
    nn = quantize(a, BW, "none")
    assert [tuple(p) for p in nn[0]] == [(0, 0, 0, 255), (0, 0, 0, 255)]
    b = np.array([[[120, 120, 120, 255], [0, 0, 0, 0], [110, 110, 110, 255]]], dtype=np.uint8)
    out = quantize(b, BW, "floyd-steinberg")
    assert [tuple(p) for p in out[0]] == [(0, 0, 0, 255), (0, 0, 0, 0), (0, 0, 0, 255)]


def _in_palette_or_clear(out, p):
    opaque = out[..., 3] == 255
    assert np.all(out[~opaque] == 0)
    rows = {tuple(r) for r in p.tolist()}
    return all(tuple(px_) in rows for px_ in out[opaque][:, :3].tolist())


def test_quantize_palette_only_invariant():
    rng = np.random.default_rng(20261028)
    for case in range(210):
        h, w = int(rng.integers(1, 41)), int(rng.integers(1, 41))
        p = random_palette(rng, int(rng.integers(1, 65)))
        img = random_rgba(rng, h, w)
        for mode in DITHERS:
            out = quantize(img, p, mode)
            assert _in_palette_or_clear(out, p), f"case {case} mode {mode} {h}x{w}"


def _fraction_diff(a, b):
    return float((a != b).any(axis=-1).mean())


def test_gradient_dithers_pairwise_different():
    """Measured differing-pixel fractions: none/FS 0.2383, none/bayer 0.1875, FS/bayer 0.2949."""
    img = gradient()
    outs = {m: quantize(img, GRAD_PAL, m) for m in DITHERS}
    fractions = {
        (a, b): _fraction_diff(outs[a], outs[b])
        for a, b in [("none", "floyd-steinberg"), ("none", "bayer"), ("floyd-steinberg", "bayer")]
    }
    for pair, frac in fractions.items():
        assert frac >= 0.05, (pair, fractions)


PINNED_HASHES = {
    "none": "bd3c646fedb23a816d2269322588aa474296592c786347e455c92b2d37fad164",
    "floyd-steinberg": "150ad8827dd24cbb522565b49473d816e15bb607793a565c9694136c66d2dc42",
    "bayer": "6231a986508aa539d5d40d1c75846e0ea993721c952053fb92985afbd1628e65",
}


def test_determinism_and_no_rng():
    img = gradient()
    for mode in DITHERS:
        first = quantize(img, GRAD_PAL, mode)
        second = quantize(img, GRAD_PAL, mode)
        assert first.tobytes() == second.tobytes()
        assert hashlib.sha256(first.tobytes()).hexdigest() == PINNED_HASHES[mode], mode
    src = Path(px.__file__).read_text()
    hits = 0
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Name) and node.id == "random":
            hits += 1
        elif isinstance(node, ast.Attribute) and node.attr == "random":
            hits += 1
        elif isinstance(node, ast.Import) and any(a.name == "random" for a in node.names):
            hits += 1
        elif isinstance(node, ast.ImportFrom) and node.module == "random":
            hits += 1
    assert hits == 0


def test_luminance_helpers():
    assert darkest_index(pal((0, 0, 255), (90, 0, 0))) == 1
    assert darkest_index(pal((0, 100, 0), (0, 0, 200))) == 1
    assert darkest_index(pal((10, 10, 10), (0, 0, 0), (0, 0, 0))) == 1
    assert lightest_index(pal((255, 255, 255), (255, 255, 255))) == 0
    assert type(darkest_index(BW)) is int
    assert type(lightest_index(BW)) is int


def test_add_outline_outward_4_connected():
    img = np.zeros((5, 5, 4), dtype=np.uint8)
    img[2, 2] = (255, 0, 0, 255)
    out = add_outline(img, (1, 2, 3))
    assert tuple(out[2, 2]) == (255, 0, 0, 255)
    for y, x in [(1, 2), (3, 2), (2, 1), (2, 3)]:
        assert tuple(out[y, x]) == (1, 2, 3, 255)
    for y, x in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        assert tuple(out[y, x]) == (0, 0, 0, 0)
    corner = np.zeros((3, 3, 4), dtype=np.uint8)
    corner[0, 0] = (255, 0, 0, 255)
    out = add_outline(corner, (1, 2, 3))
    assert out.shape == (3, 3, 4)
    painted = {(y, x) for y in range(3) for x in range(3) if out[y, x, 3] == 255}
    assert painted == {(0, 0), (0, 1), (1, 0)}
    full = solid(4, 4, (5, 6, 7, 255))
    assert add_outline(full, (1, 2, 3)).tobytes() == full.tobytes()


def _px(c):
    return (*c, 255)


def test_cleanup_rule():
    A, B, C, D, E = (10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0), (50, 0, 0)
    base = np.zeros((3, 3, 4), dtype=np.uint8)
    base[:] = _px(A)
    # (a)
    img = base.copy()
    img[1, 1] = _px(B)
    assert tuple(cleanup(img)[1, 1]) == _px(A)
    # (b) a matching neighbour keeps the centre
    img = base.copy()
    img[1, 1] = _px(B)
    img[1, 2] = _px(B)
    assert tuple(cleanup(img)[1, 1]) == _px(B)
    # (c) all distinct -> N first; two-two tie -> N first
    img = base.copy()
    img[1, 1] = _px(B)
    img[0, 1], img[1, 0], img[1, 2], img[2, 1] = _px(A), _px(C), _px(D), _px(E)
    assert tuple(cleanup(img)[1, 1]) == _px(A)
    img[0, 1], img[1, 0], img[1, 2], img[2, 1] = _px(A), _px(C), _px(C), _px(A)
    assert tuple(cleanup(img)[1, 1]) == _px(A)
    # (d) border pixel untouched
    img = base.copy()
    img[0, 0] = _px(B)
    assert tuple(cleanup(img)[0, 0]) == _px(B)
    img = base.copy()
    img[0, 1] = _px(B)
    assert tuple(cleanup(img)[0, 1]) == _px(B)
    # (e) transparency counts as a value
    img = base.copy()
    img[1, 1] = CLEAR
    assert tuple(cleanup(img)[1, 1]) == _px(A)
    img = np.zeros((3, 3, 4), dtype=np.uint8)
    img[1, 1] = _px(A)
    assert tuple(cleanup(img)[1, 1]) == (0, 0, 0, 0)
    # (f) single pass: checkerboard flips exactly the 4 interior pixels
    cb = np.zeros((4, 4, 4), dtype=np.uint8)
    for y in range(4):
        for x in range(4):
            cb[y, x] = _px(A if (x + y) % 2 == 0 else B)
    out = cleanup(cb)
    changed = {
        (y, x) for y in range(4) for x in range(4) if not np.array_equal(out[y, x], cb[y, x])
    }
    assert changed == {(1, 1), (1, 2), (2, 1), (2, 2)}


def test_apply_background():
    p = pal((0, 0, 0), (10, 20, 30), (40, 50, 60))
    img = np.zeros((2, 2, 4), dtype=np.uint8)
    img[0, 0] = (1, 2, 3, 255)
    t = apply_background(
        img, make_spec(8, 8, background="transparent", background_color_index=2), p
    )
    assert t.tobytes() == img.tobytes()
    s = apply_background(img, make_spec(8, 8, background="solid", background_color_index=1), p)
    assert tuple(s[0, 0]) == (1, 2, 3, 255)
    for y, x in [(0, 1), (1, 0), (1, 1)]:
        assert tuple(s[y, x]) == (10, 20, 30, 255)
    p2 = pal((0, 0, 0), (200, 200, 200), (50, 50, 50))
    d = apply_background(img, make_spec(8, 8, background="solid"), p2)
    assert tuple(d[1, 1]) == (200, 200, 200, 255)
    with pytest.raises(SpecError) as exc:
        apply_background(img, make_spec(8, 8, background="solid", background_color_index=3), p)
    assert "3" in str(exc.value)
    neg = ArtSpec.model_construct(subject="x", background="solid", background_color_index=-1)
    with pytest.raises(SpecError) as exc:
        apply_background(img, neg, p)
    assert "-1" in str(exc.value)


def test_grid_roundtrip():
    rng = np.random.default_rng(5)
    p = random_palette(rng, 6)
    p[3] = p[1]  # a duplicate colour must not break the roundtrip of lowest indices
    g = rng.integers(-1, 6, size=(9, 11), dtype=np.int64)
    g[g == 3] = 1
    assert np.array_equal(rgba_to_grid(grid_to_rgba(g, p), p), g)
    for bad in (6, -2):
        g2 = g.copy()
        g2[0, 0] = bad
        with pytest.raises(ValueError):
            grid_to_rgba(g2, p)
    off = np.zeros((1, 1, 4), dtype=np.uint8)
    off[0, 0] = (1, 2, 3, 255)
    q = pal((9, 9, 9), (8, 8, 8))
    with pytest.raises(ValueError):
        rgba_to_grid(off, q)


def _speck(h, w, color):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[h // 2, w // 2] = color
    return img


SPECK_PAL = pal((0, 0, 0), (255, 0, 0), (255, 255, 255))


def test_pixelate_order_and_cleanup_default():
    red = (255, 0, 0, 255)
    plain = dict(outline=False, background="transparent")
    # (a) threshold is > 32, and explicit overrides win
    out = pixelate(_speck(32, 33, red), make_spec(33, 32, **plain), SPECK_PAL)
    assert out[..., 3].sum() == 0
    out = pixelate(_speck(32, 32, red), make_spec(32, 32, **plain), SPECK_PAL)
    assert out[..., 3].sum() == 255
    out = pixelate(_speck(64, 64, red), make_spec(64, 64, **plain), SPECK_PAL, cleanup=False)
    assert out[..., 3].sum() == 255
    out = pixelate(_speck(16, 16, red), make_spec(16, 16, **plain), SPECK_PAL, cleanup=True)
    assert out[..., 3].sum() == 0
    # (b) cleanup runs before the outline
    img = _speck(32, 33, red)
    out = pixelate(img, make_spec(33, 32, outline=True, background="transparent"), SPECK_PAL)
    assert out.shape == (32, 33, 4)
    assert np.all(out == 0)
    # (c) outline runs before the background fill
    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[10:18, 10:18] = red
    out = pixelate(img, make_spec(40, 40, outline=True, background="solid"), SPECK_PAL)
    ring = [(9, c) for c in range(10, 18)] + [(18, c) for c in range(10, 18)]
    ring += [(r, 9) for r in range(10, 18)] + [(r, 18) for r in range(10, 18)]
    for y, x in ring:
        assert tuple(out[y, x]) == (0, 0, 0, 255), (y, x)
    assert tuple(out[0, 0]) == (255, 255, 255, 255)
    assert tuple(out[12, 12]) == red


def test_pixelate_invariants():
    rng = np.random.default_rng(4242)
    for case in range(54):
        ih, iw = int(rng.integers(1, 201)), int(rng.integers(1, 201))
        w, h = int(rng.integers(8, 129)), int(rng.integers(8, 129))
        p = random_palette(rng, int(rng.integers(1, 17)))
        spec = make_spec(
            w, h, dither=DITHERS[case % 3], outline=bool(case % 2), background="transparent"
        )
        out = pixelate(random_rgba(rng, ih, iw), spec, p)
        assert out.shape == (h, w, 4), case
        assert _in_palette_or_clear(out, p), case
    # RGB input is fully opaque
    rgb = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    out = pixelate(rgb, make_spec(8, 8, outline=False), random_palette(rng, 4))
    assert np.all(out[..., 3] == 255)
    # LLM-grid path is the identity when nothing is added
    for dither in ("none", "floyd-steinberg"):
        for _ in range(6):
            w, h = int(rng.integers(8, 41)), int(rng.integers(8, 41))
            p = random_palette(rng, int(rng.integers(1, 17)))
            g = rng.integers(-1, len(p), size=(h, w), dtype=np.int64)
            spec = make_spec(w, h, dither=dither, outline=False)
            rgba = grid_to_rgba(g, p)
            assert pixelate(rgba, spec, p, cleanup=False).tobytes() == rgba.tobytes()
