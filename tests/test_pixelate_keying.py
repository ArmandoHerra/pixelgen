"""Border-connected backdrop keying for opaque raster base images (PXG-38).

``key_background`` and ``KEY_TOLERANCE`` are fetched with ``getattr`` inside each test so a
run against the base ``src/`` fails on an assertion, not on a collection ``ImportError``.
Fixtures are built in NumPy at test time; no binary is committed.
"""

import ast
import hashlib
from pathlib import Path

import aseprite_oracle
import numpy as np
import pytest
from fakes import FakeRasterGenerator
from PIL import Image
from test_cli_aseprite import Harness, _grid

import pixelgen.backends.base
import pixelgen.pixelate as px
from pixelgen.export import load_sidecar
from pixelgen.palettes import load_palette
from pixelgen.spec import ArtSpec

B = (41, 173, 255)  # pico8 index 12
S = (255, 0, 77)  # pico8 index 8
S2 = (0, 228, 54)  # pico8 index 11

# Recorded at $BASE (f8716d9, PXG-35) before the first edit; base value == after value.
_NO_KEY_HASH = "451e29ed760d84f82846259ec436b6b09c85f42ce8679d6c335a8d0bdc7dc93d"
FROZEN_NO_KEY = {
    "solid-opaque": _NO_KEY_HASH,
    "transparent-alpha254": _NO_KEY_HASH,
    "solid-alpha254": _NO_KEY_HASH,
}
# sha256 of each function's source lines at $BASE (interpreter-independent: ast.dump differs
# between CPython versions, source text does not).
FROZEN_AST = {
    "pixelate": "a5d4dd961810e492a55036e591146f175bba280a83f7c033f3750efad8592164",
    "apply_background": "9c5bf0eab892ee77dcd5e37a2d96ba921a5958985299d0862162bc1d50afe001",
}


def _key():
    fn = getattr(px, "key_background", None)
    assert fn is not None, "pixelgen.pixelate.key_background is missing"
    return fn


def _tol():
    val = getattr(px, "KEY_TOLERANCE", None)
    assert val is not None, "pixelgen.pixelate.KEY_TOLERANCE is missing"
    return val


def _field(side, colour=B):
    img = np.empty((side, side, 4), dtype=np.uint8)
    img[:] = (*colour, 255)
    return img


def _keyed(out):
    return (out[..., 3] == 0) & (out[..., :3] == 0).all(axis=2)


def _spec(**kw):
    kw.setdefault("outline", False)
    kw.setdefault("background", "transparent")
    kw.setdefault("dither", "none")
    return ArtSpec(subject="x", width=16, height=16, palette="pico8", **kw)


def _source(channels=3):
    """256x256 B with an S square at [64:192] and a B hole at [112:144]."""
    img = _field(256)
    img[64:192, 64:192] = (*S, 255)
    img[112:144, 112:144] = (*B, 255)
    return img if channels == 4 else img[..., :3].copy()


# --------------------------------------------------------------------------- AC1-AC3


def test_key_flat_backdrop():
    key = _key()
    img = _field(96)
    img[24:72, 24:72] = (*S, 255)
    before = img.copy()
    out = key(img)
    inside = np.zeros((96, 96), dtype=bool)
    inside[24:72, 24:72] = True
    assert int(_keyed(out).sum()) == 6912
    assert (_keyed(out) == ~inside).all()
    assert out[inside].tobytes() == img[inside].tobytes()
    assert out.dtype == np.uint8 and out.shape == img.shape and out.flags["C_CONTIGUOUS"]
    assert img.tobytes() == before.tobytes()
    assert not np.shares_memory(out, img)


@pytest.mark.parametrize("shape", ["hole", "diamond"])
def test_key_enclosed_same_colour_survives(shape):
    key = _key()
    img = _field(96)
    if shape == "hole":
        img[24:72, 24:72] = (*S, 255)
        img[40:56, 40:56] = (*B, 255)
        out = key(img)
        assert int(_keyed(out).sum()) == 6912
        assert not _keyed(out)[40:56, 40:56].any()
        assert out[40:56, 40:56].tobytes() == img[40:56, 40:56].tobytes()
    else:
        yy, xx = np.mgrid[0:96, 0:96]
        dist = np.abs(xx - 48) + np.abs(yy - 48)
        img[dist == 20] = (*S, 255)
        out = key(img)
        assert int((dist < 20).sum()) == 761
        assert not _keyed(out)[dist < 20].any()
        assert (_keyed(out) == (dist > 20)).all()
        assert not _keyed(out)[dist == 20].any()


def test_key_noisy_gradient_backdrop():
    key = _key()
    rng = np.random.default_rng(38)
    img = _field(128)
    img[..., 0] = img[..., 0] + ((np.arange(128) * 10) // 127).astype(np.uint8)[None, :]
    noise = rng.integers(-6, 7, size=(128, 128, 3))
    img[..., :3] = np.clip(img[..., :3].astype(np.int64) + noise, 0, 255).astype(np.uint8)
    img[40:88, 40:88] = (*S, 255)
    out = key(img)
    inside = np.zeros((128, 128), dtype=bool)
    inside[40:88, 40:88] = True
    assert int(_keyed(out).sum()) == 14080
    assert (_keyed(out) == ~inside).all()
    assert out[inside].tobytes() == img[inside].tobytes()


# --------------------------------------------------------------------------- AC4


def test_key_tolerance_boundary():
    key = _key()
    assert _tol() == 24
    img = _field(32)
    img[0, 10, 0] = B[0] + 24
    img[0, 20, 0] = B[0] + 25
    out = key(img)
    assert _keyed(out)[0, 10] and not _keyed(out)[0, 20]
    assert int(_keyed(out).sum()) == 32 * 32 - 1
    strict = key(img, tolerance=0)
    assert not _keyed(strict)[0, 10] and not _keyed(strict)[0, 20]
    assert int(_keyed(strict).sum()) == 32 * 32 - 2


@pytest.mark.parametrize("bad", [-1, 256, True, 1.5], ids=["-1", "256", "true", "float"])
def test_key_rejects_bad_tolerance(bad):
    key = _key()
    with pytest.raises(ValueError):
        key(_field(8), tolerance=bad)


# --------------------------------------------------------------------------- AC5


def test_key_subject_touches_border():
    key = _key()
    img = _field(64)
    img[0:20, 20:40] = (*S, 255)
    img[0:8, 0:8] = (*S2, 255)
    out = key(img)
    subject = np.zeros((64, 64), dtype=bool)
    subject[0:20, 20:40] = True
    subject[0:8, 0:8] = True
    assert int(subject.sum()) == 464
    assert int(_keyed(out).sum()) == 3632
    assert (_keyed(out) == ~subject).all()
    assert out[subject].tobytes() == img[subject].tobytes()


def test_key_no_reference_is_noop():
    key = _key()
    img = _field(64)
    img[0, 0] = (255, 0, 77, 255)
    img[0, -1] = (0, 228, 54, 255)
    img[-1, 0] = (255, 236, 39, 255)
    img[-1, -1] = (131, 118, 156, 255)
    out = key(img)
    assert hashlib.sha256(out.tobytes()).hexdigest() == hashlib.sha256(img.tobytes()).hexdigest()
    assert not np.shares_memory(out, img)


# --------------------------------------------------------------------------- AC6


def test_key_serpentine_512():
    key = _key()
    n = 512
    img = _field(n, S)
    expect = np.zeros((n, n), dtype=bool)

    def paint(r0, r1, c0, c1):
        img[r0:r1, c0:c1] = (*B, 255)
        expect[r0:r1, c0:c1] = True

    for r0, r1, c0, c1 in ((0, 2, 0, n), (n - 2, n, 0, n), (0, n, 0, 2), (0, n, n - 2, n)):
        paint(r0, r1, c0, c1)
    paint(2, 6, 100, 104)  # the one gap in the S frame
    bands = (506 - 6) // 8
    for k in range(bands):
        top = 6 + 8 * k
        paint(top, top + 4, 6, 500)
        if k < bands - 1:
            paint(top + 4, top + 8, *((6, 10) if k % 2 else (496, 500)))
    pocket = (slice(500, 504), slice(200, 204))
    img[pocket] = (*B, 255)  # sealed B pocket inside S
    out = key(img)
    assert (_keyed(out) == expect).all()
    assert not _keyed(out)[pocket].any()


def test_pixelate_module_has_no_recursion_or_new_imports():
    tree = ast.parse(Path(px.__file__).read_text(encoding="utf-8"))
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for call in (c for c in ast.walk(fn) if isinstance(c, ast.Call)):
            assert not (isinstance(call.func, ast.Name) and call.func.id == fn.name), fn.name
    banned = {"scipy", "skimage", "cv2"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        assert not banned & {m.split(".")[0] for m in mods}, mods


# --------------------------------------------------------------------------- AC7-AC8


def _install_spy(monkeypatch):
    calls = []
    real = getattr(px, "key_background", None)

    def spy(img, **kw):
        calls.append(img.shape)
        assert real is not None
        return real(img, **kw)

    monkeypatch.setattr(px, "key_background", spy, raising=False)
    return calls


@pytest.mark.parametrize("channels", [3, 4], ids=["rgb", "rgba255"])
def test_pixelate_raster_keys_source(monkeypatch, channels):
    pal = load_palette("pico8")
    calls = _install_spy(monkeypatch)
    out = px.pixelate_raster(_source(channels), _spec(), pal)
    assert calls == [(256, 256, 4)]
    clear = out[..., 3] == 0
    inside = np.zeros((16, 16), dtype=bool)
    inside[4:12, 4:12] = True
    assert int(clear.sum()) == 192
    assert (clear == ~inside).all()
    assert (out[inside][:, 3] == 255).all()
    assert (out[7:9, 7:9, :3] == np.array(B, dtype=np.uint8)).all()
    outlined = px.pixelate_raster(_source(channels), _spec(outline=True), pal)
    assert outlined.tobytes() == out.tobytes()
    assert int((outlined[..., 3] == 0).sum()) == 192


@pytest.mark.parametrize(
    "arm,background,alpha",
    [
        ("solid-opaque", "solid", 255),
        ("transparent-alpha254", "transparent", 254),
        ("solid-alpha254", "solid", 254),
    ],
)
def test_pixelate_raster_no_key(monkeypatch, arm, background, alpha):
    def boom(*a, **kw):
        raise AssertionError("key_background must not run for this arm")

    monkeypatch.setattr(px, "key_background", boom, raising=False)
    src = _source(4)
    src[0, 0, 3] = alpha
    out = px.pixelate_raster(src, _spec(background=background), load_palette("pico8"))
    assert hashlib.sha256(out.tobytes()).hexdigest() == FROZEN_NO_KEY[arm]


# --------------------------------------------------------------------------- AC9


def test_llm_grid_never_keys(monkeypatch, tmp_path):
    calls = _install_spy(monkeypatch)
    h = Harness(monkeypatch, tmp_path)
    result = h.generate(["-o", str(h.out / "x")], {}, {"grid": _grid(1)})
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert h.llm.call_count == 2
    assert calls == []


def test_pixelate_function_ast_frozen():
    source = Path(px.__file__).read_text(encoding="utf-8")
    lines = source.splitlines()
    got = {
        n.name: hashlib.sha256("\n".join(lines[n.lineno - 1 : n.end_lineno]).encode()).hexdigest()
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in FROZEN_AST
    }
    assert got == FROZEN_AST


# --------------------------------------------------------------------------- AC10

AC10_ARMS = {
    "default": ([], 160),
    "outline": (["--outline"], 160),
    "no-outline": (["--no-outline"], 192),
    "solid": (["--background", "solid"], 0),
}


@pytest.mark.parametrize("arm", list(AC10_ARMS))
def test_cli_raster_keyed_png_matches_aseprite(monkeypatch, tmp_path, arm):
    flags, want_clear = AC10_ARMS[arm]
    calls = _install_spy(monkeypatch)
    monkeypatch.setitem(
        pixelgen.backends.base._REGISTRY, "fake-raster", "fakes:FakeRasterGenerator"
    )
    monkeypatch.setattr(FakeRasterGenerator, "log", [])
    monkeypatch.setattr(FakeRasterGenerator, "image_factory", lambda spec: _source(3))
    h = Harness(monkeypatch, tmp_path)
    args = [
        "generate", "a red apple", "--backend", "fake-raster", "--size", "16",
        "--palette", "pico8", "--aseprite", "-o", str(h.out / "x"), *flags,
    ]  # fmt: skip
    result = h.invoke(args, {})
    assert result.exit_code == 0, (result.stdout, result.stderr)
    png = np.array(Image.open(h.out / "x.png").convert("RGBA"))
    assert int((png[..., 3] == 0).sum()) == want_clear
    assert aseprite_oracle.main([str(h.out / "x.aseprite"), "--png", str(h.out / "x.png")]) == 0
    grid = np.array(load_sidecar(h.out / "x.json").grid)
    if arm == "solid":
        assert int((grid == -1).sum()) == 0
        assert calls == []
    else:
        inside = np.zeros((16, 16), dtype=bool)
        inside[4:12, 4:12] = True
        assert int((grid == -1).sum()) == 192
        assert ((grid == -1) == ~inside).all()
        assert len(calls) == 1
