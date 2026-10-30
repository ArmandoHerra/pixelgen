"""PXG-27: `.aseprite` joins the atomic export set and the sidecar (no schema bump)."""

import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from aseprite_oracle import parse_aseprite, to_sidecar_grid

import pixelgen.export as export_mod
from pixelgen import palettes
from pixelgen.errors import AsepriteError, OutputExistsError, SpecError
from pixelgen.export import TokenUsage, export, load_last, load_sidecar, output_paths
from pixelgen.spec import ArtSpec

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PINNED = {
    "knight": "f6bea8c175cf0e0864074c796861f939cb3ef804198c3d1f656d44462e180741",
    "knight-v2": "db464c3520299860c370f3bff9d243b83adcf13babdb4c570f6985edd1938a00",
    "red-apple": "970bb21ff844ef90dae3dc6b009c398636c28acb3d65d6f4601c6652f9be0527",
    "slime": "61ec7d33277f6f5c655f10e30c74e6bf02fe35b29e4f18f2a033a5b1b89249c6",
}
SMALL = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)


def rgba_from_grid(grid, palette: np.ndarray) -> np.ndarray:
    cells = np.asarray(grid, dtype=np.int64)
    out = np.zeros((*cells.shape, 4), dtype=np.uint8)
    mask = cells >= 0
    out[mask, :3] = palette[cells[mask]]
    out[mask, 3] = 255
    return out


def seeded_grid(h: int, w: int, n: int, seed: int = 3) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    cells = rng.integers(-1, n, size=(h, w))
    flat = cells.reshape(-1)
    flat[0], flat[1], flat[2] = -1, 0, n - 1
    return cells.tolist()


def make_spec(w: int = 8, h: int = 8, palette: str = "pico8") -> ArtSpec:
    return ArtSpec(subject="campfire", width=w, height=h, palette=palette, seed=7)


def do_export(output: Path, *, grid=None, palette=SMALL, spec=None, **kw):
    spec = spec or make_spec()
    grid = grid if grid is not None else seeded_grid(spec.height, spec.width, len(palette))
    args = {
        "spec": spec,
        "grid": grid,
        "palette_colors": palette,
        "model": "claude-test",
        "backend": "llm-grid",
        "token_usage": TokenUsage(input_tokens=1, output_tokens=2, calls=1),
        "scale": 4,
        "now": NOW,
    }
    args.update(kw)
    return export(output, rgba_from_grid(grid, palette), **args)


def no_tmp(directory: Path) -> bool:
    return not list(Path(directory).glob("*.tmp")) and not list(Path(directory).glob(".*.tmp"))


# --- AC1, AC2 -----------------------------------------------------------------------------


@pytest.mark.parametrize("given", ["out/campfire", "out/campfire.png", "out/a.b", "x.PNG"])
def test_output_paths_default_is_three_tuple(given):
    default = output_paths(Path(given), 8)
    assert default == output_paths(Path(given), 8, aseprite=False)
    assert len(default) == 3
    name = Path(given).name
    stem = name[:-4] if name.endswith(".png") else name
    assert [p.name for p in default] == [f"{stem}.png", f"{stem}@8x.png", f"{stem}.json"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("out/knight", "knight.aseprite"),
        ("knight.png", "knight.aseprite"),
        ("Knight.PNG", "Knight.PNG.aseprite"),
        ("k.aseprite", "k.aseprite.aseprite"),
        ("dir with space/ñ", "ñ.aseprite"),
    ],
)
def test_output_paths_aseprite_naming(given, expected):
    paths = output_paths(Path(given), 8, aseprite=True)
    assert len(paths) == 4
    assert all(p.is_absolute() for p in paths)
    assert paths[:3] == output_paths(Path(given), 8)
    assert paths[3].name == expected
    assert paths[3].parent == paths[0].parent


# --- AC3 ----------------------------------------------------------------------------------

ROUND_TRIP = [(size, name) for size in (8, 32, 128) for name in palettes.list_palettes()]


@pytest.mark.parametrize(("size", "name"), ROUND_TRIP, ids=[f"{s}-{n}" for s, n in ROUND_TRIP])
def test_export_aseprite_round_trip(tmp_path, size, name):
    palette = palettes.load_palette(name)
    n = len(palette)
    spec = make_spec(size, size, name)
    grid = seeded_grid(size, size, n, seed=size)
    result = do_export(tmp_path / "rt", grid=grid, palette=palette, spec=spec, aseprite=True)
    assert result.aseprite_path == tmp_path / "rt.aseprite"
    assert result.aseprite_path.exists()
    ase = parse_aseprite(result.aseprite_path.read_bytes())
    side = load_sidecar(result.sidecar_path)
    assert to_sidecar_grid(ase) == side.grid
    assert [f"{r:02x}{g:02x}{b:02x}" for r, g, b, _ in ase.palette[:n]] == side.palette.colors
    assert ase.transparent_index == n


# --- AC4, AC5, AC6 ------------------------------------------------------------------------


def test_export_aseprite_off_writes_no_file(tmp_path):
    result = do_export(tmp_path / "s")
    assert result.aseprite_path is None
    assert not list(tmp_path.glob("*.aseprite"))
    data = json.loads(result.sidecar_path.read_text())
    assert list(data["outputs"]) == ["png", "preview", "scale"]


def test_sidecar_and_last_record_aseprite(tmp_path):
    result = do_export(tmp_path / "s", aseprite=True)
    data = json.loads(result.sidecar_path.read_text())
    assert list(data["outputs"]) == ["png", "preview", "scale", "aseprite"]
    assert data["outputs"]["aseprite"] == "s.aseprite"
    assert data["schema_version"] == 1
    assert load_sidecar(result.sidecar_path).outputs.aseprite == "s.aseprite"
    wrapper = json.loads(result.last_path.read_text())
    assert wrapper["sidecar"]["outputs"]["aseprite"] == "s.aseprite"
    assert load_last().sidecar.outputs.aseprite == "s.aseprite"


@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", ""])
def test_load_sidecar_aseprite_name(tmp_path, bad):
    result = do_export(tmp_path / "s", aseprite=True)
    data = json.loads(result.sidecar_path.read_text())
    data["outputs"]["aseprite"] = bad
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(SpecError) as info:
        load_sidecar(path)
    message = str(info.value)
    assert "\n" not in message
    assert str(path) in message
    assert "outputs.aseprite" in message


def test_load_sidecar_aseprite_null_loads_and_is_omitted(tmp_path):
    result = do_export(tmp_path / "s")
    data = json.loads(result.sidecar_path.read_text())
    data["outputs"]["aseprite"] = None
    path = tmp_path / "null.json"
    path.write_text(json.dumps(data))
    side = load_sidecar(path)
    assert side.outputs.aseprite is None
    assert "aseprite" not in json.loads(side.model_dump_json())["outputs"]


# --- AC7, AC9 -----------------------------------------------------------------------------


def test_aseprite_clobber_refused(tmp_path):
    stray = tmp_path / "s.aseprite"
    stray.write_bytes(b"SENTINEL")
    with pytest.raises(OutputExistsError) as info:
        do_export(tmp_path / "s", aseprite=True, overwrite=False)
    assert str(stray) in str(info.value)
    for name in ("s.png", "s@4x.png", "s.json"):
        assert not (tmp_path / name).exists()
    assert stray.read_bytes() == b"SENTINEL"
    assert not (export_mod.pixelgen_home() / "last.json").exists()
    assert no_tmp(tmp_path)


def test_flag_off_ignores_stray_aseprite(tmp_path):
    stray = tmp_path / "s.aseprite"
    stray.write_bytes(b"SENTINEL")
    result = do_export(tmp_path / "s", aseprite=False, overwrite=False)
    assert stray.read_bytes() == b"SENTINEL"
    assert result.aseprite_path is None
    assert "aseprite" not in json.loads(result.sidecar_path.read_text())["outputs"]


# --- AC8 ----------------------------------------------------------------------------------


def test_aseprite_overwrite_replaces_all(tmp_path):
    old_umask = os.umask(0o077)
    try:
        first = do_export(tmp_path / "s", aseprite=True)
        four = [first.png_path, first.preview_path, first.sidecar_path, first.aseprite_path]
        inodes = {}
        for path in four:
            os.chmod(path, 0o600)
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            inodes[path] = path.stat().st_ino
        new_grid = seeded_grid(8, 8, len(SMALL), seed=99)
        second = do_export(tmp_path / "s", grid=new_grid, aseprite=True, overwrite=True)
        for path in four:
            assert stat.S_IMODE(path.stat().st_mode) == 0o644, path
            assert path.stat().st_ino != inodes[path], path
        side = load_sidecar(second.sidecar_path)
        assert side.grid == new_grid
        assert to_sidecar_grid(parse_aseprite(second.aseprite_path.read_bytes())) == side.grid
    finally:
        os.umask(old_umask)


# --- AC10 ---------------------------------------------------------------------------------


def _replay(sidecar_path: Path, out: Path, **kw):
    side = load_sidecar(sidecar_path)
    palette = np.array(
        [[int(c[i : i + 2], 16) for i in (0, 2, 4)] for c in side.palette.colors], dtype=np.uint8
    )
    result = export(
        out,
        rgba_from_grid(side.grid, palette),
        spec=side.spec,
        grid=side.grid,
        palette_colors=palette,
        model=side.model,
        backend=side.backend,
        token_usage=side.token_usage,
        scale=side.outputs.scale,
        now=side.created_at,
        **kw,
    )
    return side, result


@pytest.mark.parametrize("stem", sorted(PINNED))
def test_flag_off_bytes_frozen(tmp_path, stem):
    committed_path = next(EXAMPLES.glob(f"*/{stem}.json"))
    committed = committed_path.read_bytes()
    assert hashlib.sha256(committed).hexdigest() == PINNED[stem]
    _, result = _replay(committed_path, tmp_path / stem / stem)
    produced = result.sidecar_path.read_bytes()
    assert hashlib.sha256(produced).hexdigest() == PINNED[stem]
    assert produced == committed
    expected_last = (
        json.dumps(
            {
                "schema_version": 1,
                "sidecar_path": str(result.sidecar_path),
                "sidecar": json.loads(committed),
            },
            indent=2,
        )
        + "\n"
    ).encode()
    assert result.last_path.read_bytes() == expected_last


# --- AC11, AC12 ---------------------------------------------------------------------------


def test_examples_replay_through_oracle(tmp_path):
    paths = sorted(p for p in EXAMPLES.glob("*/*.json") if p.name != "run.json")
    assert len(paths) == 4
    for committed_path in paths:
        slug = committed_path.parent.name
        stem = committed_path.stem
        side, result = _replay(committed_path, tmp_path / slug / stem, aseprite=True)
        for produced in (
            result.png_path,
            result.preview_path,
            result.sidecar_path,
            result.aseprite_path,
        ):
            assert produced.resolve().is_relative_to(tmp_path.resolve()), produced
        ase = parse_aseprite(result.aseprite_path.read_bytes())
        assert to_sidecar_grid(ase) == side.grid
        n = len(side.palette.colors)
        assert [f"{r:02x}{g:02x}{b:02x}" for r, g, b, _ in ase.palette[:n]] == side.palette.colors
        expected = json.loads(committed_path.read_text())
        expected["outputs"]["aseprite"] = f"{stem}.aseprite"
        assert json.loads(result.sidecar_path.read_text()) == expected


# --- AC13 ---------------------------------------------------------------------------------


def test_aseprite_palette_limit(tmp_path):
    big = np.stack([np.arange(256) % 256, np.arange(256) // 2, np.zeros(256)], axis=1).astype(
        np.uint8
    )
    grid = seeded_grid(8, 8, 256)
    out = tmp_path / "newdir" / "s"
    with pytest.raises(AsepriteError) as info:
        do_export(out, grid=grid, palette=big, aseprite=True)
    assert "\n" not in str(info.value)
    assert not out.parent.exists()
    assert not (export_mod.pixelgen_home() / "last.json").exists()
    assert no_tmp(tmp_path)

    ok = do_export(tmp_path / "off" / "s", grid=grid, palette=big, aseprite=False)
    assert ok.png_path.exists() and ok.aseprite_path is None

    ok255 = do_export(
        tmp_path / "ok" / "s",
        grid=seeded_grid(8, 8, 255),
        palette=big[:255],
        aseprite=True,
    )
    ase = parse_aseprite(ok255.aseprite_path.read_bytes())
    assert len(ase.palette) == 256
    assert ase.transparent_index == 255


# --- AC14 ---------------------------------------------------------------------------------


@pytest.mark.parametrize("fail_on", ["s.json", "s.aseprite"])
def test_aseprite_rollback(tmp_path, monkeypatch, fail_on):
    real = export_mod._atomic_write_bytes

    def flaky(path, data):
        if path.name == fail_on:
            raise OSError("disk on fire")
        real(path, data)

    monkeypatch.setattr(export_mod, "_atomic_write_bytes", flaky)
    with pytest.raises(OSError, match="disk on fire"):
        do_export(tmp_path / "s", aseprite=True)
    for name in ("s.png", "s@4x.png", "s.json", "s.aseprite"):
        assert not (tmp_path / name).exists(), name
    assert no_tmp(tmp_path)
    assert not (export_mod.pixelgen_home() / "last.json").exists()
