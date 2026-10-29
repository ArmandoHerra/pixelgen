"""PXG-07: export of the PNG, preview, sidecar and last.json."""

import ast
import importlib.metadata
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pixelgen.export as export_mod
from pixelgen.config import pixelgen_home
from pixelgen.errors import NoPreviousResultError, OutputExistsError, SpecError
from pixelgen.export import (
    TokenUsage,
    export,
    load_last,
    load_sidecar,
    output_paths,
)
from pixelgen.spec import ArtSpec

W, H = 24, 16
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
PALETTE = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)


def make_spec(**kw) -> ArtSpec:
    base = {"subject": "campfire", "width": W, "height": H, "palette": "pico8", "seed": 7}
    base.update(kw)
    return ArtSpec(**base)


def make_rgba(offset: int = 0) -> np.ndarray:
    rng = np.random.default_rng(1 + offset)
    arr = rng.integers(0, 256, size=(H, W, 4), dtype=np.uint8)
    arr[..., 3] = np.array([0, 128, 255], dtype=np.uint8)[
        (np.arange(H)[:, None] + np.arange(W)[None, :] + offset) % 3
    ]
    arr[0, 0] = [10, 20, 30, 0]  # RGB under alpha 0 must survive
    return arr


def make_grid(rows: int = H, cols: int = W, shift: int = 0) -> list[list[int]]:
    return [[((y + x + shift) % 4) - 1 for x in range(cols)] for y in range(rows)]


def do_export(output: Path, **kw):
    args = {
        "rgba": make_rgba(),
        "spec": make_spec(),
        "grid": make_grid(),
        "palette_colors": PALETTE,
        "model": "claude-test",
        "backend": "llm-grid",
        "token_usage": TokenUsage(input_tokens=10, output_tokens=20, calls=2),
        "scale": 8,
        "now": NOW,
    }
    args.update(kw)
    return export(output, args.pop("rgba"), **args)


def no_tmp(directory: Path) -> bool:
    return not list(Path(directory).glob("*.tmp"))


def test_output_paths_naming():
    for given in (Path("out/campfire.png"), Path("out/campfire")):
        png, prev, side = output_paths(given, 8)
        assert png.is_absolute() and prev.is_absolute() and side.is_absolute()
        assert str(png).endswith("out/campfire.png")
        assert str(prev).endswith("out/campfire@8x.png")
        assert str(side).endswith("out/campfire.json")
    png, prev, side = output_paths(Path("out/a.b"), 3)
    assert str(png).endswith("out/a.b.png")
    assert str(prev).endswith("out/a.b@3x.png")
    assert str(side).endswith("out/a.b.json")
    assert output_paths(Path("x.PNG"), 1)[0].name == "x.png"
    assert output_paths(Path("x.PNG"), 1)[2].name == "x.json"


def test_native_png_is_exact_rgba(tmp_path):
    rgba = make_rgba()
    result = do_export(tmp_path / "s", rgba=rgba)
    with Image.open(result.png_path) as img:
        assert img.mode == "RGBA"
        assert img.size == (W, H)
        assert np.array_equal(np.asarray(img), rgba)


@pytest.mark.parametrize("scale", [1, 3, 8])
def test_preview_is_nearest_neighbour_blocks(tmp_path, scale):
    rgba = make_rgba()
    result = do_export(tmp_path / "s", rgba=rgba, scale=scale)
    assert result.preview_path.name == f"s@{scale}x.png"
    with Image.open(result.preview_path) as img:
        assert img.mode == "RGBA"
        assert img.size == (W * scale, H * scale)
        big = np.asarray(img)
    for y in range(H):
        for x in range(W):
            block = big[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale]
            assert (block == rgba[y, x]).all()


@pytest.mark.parametrize(
    "case",
    [
        "scale0",
        "scale-1",
        "scale-bool",
        "scale-float",
        "rgba-swapped",
        "rgba-float",
        "grid-ragged",
        "grid-index-too-high",
        "grid-index-minus-2",
        "palette-shape",
    ],
)
def test_invalid_inputs_write_nothing(tmp_path, case):
    kw: dict = {}
    if case == "scale0":
        kw["scale"] = 0
    elif case == "scale-1":
        kw["scale"] = -1
    elif case == "scale-bool":
        kw["scale"] = True
    elif case == "scale-float":
        kw["scale"] = 2.0
    elif case == "rgba-swapped":
        kw["rgba"] = np.zeros((W, H, 4), dtype=np.uint8)
    elif case == "rgba-float":
        kw["rgba"] = np.zeros((H, W, 4), dtype=np.float64)
    elif case == "grid-ragged":
        grid = make_grid()
        grid[3] = grid[3][:-1]
        kw["grid"] = grid
    elif case == "grid-index-too-high":
        grid = make_grid()
        grid[0][0] = len(PALETTE)
        kw["grid"] = grid
    elif case == "grid-index-minus-2":
        grid = make_grid()
        grid[0][0] = -2
        kw["grid"] = grid
    elif case == "palette-shape":
        kw["palette_colors"] = np.zeros((3, 4), dtype=np.uint8)
    target = tmp_path / "new" / "dir" / "sprite"
    with pytest.raises(ValueError):
        do_export(target, **kw)
    assert not target.parent.exists()
    assert not (pixelgen_home() / "last.json").exists()


def test_sidecar_keys_and_types(tmp_path):
    result = do_export(tmp_path / "s")
    text = result.sidecar_path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert list(data) == [
        "schema_version",
        "pixelgen_version",
        "created_at",
        "spec",
        "grid",
        "palette",
        "model",
        "backend",
        "token_usage",
        "outputs",
    ]
    assert data["schema_version"] == 1 and type(data["schema_version"]) is int
    assert data["pixelgen_version"] == importlib.metadata.version("pixelgen")
    assert data["created_at"] == "2026-10-28T12:00:00Z"
    assert data["palette"] == {"name": "pico8", "colors": ["ff0000", "00ff00", "0000ff"]}
    assert list(data["token_usage"]) == ["input_tokens", "output_tokens", "calls"]
    assert data["outputs"] == {"png": "s.png", "preview": "s@8x.png", "scale": 8}
    assert str(tmp_path) not in text
    assert os.sep + "home" + os.sep not in text


@pytest.mark.parametrize(
    ("rows", "cols", "model"),
    [(H, W, "claude-test"), (8, 12, "claude-test"), (H, W, None)],
)
def test_sidecar_round_trips(tmp_path, rows, cols, model):
    spec = make_spec()
    grid = make_grid(rows, cols)
    usage = TokenUsage(input_tokens=3, output_tokens=4, calls=1)
    result = do_export(tmp_path / "s", spec=spec, grid=grid, model=model, token_usage=usage)
    loaded = load_sidecar(result.sidecar_path)
    assert loaded.spec == spec
    assert loaded.grid == grid
    assert loaded.palette.name == spec.palette
    assert loaded.palette.colors == ["ff0000", "00ff00", "0000ff"]
    assert loaded.model == model
    assert loaded.backend == "llm-grid"
    assert loaded.token_usage == usage
    assert loaded.outputs.png == "s.png"
    assert loaded.outputs.preview == "s@8x.png"
    assert loaded.outputs.scale == 8


def _good_sidecar_dict(tmp_path) -> dict:
    result = do_export(tmp_path / "good")
    return json.loads(result.sidecar_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case",
    [
        "version-missing",
        "version-2",
        "version-str",
        "version-true",
        "version-0",
        "version-1.0",
        "non-json",
        "json-array",
        "extra-key",
        "grid-missing",
        "missing-file",
    ],
)
def test_load_sidecar_rejects(tmp_path, case):
    path = tmp_path / "bad.json"
    data = _good_sidecar_dict(tmp_path)
    version_cases = {
        "version-2": 2,
        "version-str": "1",
        "version-true": True,
        "version-0": 0,
        "version-1.0": 1.0,
    }
    if case == "version-missing":
        del data["schema_version"]
    elif case in version_cases:
        data["schema_version"] = version_cases[case]
    elif case == "extra-key":
        data["surprise"] = 1
    elif case == "grid-missing":
        del data["grid"]
    if case == "non-json":
        path.write_bytes(b"\x00\xffnot json{")
    elif case == "json-array":
        path.write_text("[1, 2]", encoding="utf-8")
    elif case != "missing-file":
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SpecError) as info:
        load_sidecar(path)
    message = str(info.value)
    assert str(path) in message
    assert "\n" not in message
    if case.startswith("version-"):
        assert "schema_version" in message


@pytest.mark.parametrize("existing", ["png", "preview", "sidecar"])
def test_refuses_to_clobber_without_partial_write(tmp_path, existing):
    target = tmp_path / "out" / "s"
    paths = dict(zip(("png", "preview", "sidecar"), output_paths(target, 8), strict=True))
    target.parent.mkdir()
    paths[existing].write_bytes(b"SENTINEL")
    with pytest.raises(OutputExistsError) as info:
        do_export(target, overwrite=False)
    assert str(paths[existing]) in str(info.value)
    for key, path in paths.items():
        if key != existing:
            assert not path.exists()
    assert paths[existing].read_bytes() == b"SENTINEL"
    assert not (pixelgen_home() / "last.json").exists()
    assert no_tmp(target.parent)


def test_overwrite_true_replaces_all(tmp_path):
    target = tmp_path / "s"
    do_export(target)
    spec_b = make_spec(subject="castle", seed=99)
    grid_b = make_grid(shift=1)
    rgba_b = make_rgba(offset=1)
    result = do_export(target, spec=spec_b, grid=grid_b, rgba=rgba_b, overwrite=True)
    with Image.open(result.png_path) as img:
        assert np.array_equal(np.asarray(img), rgba_b)
    loaded = load_sidecar(result.sidecar_path)
    assert loaded.spec == spec_b
    assert loaded.grid == grid_b
    assert load_last().sidecar.spec == spec_b


def test_creates_output_directories(tmp_path):
    result = do_export(tmp_path / "a" / "b" / "c" / "sprite")
    assert (tmp_path / "a" / "b" / "c").is_dir()
    assert result.png_path.is_file()
    assert result.preview_path.is_file()
    assert result.sidecar_path.is_file()


def test_last_json_written_and_survives_move(tmp_path):
    spec, grid = make_spec(), make_grid()
    result = do_export(tmp_path / "out" / "s", spec=spec, grid=grid)
    assert result.last_path == pixelgen_home() / "last.json"
    wrapper = json.loads(result.last_path.read_text(encoding="utf-8"))
    assert wrapper == {
        "schema_version": 1,
        "sidecar_path": str(result.sidecar_path),
        "sidecar": json.loads(result.sidecar_path.read_text(encoding="utf-8")),
    }
    last = load_last()
    assert last.sidecar_path == result.sidecar_path
    assert last.sidecar == load_sidecar(result.sidecar_path)
    shutil.rmtree(tmp_path / "out")
    last = load_last()
    assert last.sidecar.spec == spec
    assert last.sidecar.grid == grid


def test_last_json_respects_pixelgen_home_at_call_time(tmp_path, tmp_path_factory, monkeypatch):
    result = do_export(tmp_path / "first")
    assert result.last_path.resolve().is_relative_to(tmp_path_factory.getbasetemp().resolve())
    home_a, home_b = tmp_path / "homeA", tmp_path / "homeB"
    assert not home_a.exists() and not home_b.exists()
    spec_1, spec_2 = make_spec(subject="one"), make_spec(subject="two")
    monkeypatch.setenv("PIXELGEN_HOME", str(home_a))
    do_export(tmp_path / "r1", spec=spec_1)
    monkeypatch.setenv("PIXELGEN_HOME", str(home_b))
    do_export(tmp_path / "r2", spec=spec_2)
    a = json.loads((home_a / "last.json").read_text(encoding="utf-8"))
    b = json.loads((home_b / "last.json").read_text(encoding="utf-8"))
    assert a["sidecar"]["spec"]["subject"] == "one"
    assert b["sidecar"]["spec"]["subject"] == "two"


def test_load_last_missing_and_corrupt(tmp_path):
    last = pixelgen_home() / "last.json"
    with pytest.raises(NoPreviousResultError) as info:
        load_last()
    assert str(last) in str(info.value)
    last.parent.mkdir(parents=True, exist_ok=True)
    for content in ('{"schema_version": 1}', '{"schema_version": 2, "sidecar": {}}', "not json"):
        last.write_text(content, encoding="utf-8")
        with pytest.raises(SpecError) as info:
            load_last()
        assert str(last) in str(info.value)
        assert "\n" not in str(info.value)


def test_failed_write_leaves_no_partial_set(tmp_path, monkeypatch):
    target = tmp_path / "out" / "s"
    sidecar_target = output_paths(target, 8)[2]
    original = export_mod._atomic_write_bytes

    def flaky(path, data):
        if path == sidecar_target:
            raise OSError("disk full")
        original(path, data)

    monkeypatch.setattr(export_mod, "_atomic_write_bytes", flaky)
    with pytest.raises(OSError, match="disk full"):
        do_export(target, overwrite=False)
    png, preview, _ = output_paths(target, 8)
    assert not png.exists()
    assert not preview.exists()
    assert no_tmp(target.parent)
    assert not (pixelgen_home() / "last.json").exists()


def test_export_imports_stay_inside_pxg01_surface():
    root = Path(__file__).parents[1]
    forbidden = (
        "pixelgen.llm",
        "pixelgen.pixelate",
        "pixelgen.palettes",
        "pixelgen.presets",
        "pixelgen.pipeline",
        "pixelgen.backends",
        "httpx",
        "anthropic",
    )
    bad = []
    for rel in ("src/pixelgen/export.py", "tests/test_export.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                if any(name == f or name.startswith(f + ".") for f in forbidden):
                    bad.append(f"{name} in {rel}")
    assert not bad, bad
