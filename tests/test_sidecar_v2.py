"""PXG-39: sidecar schema v2 (provider, model, per-step usage units, dropped fields).

Everything is offline: ``FakeLLM`` through the ``cli.make_llm`` seam, ``FakeTransport`` for the
test-only provider backend, dummy keys, a per-test ``PIXELGEN_HOME`` (conftest). New symbols are
fetched through ``_Lazy`` so a tree without v2 fails these tests by assertion, not at collection.
"""

import hashlib
import importlib
import importlib.util
import json
import re
import shutil
import sys
import types
import typing
from datetime import UTC, datetime
from pathlib import Path

import fakes
import numpy as np
import pytest
from aseprite_oracle import compare_with_sidecar, parse_aseprite
from typer.testing import CliRunner

import pixelgen.cli as cli
from pixelgen.backends import base
from pixelgen.backends.base import GenerationUsage, create_generator
from pixelgen.cli import app
from pixelgen.config import pixelgen_home
from pixelgen.errors import PixelgenError, SpecError
from pixelgen.export import TokenUsage, load_last, load_sidecar
from pixelgen.llm import Usage
from pixelgen.palettes import load_palette
from pixelgen.spec import ArtSpec

runner = CliRunner()
REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
V1_KEYS = [
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
V2_KEYS = ["provider", "provider_model", "usage", "dropped_fields"]
USAGE_KEYS = ["step", "provider", "model", "unit", "calls", "quantity"]
ANTHROPIC_DUMMY = "dummy-anthropic-7f3a"
PROV_DUMMY = "dummy-prov-9c1e"
FROZEN_SHA = {
    "knight/knight.json": "f6bea8c175cf0e0864074c796861f939cb3ef804198c3d1f656d44462e180741",
    "knight/knight-v2.json": "db464c3520299860c370f3bff9d243b83adcf13babdb4c570f6985edd1938a00",
    "red-apple/red-apple.json": "970bb21ff844ef90dae3dc6b009c398636c28acb3d65d6f4601c6652f9be0527",
    "slime/slime.json": "61ec7d33277f6f5c655f10e30c74e6bf02fe35b29e4f18f2a033a5b1b89249c6",
    "knight/run.json": "a8f4bccab18b9eaa9e12b9d3c93febef283c6cac192b07d0fbe063ad56bbafa3",
    "red-apple/run.json": "4054eb8cbb00b0d166c3d553332e75843bdb93c81361f8ae8d78205d1a541a9d",
    "slime/run.json": "75f49387762959fb9f99d2efcfd5924c5d1c0218d378585ec9528eb50ca2ac77",
}
SIDECARS = sorted(k for k in FROZEN_SHA if not k.endswith("run.json"))


class _Lazy:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        module = importlib.import_module(self._name)
        assert hasattr(module, attr), f"{self._name}.{attr} is missing"
        return getattr(module, attr)


export = _Lazy("pixelgen.export")
pipeline = _Lazy("pixelgen.pipeline")
providers = _Lazy("pixelgen.providers")


def _grid(seed=0, side=16, colours=16):
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5 + seed) % colours
            for c in range(side)
        ]
        for r in range(side)
    ]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_DUMMY)
    monkeypatch.setenv("FAKEPROV_API_KEY", PROV_DUMMY)
    return tmp_path


def _invoke(monkeypatch, args, *script):
    llm = fakes.FakeLLM(*script)
    monkeypatch.setattr(cli, "make_llm", lambda: llm)
    return llm, runner.invoke(app, args)


def _generate(monkeypatch, tmp, *script):
    return _invoke(
        monkeypatch,
        ["generate", "a red apple", "--size", "16", "--palette", "pico8", "-o", str(tmp / "a")],
        *script,
    )


def _make_prov(name="fakeprov", units="images"):
    return providers.ProviderConfig(
        name=name,
        key_env="FAKEPROV_API_KEY",
        default_model="fake-image-1",
        model_env=None,
        base_url="https://fake-image.invalid/v1",
        usage_units=units,
    )


def _run_generate(tmp, llm, transport, backend="fakeprov", **kw):
    generator = create_generator(backend, llm=llm, transport=transport)
    return pipeline.run_generate(
        "a red apple",
        llm=llm,
        generator=generator,
        backend_name=backend,
        preset_name=None,
        cli_overrides={"width": 16, "height": 16, "palette": "pico8"},
        output=tmp / "p",
        overwrite=False,
        scale=4,
        **kw,
    )


def _direct_export(out, **kw):
    spec = ArtSpec(subject="s", width=8, height=8, palette="pico8", seed=1)
    palette = load_palette("pico8")
    grid = [[(r + c) % 16 for c in range(8)] for r in range(8)]
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            rgba[r, c, :3] = palette[grid[r][c]]
            rgba[r, c, 3] = 255
    return export.export(
        out,
        rgba,
        spec=spec,
        grid=grid,
        palette_colors=palette,
        model="m",
        backend="llm-grid",
        token_usage=TokenUsage(input_tokens=1, output_tokens=2, calls=1),
        scale=2,
        now=datetime(2026, 11, 2, tzinfo=UTC),
        **kw,
    )


# --- AC1 ------------------------------------------------------------------------------------


def test_generate_llm_grid_writes_v2(env, monkeypatch):
    llm, result = _generate(monkeypatch, env, {}, {"grid": _grid()})
    assert result.exit_code == 0, (result.stdout, result.stderr)
    data = _json(env / "a.json")
    assert data["schema_version"] == 2
    assert list(data) == V1_KEYS + V2_KEYS
    assert (data["backend"], data["model"]) == ("llm-grid", "fake-model")
    assert (data["provider"], data["provider_model"]) == ("anthropic", "fake-model")
    i, o = 10, 20  # FakeLLM default Usage per call
    assert data["usage"] == [
        {
            "step": "infer_spec",
            "provider": "anthropic",
            "model": "fake-model",
            "unit": "tokens",
            "calls": 1,
            "quantity": i + o,
        },
        {
            "step": "generate",
            "provider": "anthropic",
            "model": "fake-model",
            "unit": "tokens",
            "calls": 1,
            "quantity": i + o,
        },
    ]
    tu = data["token_usage"]
    assert tu == {"input_tokens": 2 * i, "output_tokens": 2 * o, "calls": 2}
    assert sum(u["quantity"] for u in data["usage"]) == tu["input_tokens"] + tu["output_tokens"]
    assert llm.call_count == 2


# --- AC2 ------------------------------------------------------------------------------------


def test_to_usage_record_quantity_rule():
    def rec(unit, usage, step="generate"):
        return pipeline.to_usage_record(step, provider="p", model="m", unit=unit, usage=usage)

    r = rec("tokens", GenerationUsage(3, 100, 200))
    assert (r.calls, r.quantity) == (3, 300)
    r = rec("images", GenerationUsage(2, 0, 0))
    assert (r.calls, r.quantity) == (2, 2)
    r = rec("tokens", Usage(5, 7), step="infer_spec")
    assert (r.calls, r.quantity) == (1, 12)
    assert list(r.model_dump()) == USAGE_KEYS


# --- AC3 ------------------------------------------------------------------------------------


def test_v2_key_order_and_defaults(env):
    result = _direct_export(env / "d")
    data = _json(result.sidecar_path)
    assert list(data) == V1_KEYS + V2_KEYS
    assert data["schema_version"] == 2
    assert (data["provider"], data["provider_model"], data["usage"], data["dropped_fields"]) == (
        None,
        None,
        [],
        None,
    )
    rec = export.UsageRecord(
        step="generate", provider="p", model=None, unit="calls", calls=2, quantity=2
    )
    result = _direct_export(env / "e", provider="p", usage=[rec])
    data = _json(result.sidecar_path)
    assert list(data["usage"][0]) == USAGE_KEYS
    assert data["provider"] == "p" and data["model"] == "m"
    assert list(data["outputs"]) == ["png", "preview", "scale"]
    with_ase = _direct_export(env / "f", aseprite=True)
    assert list(_json(with_ase.sidecar_path)["outputs"]) == ["png", "preview", "scale", "aseprite"]


# --- AC4 ------------------------------------------------------------------------------------


def test_dropped_fields_recorded(env, monkeypatch):
    reply = {"bogus": 1, "palette": "nope", "seed": 4, "width": 16}
    _, result = _generate(monkeypatch, env, reply, {"grid": _grid()})
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert _json(env / "a.json")["dropped_fields"] == [
        {"field": "bogus", "reason": "unknown field", "value": "1"},
        {"field": "palette", "reason": "unknown palette", "value": "'nope'"},
        {"field": "seed", "reason": "not inferable", "value": "4"},
    ]
    clean = env / "clean"
    clean.mkdir()
    _, result = _generate(monkeypatch, clean, {}, {"grid": _grid()})
    assert result.exit_code == 0
    assert _json(clean / "a.json")["dropped_fields"] == []
    src = env / "knight.json"
    shutil.copy(EXAMPLES / "knight" / "knight.json", src)
    _, result = _invoke(
        monkeypatch,
        ["refine", "make it blue", "--from", str(src)],
        {"spec": {}, "grid": _grid(6, side=32)},
    )
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert _json(env / "knight-v2.json")["dropped_fields"] is None


# --- AC5 ------------------------------------------------------------------------------------


def test_provider_backend_writes_image_units(env, monkeypatch):
    fakes.register_fake_provider(monkeypatch, name="fakeprov", provider=_make_prov())
    transport = fakes.FakeTransport(fakes.fake_image_reply(64, 64, seed=3))
    llm = fakes.FakeLLM({})
    result = _run_generate(env, llm, transport)
    data = _json(result.export.sidecar_path)
    assert (data["provider"], data["provider_model"], data["model"]) == (
        "fakeprov",
        "fake-image-1",
        "fake-model",
    )
    assert data["usage"][1] == {
        "step": "generate",
        "provider": "fakeprov",
        "model": "fake-image-1",
        "unit": "images",
        "calls": 1,
        "quantity": 1,
    }
    assert data["usage"][0]["provider"] == "anthropic"
    assert transport.call_count == 1


def test_unknown_usage_unit_refused_before_spend(env, monkeypatch):
    fakes.register_fake_provider(monkeypatch, name="fakeprov", provider=_make_prov())
    original = base.provider_for
    bad = types.SimpleNamespace(
        name="fakeprov",
        key_env="FAKEPROV_API_KEY",
        default_model="fake-image-1",
        model_env=None,
        base_url="https://fake-image.invalid/v1",
        usage_units="dollars",
    )
    monkeypatch.setattr(
        base, "provider_for", lambda name: bad if name == "fakeprov" else original(name)
    )
    transport = fakes.FakeTransport(fakes.fake_image_reply(64, 64, seed=3))
    with pytest.raises(PixelgenError) as info:
        _run_generate(env, fakes.FakeLLM({}), transport)
    assert "dollars" in str(info.value)
    assert transport.call_count == 0
    assert not list(env.glob("p*"))
    assert not (pixelgen_home() / "last.json").exists()


def test_registered_usage_units_known(env, monkeypatch):
    with pytest.raises(ValueError):
        _make_prov(units="dollars")
    assert export.USAGE_UNITS == providers.USAGE_UNITS
    annotation = export.UsageRecord.model_fields["unit"].annotation
    assert typing.get_args(annotation) == export.USAGE_UNITS
    step_annotation = export.UsageRecord.model_fields["step"].annotation
    assert typing.get_args(step_annotation) == export.USAGE_STEPS
    names = base.available_backends()
    assert names == ["diffusion", "llm-grid", "openrouter"]
    for name in names:
        assert base.provider_for(name).usage_units in export.USAGE_UNITS


# --- AC6, AC7 -------------------------------------------------------------------------------


def test_refine_from_committed_v1_sidecar_writes_v2(env, monkeypatch):
    src = env / "knight.json"
    shutil.copy(EXAMPLES / "knight" / "knight.json", src)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    llm, result = _invoke(
        monkeypatch,
        ["refine", "make it blue", "--from", str(src)],
        {"spec": {}, "grid": _grid(6, side=32)},
    )
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before
    assert _json(src)["schema_version"] == 1
    data = _json(env / "knight-v2.json")
    assert data["schema_version"] == 2
    assert data["usage"] == [
        {
            "step": "refine",
            "provider": "anthropic",
            "model": "fake-model",
            "unit": "tokens",
            "calls": 1,
            "quantity": 30,
        }
    ]
    assert data["provider"] == "anthropic" and data["dropped_fields"] is None
    assert data["backend"] == "llm-grid"
    assert llm.call_count == 1


def test_refine_from_v1_last_json(env, monkeypatch):
    src = env / "knight.json"
    shutil.copy(EXAMPLES / "knight" / "knight.json", src)
    home = pixelgen_home()
    home.mkdir(parents=True, exist_ok=True)
    wrapper = {"schema_version": 1, "sidecar_path": str(src), "sidecar": _json(src)}
    (home / "last.json").write_text(json.dumps(wrapper), encoding="utf-8")
    _, result = _invoke(monkeypatch, ["refine", "x"], {"spec": {}, "grid": _grid(6, side=32)})
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert _json(env / "knight-v2.json")["schema_version"] == 2
    last = _json(home / "last.json")
    assert last["schema_version"] == 2 and last["sidecar"]["schema_version"] == 2


# --- AC8 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("rel", SIDECARS)
def test_v1_examples_frozen_and_redump_identical(rel):
    raw = (EXAMPLES / rel).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA[rel]
    side = load_sidecar(EXAMPLES / rel)
    assert side.schema_version == 1
    assert side.provider is None and side.provider_model is None
    assert side.usage is None and side.dropped_fields is None
    assert (side.model_dump_json(indent=2) + "\n").encode("utf-8") == raw
    for run in FROZEN_SHA:
        if run.endswith("run.json"):
            assert hashlib.sha256((EXAMPLES / run).read_bytes()).hexdigest() == FROZEN_SHA[run]


# --- AC10 -----------------------------------------------------------------------------------


@pytest.mark.parametrize("version", [3, 0, "2", True, 2.0, "missing"], ids=repr)
def test_unknown_version_refused(tmp_path, monkeypatch, version):
    data = _json(EXAMPLES / "knight" / "knight.json")
    if version == "missing":
        del data["schema_version"]
    else:
        data["schema_version"] = version
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SpecError) as info:
        load_sidecar(path)
    message = str(info.value)
    assert str(path) in message and "schema_version" in message and "\n" not in message
    if version == 3:
        assert message.startswith(f"{path}: unsupported sidecar schema_version 3")
    last = tmp_path / "home" / "last.json"
    last.parent.mkdir()
    monkeypatch.setenv("PIXELGEN_HOME", str(last.parent))
    wrapper = {"schema_version": 3, "sidecar_path": "s.json", "sidecar": data}
    last.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(SpecError) as info:
        load_last()
    assert "unsupported last.json schema_version 3" in str(info.value)


def test_refine_from_v3_sidecar_exits_2_with_no_spend(env, monkeypatch):
    data = _json(EXAMPLES / "knight" / "knight.json")
    data["schema_version"] = 3
    src = env / "v3.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    llm, result = _invoke(monkeypatch, ["refine", "x", "--from", str(src)])
    assert result.exit_code == 2
    assert result.stderr.startswith("pixelgen: error:")
    assert llm.call_count == 0


# --- AC11 -----------------------------------------------------------------------------------


def _v2_dict(tmp_path):
    result = _direct_export(tmp_path / "v")
    return _json(result.sidecar_path)


def _expect_rejected(tmp_path, data, field):
    path = tmp_path / "x.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SpecError) as info:
        load_sidecar(path)
    message = str(info.value)
    assert str(path) in message and field in message and "\n" not in message


def test_version_key_sets_strict(tmp_path):
    v1 = _json(EXAMPLES / "knight" / "knight.json")
    for key, value in (
        ("provider", "p"),
        ("provider_model", None),
        ("usage", []),
        ("dropped_fields", None),
    ):
        _expect_rejected(tmp_path, {**v1, key: value}, key)
    good = _v2_dict(tmp_path)
    assert load_sidecar(_write(tmp_path, good)).usage == []
    for key in ("usage", "provider"):
        _expect_rejected(tmp_path, {k: v for k, v in good.items() if k != key}, key)
    record = {
        "step": "generate",
        "provider": "p",
        "model": None,
        "unit": "calls",
        "calls": 1,
        "quantity": 1,
    }
    ok = {**good, "usage": [record]}
    assert load_sidecar(_write(tmp_path, ok)).usage[0].unit == "calls"
    for field, bad in (("unit", "dollars"), ("step", "render"), ("quantity", -1), ("extra", 1)):
        _expect_rejected(tmp_path, {**good, "usage": [{**record, field: bad}]}, "usage")
    drop = {"field": "a", "reason": "b", "value": "c"}
    assert load_sidecar(_write(tmp_path, {**good, "dropped_fields": [drop]})).dropped_fields
    _expect_rejected(tmp_path, {**good, "dropped_fields": [{**drop, "extra": 1}]}, "dropped_fields")


def _write(tmp_path, data):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- AC12 -----------------------------------------------------------------------------------


def test_last_json_v1_and_v2_load(tmp_path, monkeypatch):
    home = tmp_path / "lasthome"
    home.mkdir()
    monkeypatch.setenv("PIXELGEN_HOME", str(home))
    v1 = EXAMPLES / "knight" / "knight.json"
    (home / "last.json").write_text(
        json.dumps({"schema_version": 1, "sidecar_path": str(v1), "sidecar": _json(v1)}),
        encoding="utf-8",
    )
    last = load_last()
    assert last.sidecar == load_sidecar(v1) and last.sidecar.schema_version == 1
    result = _direct_export(tmp_path / "w")
    wrapper = _json(result.last_path)
    assert wrapper["schema_version"] == 2
    last = load_last()
    assert last.sidecar == load_sidecar(result.sidecar_path)
    assert last.sidecar.schema_version == 2


# --- AC13 -----------------------------------------------------------------------------------


def _load_checker():
    script = REPO / "scripts" / "check_examples.py"
    spec = importlib.util.spec_from_file_location("check_examples_v2", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_examples_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


def _replay_v2(src_json, dest_dir):
    side = load_sidecar(src_json)
    palette = np.array(
        [[int(c[i : i + 2], 16) for i in (0, 2, 4)] for c in side.palette.colors], dtype=np.uint8
    )
    rgba = np.zeros((*np.asarray(side.grid).shape, 4), dtype=np.uint8)
    for (y, x), idx in np.ndenumerate(np.asarray(side.grid)):
        if idx >= 0:
            rgba[y, x, :3] = palette[idx]
            rgba[y, x, 3] = 255
    return export.export(
        dest_dir / src_json.stem,
        rgba,
        spec=side.spec,
        grid=side.grid,
        palette_colors=palette,
        model=side.model,
        backend=side.backend,
        token_usage=side.token_usage,
        scale=side.outputs.scale,
        now=side.created_at,
    )


def test_checker_accepts_v1_and_v2_tree(tmp_path):
    checker = _load_checker()
    root = tmp_path / "examples"
    shutil.copytree(EXAMPLES / "slime", root / "slime")  # v1, as committed
    v2_dir = root / "red-apple"
    v2_dir.mkdir(parents=True)
    result = _replay_v2(EXAMPLES / "red-apple" / "red-apple.json", v2_dir)
    shutil.copy(EXAMPLES / "red-apple" / "run.json", v2_dir / "run.json")
    assert _json(result.sidecar_path)["schema_version"] == 2
    assert _json(root / "slime" / "slime.json")["schema_version"] == 1
    assert checker.check_tree(root) == []


# --- AC14 -----------------------------------------------------------------------------------


def test_oracle_sidecar_reader_accepts_v2(tmp_path):
    result = _direct_export(tmp_path / "o", aseprite=True)
    data = _json(result.sidecar_path)
    assert data["schema_version"] == 2
    ase = parse_aseprite(result.aseprite_path.read_bytes())
    assert compare_with_sidecar(ase, data) == []


# --- AC15 -----------------------------------------------------------------------------------


def _keys_at_any_depth(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _keys_at_any_depth(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _keys_at_any_depth(item)


def test_no_key_material_or_money_in_v2(env, monkeypatch):
    _generate(monkeypatch, env, {}, {"grid": _grid()})
    fakes.register_fake_provider(monkeypatch, name="fakeprov", provider=_make_prov())
    result = _run_generate(
        env, fakes.FakeLLM({}), fakes.FakeTransport(fakes.fake_image_reply(64, 64, seed=3))
    )
    home = pixelgen_home()
    files = [env / "a.json", result.export.sidecar_path, home / "last.json"]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert ANTHROPIC_DUMMY not in text and PROV_DUMMY not in text
        assert "$" not in text
        for key in _keys_at_any_depth(json.loads(text)):
            assert not re.search(r"(?i)cost|price|usd|dollar", key), key
