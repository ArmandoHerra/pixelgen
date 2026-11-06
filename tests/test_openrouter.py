"""Unit tests for the OpenRouter backend (PXG-42, AC1-AC5, AC14). Offline: FakeTransport only."""

import ast
import base64
import uuid
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeTransport
from openrouter_fixture import images_reply, jpeg_b64, png_b64, png_bytes, sprite_on_backdrop

from pixelgen import providers
from pixelgen.backends import base
from pixelgen.backends import openrouter as orr
from pixelgen.backends.base import available_backends, check_backend_env, create_generator
from pixelgen.backends.openrouter import (
    ModelIdError,
    OpenRouterGenerator,
    OpenRouterResponseError,
    build_prompt,
    centre_crop,
    decode_image,
)
from pixelgen.errors import PixelgenError
from pixelgen.spec import ArtSpec

SRC = Path(__file__).resolve().parents[1] / "src" / "pixelgen" / "backends" / "openrouter.py"
MODEL_VAR = "PIXELGEN_OPENROUTER_MODEL"


@pytest.fixture(autouse=True)
def _clear_model(monkeypatch):
    monkeypatch.delenv(MODEL_VAR, raising=False)


def _small_reply():
    return images_reply(png_b64(sprite_on_backdrop(1, 64)))


# --- AC1 ---------------------------------------------------------------------------------------


def test_openrouter_config_and_registry(monkeypatch):
    assert providers.OPENROUTER == providers.ProviderConfig(
        "openrouter",
        "OPENROUTER_API_KEY",
        "openai/gpt-image-2",
        "PIXELGEN_OPENROUTER_MODEL",
        "https://openrouter.ai/api/v1",
        "images",
    )
    assert base.provider_for("openrouter") is providers.OPENROUTER
    assert base.uses_transport("openrouter") is True
    assert base._resolve("openrouter") is OpenRouterGenerator
    assert available_backends() == ["diffusion", "llm-grid", "openrouter"]
    assert providers.required_keys("openrouter") == ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert isinstance(create_generator("openrouter", llm=None), OpenRouterGenerator)
    monkeypatch.setenv(MODEL_VAR, "Bad Id")
    assert isinstance(create_generator("openrouter", llm=None), OpenRouterGenerator)


# --- AC2 ---------------------------------------------------------------------------------------


def test_model_id_validation():
    for env in (
        {},
        {MODEL_VAR: "  "},
        {MODEL_VAR: "openai/gpt-image-2"},
        {MODEL_VAR: "black-forest-labs/flux.2-klein-4b"},
        {MODEL_VAR: "google/gemini-3.1-flash-lite-image-20260630"},
        {MODEL_VAR: "x/y:free"},
    ):
        assert OpenRouterGenerator.check_env(env) is None, env
    sentinel = "sk-or-v1-pxg42-" + uuid.uuid4().hex
    for bad in (
        "gpt-image-2",
        "openai/",
        "OpenAI/GPT",
        "a/b/c",
        "openai/gpt image",
        "https://x/y",
        sentinel,
    ):
        with pytest.raises(ModelIdError) as info:
            OpenRouterGenerator.check_env({MODEL_VAR: bad})
        msg = str(info.value)
        assert MODEL_VAR in msg and "\n" not in msg
        assert isinstance(info.value, PixelgenError)
        if bad in (sentinel, "https://x/y", "a/b/c"):
            assert bad not in msg
    assert check_backend_env("llm-grid", {MODEL_VAR: "bad"}) is None
    assert check_backend_env("diffusion", {MODEL_VAR: "bad"}) is None


# --- AC3 ---------------------------------------------------------------------------------------


def _generate(spec, transport, monkeypatch, env_model=None):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-or-key")
    if env_model:
        monkeypatch.setenv(MODEL_VAR, env_model)
    gen = create_generator("openrouter", llm=None, transport=transport)
    return gen.generate(spec, np.zeros((2, 3), dtype=np.uint8))


def _body(spec, model="openai/gpt-image-2"):
    return {
        "model": model,
        "prompt": build_prompt(spec),
        "size": "1024x1024",
        "quality": "low",
        "output_format": "png",
        "n": 1,
    }


def test_request_shape(monkeypatch):
    spec = ArtSpec(subject="a red apple", width=16, height=16)
    ft = FakeTransport(_small_reply())
    result = _generate(spec, ft, monkeypatch)
    assert ft.call_count == 1
    call = ft.calls[0]
    assert call.url == "https://openrouter.ai/api/v1/images"
    assert call.headers == {"Authorization": "Bearer dummy-or-key"}
    assert call.body == _body(spec)
    assert call.timeout == providers.DEFAULT_TIMEOUT_SECONDS
    assert result.image is not None and result.image.shape == (64, 64, 3)
    assert result.usage.calls == 1

    for over in ({"seed": 7}, {"background": "transparent"}, {"view": "side"}):
        sp = ArtSpec(subject="a red apple", width=16, height=16, **over)
        ft = FakeTransport(_small_reply())
        _generate(sp, ft, monkeypatch)
        assert ft.call_count == 1 and ft.calls[0].body == _body(sp)
    assert "side view" in ft.calls[0].body["prompt"]
    noted = ArtSpec(subject="a pear", width=16, height=16, notes="chunky")
    assert build_prompt(noted).endswith(" Style notes: chunky.")

    ft = FakeTransport(_small_reply())
    _generate(spec, ft, monkeypatch, env_model="black-forest-labs/flux.2-klein-4b")
    assert ft.calls[0].body == _body(spec, "black-forest-labs/flux.2-klein-4b")


# --- AC4 ---------------------------------------------------------------------------------------


def test_centre_crop_to_target_aspect():
    img = np.random.default_rng(4).integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8)
    assert np.array_equal(centre_crop(img, 16, 16), img)
    out = centre_crop(img, 40, 24)
    assert out.shape == (614, 1024, 3) and np.array_equal(out, img[205:819, :])
    out = centre_crop(img, 24, 40)
    assert out.shape == (1024, 614, 3) and np.array_equal(out, img[:, 205:819])
    out = centre_crop(img, 128, 8)
    assert out.shape == (64, 1024, 3) and np.array_equal(out, img[480:544, :])
    out = centre_crop(img, 8, 128)
    assert out.shape == (1024, 64, 3) and np.array_equal(out, img[:, 480:544])
    wide = np.random.default_rng(5).integers(0, 256, size=(1024, 1536, 3), dtype=np.uint8)
    assert np.array_equal(centre_crop(wide, 16, 16), wide[:, 256:1280])
    assert centre_crop(img, 40, 24).flags["C_CONTIGUOUS"]


# --- AC5 ---------------------------------------------------------------------------------------

_PNG = png_b64(sprite_on_backdrop(2, 32))
_SIG_ONLY = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(20)).decode("ascii")
_WIDE = png_b64(np.zeros((8, 4097, 3), dtype=np.uint8))

ERROR_ROWS = [
    ({}, "had no image data"),
    ({"data": []}, "had no image data"),
    ({"data": "x"}, "had no image data"),
    ({"data": [{"media_type": "image/png"}]}, "had no b64_json image"),
    ({"data": [{"b64_json": 7}]}, "had no b64_json image"),
    ({"data": [{"b64_json": "@@@"}]}, "was not valid base64"),
    ({"data": [{"b64_json": jpeg_b64(sprite_on_backdrop(3, 32))}]}, "was not a PNG"),
    ({"data": [{"b64_json": _SIG_ONLY}]}, "could not be decoded"),
    ({"data": [{"b64_json": _WIDE}]}, "is 4097x8"),
]

TOLERATED_ROWS = [
    {"data": [{"b64_json": _PNG}]},
    {"data": [{"b64_json": _PNG, "media_type": "image/webp"}]},
    {"data": [{"b64_json": _PNG}, {"b64_json": "@@@"}]},
    {"data": [{"b64_json": _PNG}], "mystery": {"x": 1}},
]


def test_response_decoding_matrix():
    for reply, needle in ERROR_ROWS:
        with pytest.raises(OpenRouterResponseError) as info:
            decode_image(reply)
        msg = str(info.value)
        assert isinstance(info.value, PixelgenError)
        assert needle in msg and "\n" not in msg and "openrouter.ai" in msg
        assert info.value.__cause__ is None
    for reply in TOLERATED_ROWS:
        arr = decode_image(reply)
        assert arr.dtype == np.uint8 and arr.shape == (32, 32, 3)
    assert decode_image(images_reply(_PNG)).dtype == np.uint8
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    assert decode_image({"data": [{"b64_json": png_b64(rgba)}]}).shape == (8, 8, 4)
    assert png_bytes(rgba)[:8] == orr.PNG_SIGNATURE


# --- AC14 --------------------------------------------------------------------------------------

_FORBIDDEN = ("urllib.request", "http", "socket")
_ALLOWED_ROOTS = ("numpy", "PIL", "pixelgen")


def _collect_imports(tree: ast.AST, package: str = "pixelgen.backends") -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                module = ".".join(parts + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            names.add(module)
            names.update(f"{module}.{a.name}" for a in node.names)
    return names


def _offenders(names: set[str]) -> list[str]:
    return sorted(n for n in names if any(n == f or n.startswith(f + ".") for f in _FORBIDDEN))


def test_openrouter_imports_and_offline():
    names = _collect_imports(ast.parse(SRC.read_text(encoding="utf-8")))
    assert not _offenders(names), _offenders(names)
    stdlib = set(__import__("sys").stdlib_module_names)
    for n in names:
        root = n.split(".")[0]
        assert root in stdlib or root in _ALLOWED_ROOTS, n
    # the walker itself: forbidden forms are seen, relative imports and urllib.parse are not
    bad = _collect_imports(ast.parse("from urllib import request\n"))
    assert "urllib.request" in bad
    assert _offenders({"urllib.request"}) == ["urllib.request"]
    assert _offenders(_collect_imports(ast.parse("import http.client\nimport socket"))) == [
        "http.client",
        "socket",
    ]
    good = _collect_imports(
        ast.parse("from .base import Generator\nfrom . import base\nfrom ..providers import Y")
    )
    assert {"pixelgen.backends.base", "pixelgen.backends.base.Generator"} <= good
    assert {"pixelgen.backends", "pixelgen.providers"} <= good
    assert not _offenders(good)
    assert not _offenders(_collect_imports(ast.parse("from urllib.parse import urlsplit")))
