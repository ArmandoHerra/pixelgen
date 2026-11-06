"""CLI tests for ``--backend openrouter`` (PXG-42, AC6-AC13, AC16). Offline.

Every run is in-process (``CliRunner`` through ``cli.app``): the spec LLM is a ``FakeLLM`` behind
``cli.make_llm`` and the transport a ``FakeTransport`` behind ``cli.make_transport``, except
AC10 and AC12's 401 row, which use the real ``UrllibTransport`` with an injected opener (no
socket). A tripwire on ``UrllibTransport.post_json`` fails any other test that reaches it.
"""

import base64
import functools
import io
import json
import os
import re
import urllib.error
import uuid

import numpy as np
import pytest
from aseprite_oracle import main as oracle_main
from fakes import FakeLLM, FakeTransport
from openrouter_fixture import (
    COST_SENTINEL,
    TOKEN_SENTINEL,
    images_reply,
    jpeg_b64,
    png_b64,
    sprite_on_backdrop,
)
from PIL import Image
from typer.testing import CliRunner

import pixelgen.cli as cli
from pixelgen import providers
from pixelgen.backends.openrouter import build_prompt
from pixelgen.cli import app
from pixelgen.palettes import load_palette
from pixelgen.pixelate import grid_to_rgba, rgba_to_grid
from pixelgen.providers import TransportError
from pixelgen.spec import ArtSpec

runner = CliRunner()
MODEL_VAR = "PIXELGEN_OPENROUTER_MODEL"
MISSING_OR = (
    "pixelgen: error: OPENROUTER_API_KEY is not set; the 'openrouter' backend needs it. "
    "Export it in your shell and try again."
)


@functools.lru_cache(maxsize=1)
def _good_b64() -> str:
    return png_b64(sprite_on_backdrop(11))


def _good_reply() -> dict:
    return images_reply(_good_b64())


class Hx:
    def __init__(self, mp, tmp_path, *, tripwire=True):
        self.mp = mp
        self.tmp = tmp_path
        self.llm: FakeLLM | None = None
        self.transport = None
        self.make_llm_calls = 0
        self.make_transport_calls = 0
        mp.chdir(tmp_path)
        mp.delenv(MODEL_VAR, raising=False)
        if tripwire:

            def _tripped(self, *a, **k):
                raise AssertionError("tripwire: the real UrllibTransport.post_json was reached")

            mp.setattr(providers.UrllibTransport, "post_json", _tripped)

    @property
    def out(self):
        return self.tmp / "out"

    def run(
        self,
        args,
        *replies,
        llm_script=({},),
        anthropic="dummy-key",
        openrouter="dummy-or-key",
        transport=None,
    ):
        for var, value in (("ANTHROPIC_API_KEY", anthropic), ("OPENROUTER_API_KEY", openrouter)):
            if value is None:
                self.mp.delenv(var, raising=False)
            else:
                self.mp.setenv(var, value)
        self.llm = FakeLLM(*llm_script)
        self.transport = transport if transport is not None else FakeTransport(*replies)

        def _make_llm():
            self.make_llm_calls += 1
            return self.llm

        def _make_transport():
            self.make_transport_calls += 1
            return self.transport

        self.mp.setattr(cli, "make_llm", _make_llm)
        self.mp.setattr(cli, "make_transport", _make_transport)
        return runner.invoke(app, args)

    def gen(self, *extra, reply=None, name="x", **kw):
        args = ["generate", "a red apple", "--backend", "openrouter", "-o", f"out/{name}", *extra]
        if not any(f in ("--size", "--width") for f in extra):
            args += ["--size", "16"]
        return self.run(args, reply if reply is not None else _good_reply(), **kw)

    def zero_call(self):
        return (
            self.make_llm_calls == 0
            and self.make_transport_calls == 0
            and self.llm.call_count == 0
            and self.transport.call_count == 0
        )


@pytest.fixture
def hx(monkeypatch, tmp_path):
    return Hx(monkeypatch, tmp_path)


def _size(path):
    with Image.open(path) as im:
        return im.size


def _rgba(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"))


def _lines(result):
    return [ln for ln in result.stderr.splitlines() if ln.strip()]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_refusal(hx, result, code, *, no_out=True):
    assert result.exit_code == code, (result.stdout, result.stderr)
    lines = _lines(result)
    assert len(lines) == 1 and lines[0].startswith("pixelgen: error:"), result.stderr
    assert hx.zero_call()
    if no_out:
        assert not list(hx.out.glob("x.*")) if hx.out.is_dir() else True
    return lines[0]


# --- AC6 ---------------------------------------------------------------------------------------


def test_openrouter_generate_end_to_end(hx):
    result = hx.gen("--palette", "pico8")
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert {p.name for p in hx.out.iterdir()} == {"x.png", "x@8x.png", "x.json"}
    png = _rgba(hx.out / "x.png")
    assert png.shape == (16, 16, 4)
    palette = load_palette("pico8")
    opaque = png[..., 3] == 255
    assert opaque.any()
    colours = {tuple(c) for c in png[opaque][:, :3].tolist()}
    assert colours <= {tuple(c) for c in palette.tolist()}
    assert _size(hx.out / "x@8x.png") == (128, 128)
    sc = _json(hx.out / "x.json")
    assert sc["schema_version"] == 2
    assert sc["backend"] == "openrouter" and sc["provider"] == "openrouter"
    assert sc["provider_model"] == "openai/gpt-image-2" and sc["model"] == "fake-model"
    assert hx.llm.call_count == 1
    assert hx.make_transport_calls == 1 and hx.transport.call_count == 1


# --- AC7 ---------------------------------------------------------------------------------------


def test_openrouter_usage_record_and_line(hx):
    result = hx.gen()
    assert result.exit_code == 0, (result.stdout, result.stderr)
    sc = _json(hx.out / "x.json")
    assert sc["usage"][0]["step"] == "infer_spec"
    assert sc["usage"][1] == {
        "step": "generate",
        "provider": "openrouter",
        "model": "openai/gpt-image-2",
        "unit": "images",
        "calls": 1,
        "quantity": 1,
    }
    assert sc["token_usage"] == {"input_tokens": 10, "output_tokens": 20, "calls": 2}
    last = (hx.tmp / "pixelgen-home" / "last.json").read_text(encoding="utf-8")
    blobs = [(hx.out / "x.json").read_text(encoding="utf-8"), last, result.stdout, result.stderr]
    for blob in blobs:
        assert str(TOKEN_SENTINEL) not in blob
        assert str(COST_SENTINEL) not in blob

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert not [k for k in keys(sc) if re.search(r"(?i)cost|price|usd", k)]
    assert _lines(result)[-1] == "pixelgen: usage: " + json.dumps(
        sc["usage"], ensure_ascii=True, separators=(", ", ": ")
    )
    other = "black-forest-labs/flux.2-klein-4b"
    hx.mp.setenv(MODEL_VAR, other)
    result = hx.run(
        ["generate", "a red apple", "--backend", "openrouter", "-o", "out/y", "--size", "16"],
        _good_reply(),
    )
    assert result.exit_code == 0, result.stderr
    assert (
        _json(hx.out / "y.json")["provider_model"] == other == hx.transport.calls[0].body["model"]
    )


# --- AC8 ---------------------------------------------------------------------------------------

SENTINEL_MODEL = "sk-or-v1-pxg42-" + uuid.uuid4().hex


def test_openrouter_refused_before_spend(hx):
    line = _assert_refusal(hx, hx.gen(openrouter=None), 1)
    assert line == MISSING_OR
    line = _assert_refusal(hx, hx.gen(openrouter="   "), 1)
    assert line == MISSING_OR
    line = _assert_refusal(hx, hx.gen(anthropic=None), 1)
    assert "ANTHROPIC_API_KEY" in line
    hx.mp.setenv(MODEL_VAR, SENTINEL_MODEL)
    line = _assert_refusal(hx, hx.gen(anthropic=None, openrouter=None), 2)
    assert MODEL_VAR in line and SENTINEL_MODEL not in line
    hx.mp.delenv(MODEL_VAR)
    (hx.tmp / "out").write_text("not a directory", encoding="utf-8")
    line = _assert_refusal(hx, hx.gen(), 1, no_out=False)
    assert "/out'" in line and "_KEY" not in line
    assert (hx.tmp / "out").read_text(encoding="utf-8") == "not a directory"


# --- AC9 ---------------------------------------------------------------------------------------

BAD_REPLIES = [
    ({}, "had no image data"),
    ({"data": []}, "had no image data"),
    ({"data": "x"}, "had no image data"),
    ({"data": [{"media_type": "image/png"}]}, "had no b64_json image"),
    ({"data": [{"b64_json": 7}]}, "had no b64_json image"),
    ({"data": [{"b64_json": "@@@"}]}, "was not valid base64"),
    ({"data": [{"b64_json": jpeg_b64(sprite_on_backdrop(3, 32))}]}, "was not a PNG"),
    (
        {
            "data": [
                {"b64_json": base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(20)).decode("ascii")}
            ]
        },
        "could not be decoded",
    ),
    ({"data": [{"b64_json": png_b64(np.zeros((8, 4097, 3), dtype=np.uint8))}]}, "is 4097x8"),
]


@pytest.mark.parametrize(
    "reply,needle", BAD_REPLIES, ids=[f"{i}-{n}" for i, (_, n) in enumerate(BAD_REPLIES)]
)
@pytest.mark.parametrize("aseprite", [False, True], ids=["png", "aseprite"])
def test_openrouter_bad_response_fails_safe(hx, reply, needle, aseprite):
    extra = ["--aseprite"] if aseprite else []
    hx.mp.setenv("ANTHROPIC_API_KEY", "dummy-key")
    result = hx.run(
        ["generate", "a red apple", "--backend", "openrouter", "-o", "out/x", "--size", "16"]
        + extra,
        reply,
        reply,
        reply,
    )
    assert result.exit_code == 1, (result.stdout, result.stderr)
    lines = _lines(result)
    assert len(lines) == 1 and lines[0].startswith("pixelgen: error:") and needle in lines[0]
    assert "Traceback" not in result.stderr
    assert not list(hx.out.glob("x*")) if hx.out.is_dir() else True
    assert not (hx.tmp / "pixelgen-home" / "last.json").exists()
    assert hx.transport.call_count == 1 and hx.llm.call_count == 1


def test_openrouter_transport_error_fails_safe(hx):
    msg = "the request to openrouter.ai failed (HTTP 502); gave up after 3 attempts"
    result = hx.gen(reply=TransportError(msg))
    assert result.exit_code == 1, (result.stdout, result.stderr)
    assert _lines(result) == [f"pixelgen: error: {msg}"]
    assert not list(hx.out.glob("x*")) if hx.out.is_dir() else True
    assert not (hx.tmp / "pixelgen-home" / "last.json").exists()
    assert hx.transport.call_count == 1 and hx.llm.call_count == 1


# --- AC10 --------------------------------------------------------------------------------------


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _Opener:
    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def open(self, req, timeout=None):
        self.requests.append(req)
        item = self.script[len(self.requests) - 1]
        if isinstance(item, int):
            raise urllib.error.HTTPError(req.full_url, item, "err", {}, None)
        return _Resp(json.dumps(item).encode("utf-8"))


def _urllib_run(hx, *script, **kw):
    opener = _Opener(*script)
    transport = providers.UrllibTransport(opener=opener, sleep=lambda s: None)
    result = hx.run(
        ["generate", "a red apple", "--backend", "openrouter", "-o", "out/x", "--size", "16"],
        transport=transport,
        **kw,
    )
    return result, opener


def test_openrouter_attempt_budget_through_urllib_transport(monkeypatch, tmp_path):
    hx = Hx(monkeypatch, tmp_path, tripwire=False)
    result, opener = _urllib_run(hx, 502, 502, 502)
    assert result.exit_code == 1
    assert "HTTP 502" in result.stderr and "gave up after 3 attempts" in result.stderr
    assert len(opener.requests) == 3
    result, opener = _urllib_run(hx, 503, _good_reply())
    assert result.exit_code == 0, result.stderr
    assert len(opener.requests) == 2
    bad = {"data": [{"b64_json": "@@@"}]}
    result, opener = _urllib_run(hx, bad, bad, bad)
    assert result.exit_code == 1 and "was not valid base64" in result.stderr
    assert len(opener.requests) == 1
    result, opener = _urllib_run(hx, 401, 401, 401)
    assert result.exit_code == 1 and "check the API key" in result.stderr
    assert len(opener.requests) == 1
    # callee side: the request the transport actually built
    result, opener = _urllib_run(hx, _good_reply())
    assert result.exit_code == 0, result.stderr
    req = opener.requests[0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://openrouter.ai/api/v1/images"
    assert req.get_header("Authorization") == "Bearer dummy-or-key"
    inferred = ArtSpec.model_validate(_json(hx.out / "x.json")["spec"])
    assert json.loads(req.data) == {
        "model": "openai/gpt-image-2",
        "prompt": build_prompt(inferred),
        "size": "1024x1024",
        "quality": "low",
        "output_format": "png",
        "n": 1,
    }


# --- AC11 --------------------------------------------------------------------------------------


def test_refine_from_openrouter_output_never_calls_provider(hx):
    assert hx.gen().exit_code == 0
    first_transport = hx.transport
    made = hx.make_transport_calls
    grid = _json(hx.out / "x.json")["grid"]
    grid[0][0] = (grid[0][0] + 1) % 16
    hx.mp.setenv(MODEL_VAR, "Not An Id")
    result = hx.run(
        ["refine", "add eyes", "--from", "out/x.json"],
        llm_script=({"spec": {}, "grid": grid},),
        openrouter=None,
    )
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert (hx.out / "x-v2.png").is_file()
    v2 = _json(hx.out / "x-v2.json")
    assert v2["backend"] == "llm-grid"
    assert [r["step"] for r in v2["usage"]] == ["refine"]
    assert v2["usage"][0]["provider"] == "anthropic"
    assert hx.make_transport_calls == made
    assert first_transport.call_count == 1


# --- AC12 --------------------------------------------------------------------------------------


def test_openrouter_key_never_shown(monkeypatch, tmp_path, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    hx = Hx(monkeypatch, tmp_path, tripwire=False)  # one real-transport row below
    k1 = "pxg42-sentinel-" + uuid.uuid4().hex
    k2 = "pxg42-sentinel-" + uuid.uuid4().hex
    outputs = []

    def collect(result):
        outputs.extend([result.stdout, result.stderr])

    ok = hx.gen(anthropic=k1, openrouter=k2)
    assert ok.exit_code == 0, ok.stderr
    assert hx.transport.calls[0].headers["Authorization"] == f"Bearer {k2}"  # positive control
    collect(ok)
    not_png = hx.gen(
        reply={"data": [{"b64_json": jpeg_b64(sprite_on_backdrop(3, 32))}]},
        name="y",
        anthropic=k1,
        openrouter=k2,
    )
    assert not_png.exit_code == 1 and "was not a PNG" in not_png.stderr
    collect(not_png)
    monkeypatch.setenv(MODEL_VAR, SENTINEL_MODEL)
    refused = hx.gen(name="z", anthropic=k1, openrouter=k2)
    assert refused.exit_code == 2
    collect(refused)
    monkeypatch.delenv(MODEL_VAR)
    result, opener = _urllib_run(hx, 401, anthropic=k1, openrouter=k2)
    assert result.exit_code == 1 and "check the API key" in result.stderr
    assert opener.requests[0].get_header("Authorization") == f"Bearer {k2}"
    collect(result)
    blobs = "\n".join(outputs) + caplog.text
    for k in (k1, k2):
        assert k not in blobs
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert k.encode() not in path.read_bytes(), path


# --- AC13 --------------------------------------------------------------------------------------

ROWS = [
    ([], (32, 32)),
    (["--dither", "floyd-steinberg"], (32, 32)),
    (["--dither", "bayer"], (32, 32)),
    (["--outline"], (32, 32)),
    (["--no-outline"], (32, 32)),
    (["--background", "solid"], (32, 32)),
    (["--background", "transparent"], (32, 32)),
    (["--size", "48"], (48, 48)),
    (["--palette", "gameboy"], (32, 32)),
    (["--width", "40", "--height", "24"], (40, 24)),
]


def _ase_run(hx, flags, name="x"):
    args = ["generate", "a red apple", "--backend", "openrouter", "--aseprite", "-o"]
    args.append(f"out/{name}")
    if not any(f in ("--size", "--width") for f in flags):
        args += ["--size", "32"]
    return hx.run(args + flags, _good_reply())


@pytest.mark.parametrize("flags,size", ROWS, ids=[" ".join(r[0]) or "default" for r in ROWS])
def test_openrouter_png_matches_aseprite(hx, flags, size):
    result = _ase_run(hx, flags)
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert oracle_main([str(hx.out / "x.aseprite"), "--png", str(hx.out / "x.png")]) == 0
    assert _size(hx.out / "x.png") == size


def test_openrouter_png_matches_aseprite_non_vacuity(hx):
    pngs = {}
    for name, flags in (
        ("default", []),
        ("nooutline", ["--no-outline"]),
        ("solid", ["--background", "solid"]),
        ("bayer", ["--dither", "bayer"]),
        ("gameboy", ["--palette", "gameboy"]),
    ):
        result = _ase_run(hx, flags, name)
        assert result.exit_code == 0, (name, result.stderr)
        pngs[name] = (hx.out / f"{name}.png").read_bytes()
    default = _rgba(hx.out / "default.png")
    solid = _rgba(hx.out / "solid.png")
    assert (default[..., 3] == 0).sum() > 0
    assert (solid[..., 3] == 0).sum() == 0
    for name in ("nooutline", "solid", "bayer", "gameboy"):
        assert pngs[name] != pngs["default"], name
    pal = load_palette("pico8")
    assert grid_to_rgba(rgba_to_grid(default, pal), pal).shape == default.shape


# --- AC16 --------------------------------------------------------------------------------------


def test_openrouter_refusal_under_ci_colour_forcing(hx):
    hx.mp.setenv("GITHUB_ACTIONS", "true")
    hx.mp.setenv("FORCE_COLOR", "1")
    hx.mp.delenv("NO_COLOR", raising=False)
    result = hx.gen(openrouter=None)
    assert result.exit_code == 1
    assert result.stderr == MISSING_OR + "\n"
    assert "\x1b[" not in result.stderr
    assert os.environ.get("NO_COLOR") is None
