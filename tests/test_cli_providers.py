"""CLI tests for the provider seam (PXG-37, AC10-AC16).

In-process ``CliRunner`` only, ``FakeLLM`` / ``FakeTransport`` through the ``cli.make_llm`` /
``cli.make_transport`` seams, offline. New symbols are fetched lazily so a tree without the
seam fails these tests by assertion, not at collection.
"""

import importlib
import io
import logging
import os
import urllib.error
import uuid
from pathlib import Path

import numpy as np
import pytest
import typer.main
from PIL import Image
from typer.testing import CliRunner

import pixelgen.cli as cli
from pixelgen.cli import app
from pixelgen.export import load_sidecar
from pixelgen.palettes import load_palette

runner = CliRunner()

L1 = "pixelgen: error: ANTHROPIC_API_KEY is not set. Export your Anthropic API key and try again.\n"
L3 = "pixelgen: error: unknown backend 'nope'; available: diffusion, llm-grid, openrouter\n"
L4 = (
    "pixelgen: error: The diffusion backend is not implemented in this version of pixelgen; "
    "use --backend llm-grid (the default).\n"
)
FAKE_MISSING = (
    "pixelgen: error: FAKEIMG_TEST_KEY is not set; the 'fake-image' backend needs it. "
    "Export it in your shell and try again.\n"
)


class _Lazy:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        try:
            module = importlib.import_module(self._name)
        except ImportError as err:
            raise AssertionError(f"module {self._name} is missing: {err}") from None
        assert hasattr(module, attr), f"{self._name}.{attr} is missing"
        return getattr(module, attr)


providers = _Lazy("pixelgen.providers")
fakes = _Lazy("fakes")
pipeline = _Lazy("pixelgen.pipeline")


def _grid(seed=0, side=16, colours=16):
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5 + seed) % colours
            for c in range(side)
        ]
        for r in range(side)
    ]


class _NoTransport:
    """Stand-in for rows that never script a transport (a call would be a bug)."""

    call_count = 0


class Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.llm = None
        self.transport = None
        self.make_llm_calls = 0
        self.make_transport_calls = 0
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("FAKEIMG_TEST_KEY", raising=False)
        monkeypatch.delenv("FAKEIMG_TEST_MODEL", raising=False)

    @property
    def out(self) -> Path:
        return self.tmp / "out"

    def setenv(self, anthropic="dummy-key", fake="dummy-fake-key"):
        for name, value in (("ANTHROPIC_API_KEY", anthropic), ("FAKEIMG_TEST_KEY", fake)):
            if value is None:
                self.mp.delenv(name, raising=False)
            else:
                self.mp.setenv(name, value)

    def invoke(self, args, *, llm_script=(), transport=None, anthropic="dummy-key", fake=None):
        self.setenv(anthropic, fake)
        self.llm = fakes.FakeLLM(*llm_script)
        self.transport = transport if transport is not None else _NoTransport()

        def fake_make_llm():
            self.make_llm_calls += 1
            return self.llm

        def fake_make_transport():
            self.make_transport_calls += 1
            return self.transport

        self.mp.setattr(cli, "make_llm", fake_make_llm)
        self.mp.setattr(cli, "make_transport", fake_make_transport, raising=False)
        return runner.invoke(app, args)

    def fake_generate(self, *extra, fake="dummy-fake-key", anthropic="dummy-key", reply_seed=3):
        return self.invoke(
            ["generate", "a red apple", "--size", "16", "--palette", "pico8",
             "--backend", "fake-image", "-o", "out/x", *extra],
            llm_script=({},),
            transport=fakes.FakeTransport(fakes.fake_image_reply(64, 64, seed=reply_seed)),
            fake=fake,
            anthropic=anthropic,
        )  # fmt: skip

    def assert_zero_call_refusal(self, result, code, tokens, absent=("_KEY",), check_out=True):
        assert result.exit_code == code, (result.stdout, result.stderr)
        lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
        assert len(lines) == 1 and lines[0].startswith("pixelgen: error:"), lines
        for token in tokens:
            assert token in lines[0]
        for token in absent:
            assert token not in lines[0]
        assert self.make_llm_calls == 0
        assert self.llm.call_count == 0
        assert self.make_transport_calls == 0
        assert self.transport.call_count == 0
        if check_out:
            assert not self.out.exists()


@pytest.fixture
def ch(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


@pytest.fixture
def fake_registered(monkeypatch):
    fakes.register_fake_provider(monkeypatch)


# --- AC10 -----------------------------------------------------------------------------------


def test_fake_provider_generate_end_to_end(ch, fake_registered, monkeypatch):
    result = ch.fake_generate()
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert {p.name for p in ch.out.iterdir()} == {"x.png", "x@8x.png", "x.json"}
    png = np.array(Image.open(ch.out / "x.png").convert("RGBA"))
    assert png.shape == (16, 16, 4)
    pal = {tuple(int(v) for v in c[:3]) for c in load_palette("pico8")}
    opaque = png[png[..., 3] == 255]
    assert len(opaque) > 0
    assert {tuple(int(v) for v in px[:3]) for px in opaque} <= pal
    sc = load_sidecar(ch.out / "x.json")
    assert (sc.backend, sc.model, sc.schema_version) == ("fake-image", "fake-model", 2)
    assert ch.llm.call_count == 1
    assert ch.make_transport_calls == 1 and ch.transport.call_count == 1
    call = ch.transport.calls[0]
    assert call.url == "https://fake-image.invalid/v1/images"
    assert call.headers["Authorization"] == "Bearer dummy-fake-key"
    assert call.body["model"] == "fake-image-1"
    assert call.timeout == providers.DEFAULT_TIMEOUT_SECONDS

    monkeypatch.setenv("FAKEIMG_TEST_MODEL", "fake-image-2")
    result = ch.invoke(
        ["generate", "a red apple", "--size", "16", "--palette", "pico8",
         "--backend", "fake-image", "-o", "out/y"],
        llm_script=({},),
        transport=fakes.FakeTransport(fakes.fake_image_reply(64, 64, seed=3)),
        fake="dummy-fake-key",
    )  # fmt: skip
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert ch.transport.calls[0].body["model"] == "fake-image-2"


# --- AC11 -----------------------------------------------------------------------------------


def test_provider_key_check_refuses_before_spend(ch, fake_registered):
    r = ch.fake_generate(fake=None)
    ch.assert_zero_call_refusal(r, 1, ["FAKEIMG_TEST_KEY"], absent=["ANTHROPIC_API_KEY"])
    assert r.stderr == FAKE_MISSING
    r = ch.fake_generate(fake="   ")
    ch.assert_zero_call_refusal(r, 1, ["FAKEIMG_TEST_KEY"], absent=["ANTHROPIC_API_KEY"])
    assert r.stderr == FAKE_MISSING
    r = ch.fake_generate(anthropic=None)
    ch.assert_zero_call_refusal(r, 1, ["ANTHROPIC_API_KEY"], absent=[])
    assert r.stderr == L1
    r = ch.fake_generate(anthropic=None, fake=None)
    ch.assert_zero_call_refusal(r, 1, ["ANTHROPIC_API_KEY"], absent=["FAKEIMG"])


@pytest.mark.parametrize("bad", ["\rx", "\u2019x", "\x7fx", "\u00e9x"])
def test_provider_key_check_refuses_control_character_key(ch, fake_registered, bad):
    s = "pxg37-sentinel-" + uuid.uuid4().hex
    before = _last_json(ch)
    r = ch.fake_generate(fake=s + bad)
    ch.assert_zero_call_refusal(
        r, 1, ["FAKEIMG_TEST_KEY", "not printable ASCII"], absent=["ANTHROPIC_API_KEY"]
    )
    assert s not in r.stdout + r.stderr and "Traceback" not in r.stderr
    assert _last_json(ch) == before


def _last_json(ch):
    path = Path(os.environ["PIXELGEN_HOME"]) / "last.json"
    return path.read_bytes() if path.exists() else None


@pytest.mark.parametrize("bad", ["\rx", "\u2019x"])
def test_provider_key_check_refuses_malformed_anthropic_key(ch, bad):
    s = "pxg37-sentinel-" + uuid.uuid4().hex
    before = _last_json(ch)
    r = ch.invoke(["generate", "x", "--size", "16"], anthropic=s + bad)
    ch.assert_zero_call_refusal(r, 1, ["ANTHROPIC_API_KEY", "not printable ASCII"], absent=[])
    assert s not in r.stdout + r.stderr and "Traceback" not in r.stderr
    assert _last_json(ch) == before

    r = ch.invoke(["refine", "x", "--from", "nowhere.json"], anthropic=s + bad)
    assert r.exit_code != 0 and s not in r.stdout + r.stderr  # source error precedes the key check
    # a real source: seed one through a good run, then refuse the refine
    ok = ch.invoke(
        ["generate", "x", "--size", "16", "-o", "out/x"], llm_script=({}, {"grid": _grid(1)})
    )
    assert ok.exit_code == 0, (ok.stdout, ok.stderr)
    calls, llm_calls, last = ch.make_llm_calls, ch.llm.call_count, _last_json(ch)
    r = ch.invoke(["refine", "more", "--from", "out/x.json"], anthropic=s + bad)
    assert r.exit_code == 1
    lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1 and "ANTHROPIC_API_KEY" in lines[0] and "not printable ASCII" in lines[0]
    assert s not in r.stdout + r.stderr and "Traceback" not in r.stderr
    assert ch.make_llm_calls == calls and ch.llm.call_count == 0 and ch.make_transport_calls == 0
    assert llm_calls == 2 and _last_json(ch) == last


def test_provider_key_check_refuses_keeps_whitespace_padded_keys_working(ch, fake_registered):
    r = ch.fake_generate(fake="  dummy-fake-key \n", anthropic="\tdummy-key  ")
    assert r.exit_code == 0, (r.stdout, r.stderr)
    assert ch.transport.calls[0].headers["Authorization"] == "Bearer dummy-fake-key"


# --- AC12 -----------------------------------------------------------------------------------


def test_unknown_or_unavailable_provider_refused_before_key_check(ch, monkeypatch, fake_registered):
    def gen(backend, *extra):
        return ch.invoke(
            ["generate", "a red apple", "--size", "16", "--backend", backend, *extra],
            anthropic=None,
            fake=None,
        )

    r = gen("nope")
    ch.assert_zero_call_refusal(r, 2, ["unknown backend 'nope'", "fake-image"])
    r = gen("diffusion")
    ch.assert_zero_call_refusal(r, 1, ["diffusion backend is not implemented"])
    fakes.register_fake_provider(
        monkeypatch, name="fake-down", target="fakes:FakeUnavailableGenerator"
    )
    r = gen("fake-down")
    ch.assert_zero_call_refusal(r, 1, ["fake-down"])
    (ch.tmp / "out").write_text("a file", encoding="utf-8")
    r = gen("fake-image", "-o", "out/x")
    ch.assert_zero_call_refusal(r, 1, ["out"], check_out=False)


# --- AC13 -----------------------------------------------------------------------------------


def test_refine_needs_only_the_spec_llm_key(ch, fake_registered):
    assert ch.fake_generate().exit_code == 0
    source_transport = ch.transport
    transports_before = ch.make_transport_calls
    r = ch.invoke(
        ["refine", "add eyes", "--from", "out/x.json"],
        llm_script=({"spec": {}, "grid": _grid(7)},),
        transport=source_transport,
        fake=None,
    )
    assert r.exit_code == 0, (r.stdout, r.stderr)
    assert (ch.out / "x-v2.png").is_file()
    assert load_sidecar(ch.out / "x-v2.json").backend == "llm-grid"
    assert ch.make_transport_calls == transports_before
    assert source_transport.call_count == 1

    llm_before = ch.make_llm_calls
    r = ch.invoke(
        ["refine", "more", "--from", "out/x.json"],
        llm_script=({"spec": {}, "grid": _grid(8)},),
        anthropic=None,
        fake="dummy-fake-key",
    )
    assert r.exit_code == 1
    assert r.stderr == L1
    assert ch.make_llm_calls == llm_before
    assert pipeline.REFINE_BACKEND == "llm-grid"


# --- AC14 -----------------------------------------------------------------------------------

GENERATE_OPTIONS = {
    "--size", "--width", "--height", "--palette", "--preset", "--dither", "--outline",
    "--no-outline", "--background", "--view", "--seed", "--scale", "-o", "--output",
    "--backend", "--show-spec", "--aseprite", "--no-aseprite",
}  # fmt: skip


def test_llm_grid_key_check_and_messages_frozen(ch):
    r = ch.invoke(["generate", "x", "--size", "16"], anthropic=None)
    assert (r.exit_code, r.stderr) == (1, L1)
    r = ch.invoke(["generate", "x", "--size", "16", "--backend", "nope"], anthropic=None)
    assert (r.exit_code, r.stderr) == (2, L3)
    r = ch.invoke(["generate", "x", "--size", "16", "--backend", "diffusion"], anthropic=None)
    assert (r.exit_code, r.stderr) == (1, L4)

    r = ch.invoke(
        ["generate", "x", "--size", "16", "-o", "out/x"], llm_script=({}, {"grid": _grid(1)})
    )
    assert r.exit_code == 0, (r.stdout, r.stderr)
    assert ch.make_transport_calls == 0
    r = ch.invoke(["refine", "x", "--from", "out/x.json"], anthropic=None)
    assert (r.exit_code, r.stderr) == (1, L1)

    command = typer.main.get_command(app).commands["generate"]
    declared = {
        opt for p in command.params for opt in (*p.opts, *p.secondary_opts) if opt.startswith("-")
    }
    assert declared == GENERATE_OPTIONS
    backend = {p.name: p for p in command.params}["backend"]
    assert backend.help == "Generator backend."
    assert backend.default == "llm-grid"


# --- AC15 -----------------------------------------------------------------------------------


def _tree_bytes(root: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())


def test_no_key_value_in_any_output(ch, fake_registered, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    s1 = "pxg37-sentinel-" + uuid.uuid4().hex
    s2 = "pxg37-sentinel-" + uuid.uuid4().hex

    def check(result):
        haystack = (result.stdout + result.stderr + caplog.text).encode() + _tree_bytes(tmp_path)
        for s in (s1, s2):
            assert s.encode() not in haystack

    # (a) success through the fake transport
    r = ch.fake_generate(anthropic=s1, fake=s2)
    assert r.exit_code == 0, (r.stdout, r.stderr)
    assert ch.transport.calls[0].headers["Authorization"] == f"Bearer {s2}"
    check(r)

    # (b) a real UrllibTransport whose opener answers 401 with the key everywhere
    class Opener:
        requests: list = []

        def open(self, req, timeout=None):
            self.requests.append(req)
            raise urllib.error.HTTPError(
                "https://fake-image.invalid/v1/images",
                401,
                s2,
                {"X-Leak": s2},
                io.BytesIO(s2.encode()),
            )

    opener = Opener()
    real = providers.UrllibTransport(opener=opener, sleep=lambda s: None)
    r = ch.invoke(
        ["generate", "a red apple", "--size", "16", "--backend", "fake-image", "-o", "out/z"],
        llm_script=({},),
        transport=real,
        anthropic=s1,
        fake=s2,
    )
    assert r.exit_code == 1
    lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1 and "HTTP 401" in lines[0]
    assert opener.requests[0].get_header("Authorization") == f"Bearer {s2}"
    check(r)

    # (c) missing provider key with the Anthropic sentinel set
    r = ch.fake_generate(anthropic=s1, fake=None)
    assert r.exit_code == 1
    check(r)

    # (d) llm-grid success
    r = ch.invoke(
        ["generate", "x", "--size", "16", "-o", "out/g"],
        llm_script=({}, {"grid": _grid(1)}),
        anthropic=s1,
        fake=s2,
    )
    assert r.exit_code == 0, (r.stdout, r.stderr)
    check(r)


# --- AC16 -----------------------------------------------------------------------------------


def test_provider_refusal_under_ci_colour_forcing(ch, fake_registered, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("FORCE_COLOR", "1")
    r = ch.fake_generate(fake=None)
    assert r.exit_code == 1
    assert r.stderr == FAKE_MISSING
    assert "\x1b[" not in r.stderr
