"""Unit tests for the provider seam (PXG-37, AC1-AC9).

New symbols are fetched lazily (``_Lazy``) so these tests fail by assertion, not by an
ImportError at collection, against a tree without the seam. Offline: the transport is
exercised only through injected openers.
"""

import ast
import dataclasses
import http.client
import importlib
import io
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

import pixelgen.backends.base as base
from pixelgen import llm
from pixelgen.errors import BackendUnavailableError, MissingAPIKeyError, PixelgenError

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture(autouse=True)
def _clean_fake_env(monkeypatch):
    monkeypatch.delenv("FAKEIMG_TEST_KEY", raising=False)
    monkeypatch.delenv("FAKEIMG_TEST_MODEL", raising=False)


class _Resp:
    def __init__(self, raw):
        self._raw = raw

    def read(self, amt=None):
        return self._raw if amt is None else self._raw[:amt]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Opener:
    """Recording opener: each item is bytes (a 200 body) or an exception to raise."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []
        self.timeouts = []

    def open(self, req, timeout=None):
        self.requests.append(req)
        self.timeouts.append(timeout)
        item = self.script[min(len(self.requests), len(self.script)) - 1]
        if isinstance(item, BaseException):
            raise item
        return _Resp(item)


def _http(code, *, reason="r", body=b"", headers=None, url="https://h.invalid/v1/x"):
    return urllib.error.HTTPError(url, code, reason, headers or {}, io.BytesIO(body))


def _transport(opener, sleeps=None):
    sleeps = [] if sleeps is None else sleeps
    return providers.UrllibTransport(opener=opener, sleep=sleeps.append), sleeps


URL = "https://h.invalid/v1/x"


def _post(transport, url=URL):
    return transport.post_json(
        url, headers={"Authorization": "Bearer t"}, body={"k": 1}, timeout=5.0
    )


# --- AC1 ------------------------------------------------------------------------------------


def test_provider_config_shape_and_validation():
    cfg = providers.ProviderConfig
    assert [f.name for f in dataclasses.fields(cfg)] == [
        "name", "key_env", "default_model", "model_env", "base_url", "usage_units",
    ]  # fmt: skip
    good = cfg("p", "P_KEY", "m", "P_MODEL", "https://x.invalid", "images")
    with pytest.raises(dataclasses.FrozenInstanceError):
        good.name = "q"  # type: ignore[misc]
    bad = [
        {"key_env": "PIXELGEN_FOO_API_KEY"},
        {"key_env": "lower"},
        {"usage_units": "dollars"},
        {"base_url": "http://example.invalid"},
        {"base_url": "file:///x"},
        {"name": "Bad Name"},
    ]
    for override in bad:
        kwargs = dict(
            name="p", key_env="P_KEY", default_model="m", model_env="P_MODEL",
            base_url="https://x.invalid", usage_units="images",
        )  # fmt: skip
        kwargs.update(override)
        with pytest.raises(ValueError):
            cfg(**kwargs)
    assert fakes.FAKE_PROVIDER.name == "fake-image"
    cfg("p", None, None, None, "http://127.0.0.1:7860", "calls")


def test_anthropic_config_matches_llm():
    p = providers
    assert p.ANTHROPIC.key_env == "ANTHROPIC_API_KEY"
    assert p.ANTHROPIC.model_env == "PIXELGEN_MODEL"
    assert p.ANTHROPIC.default_model == llm.DEFAULT_MODEL
    assert p.SPEC_LLM_PROVIDER is p.ANTHROPIC
    for env in ({}, {"PIXELGEN_MODEL": "m"}, {"PIXELGEN_MODEL": "  "}):
        assert p.resolve_model(p.ANTHROPIC, env) == llm.resolve_model(env)
    assert p.resolve_key(p.ANTHROPIC, {"ANTHROPIC_API_KEY": " k "}) == "k"
    with pytest.raises(MissingAPIKeyError) as exc:
        p.resolve_key(p.ANTHROPIC, {})
    assert str(exc.value) == MissingAPIKeyError.default_message
    assert p.resolve_key(p.DIFFUSION_STUB, {}) is None


# --- AC2 ------------------------------------------------------------------------------------


def _base(attr):
    assert hasattr(base, attr), f"pixelgen.backends.base.{attr} is missing"
    return getattr(base, attr)


def test_registry_carries_provider_config():
    provider_for, uses_transport = _base("provider_for"), _base("uses_transport")
    assert provider_for("llm-grid") is providers.ANTHROPIC
    assert provider_for("diffusion") is providers.DIFFUSION_STUB
    assert uses_transport("llm-grid") is False
    assert uses_transport("diffusion") is False
    assert base.available_backends() == ["diffusion", "llm-grid", "openrouter"]
    with pytest.raises(BackendUnavailableError) as exc:
        provider_for("nope")
    assert "unknown backend 'nope'; available: diffusion, llm-grid, openrouter" in str(exc.value)


def test_required_keys(monkeypatch):
    assert providers.required_keys("llm-grid") == ("ANTHROPIC_API_KEY",)
    assert providers.required_keys("diffusion") == ("ANTHROPIC_API_KEY",)
    fakes.register_fake_provider(monkeypatch)
    assert providers.required_keys("fake-image") == ("ANTHROPIC_API_KEY", "FAKEIMG_TEST_KEY")
    for key_env in ("ANTHROPIC_API_KEY", None):
        cfg = providers.ProviderConfig("dup", key_env, None, None, None, "images")
        fakes.register_fake_provider(monkeypatch, name="dup", provider=cfg)
        assert providers.required_keys("dup") == ("ANTHROPIC_API_KEY",)
    with pytest.raises(BackendUnavailableError):
        providers.required_keys("nope")


# --- AC3 ------------------------------------------------------------------------------------


def test_transport_protocol_and_fake_transport():
    assert isinstance(providers.UrllibTransport(), providers.Transport)
    assert isinstance(fakes.FakeTransport(), providers.Transport)
    ft = fakes.FakeTransport({"a": 1}, providers.TransportError("x"))
    first = ft.post_json("u", headers={"h": "v"}, body={"b": 1}, timeout=3.0)
    assert first == {"a": 1}
    first["a"] = 2
    with pytest.raises(providers.TransportError):
        ft.post_json("u2", headers={}, body={}, timeout=1.0)
    assert ft.call_count == 2
    c = ft.calls[0]
    assert (c.url, c.headers, c.body, c.timeout) == ("u", {"h": "v"}, {"b": 1}, 3.0)
    with pytest.raises(AssertionError):
        ft.post_json("u3", headers={}, body={}, timeout=1.0)
    again = fakes.FakeTransport({"a": 1})
    again.post_json("u", headers={}, body={}, timeout=1.0)["a"] = 9
    assert again._script[0] == {"a": 1}


# --- AC4 ------------------------------------------------------------------------------------


def _assert_request_shape(opener):
    req = opener.requests[0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://h.invalid/v1/x"
    assert json.loads(req.data) == {"k": [1, 2]}
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("Authorization") == "Bearer t"
    assert opener.timeouts == [7.5]


def test_urllib_transport_request_shape(monkeypatch):
    opener = Opener(b'{"ok": true}')
    t = providers.UrllibTransport(opener=opener)
    out = t.post_json(
        "https://h.invalid/v1/x",
        headers={"Authorization": "Bearer t"},
        body={"k": [1, 2]},
        timeout=7.5,
    )
    assert out == {"ok": True}
    _assert_request_shape(opener)

    second = Opener(b'{"ok": true}')
    monkeypatch.setattr(providers_module(), "build_opener", lambda: second)
    out = providers.UrllibTransport().post_json(
        "https://h.invalid/v1/x",
        headers={"Authorization": "Bearer t"},
        body={"k": [1, 2]},
        timeout=7.5,
    )
    assert out == {"ok": True}
    _assert_request_shape(second)


def providers_module():
    assert providers.ANTHROPIC  # asserts the module exists
    return sys.modules["pixelgen.providers"]


# --- AC5 ------------------------------------------------------------------------------------

OK = b'{"ok": true}'

RETRY_ROWS = {
    "503-503-200": (lambda: [_http(503), _http(503), OK], True, 3, [0.5, 1.0]),
    "urlerror-x3": (lambda: [urllib.error.URLError("boom")], False, 3, [0.5, 1.0]),
    "timeout-x3": (lambda: [TimeoutError()], False, 3, [0.5, 1.0]),
    "429-x3": (lambda: [_http(429)], False, 3, [0.5, 1.0]),
    "400": (lambda: [_http(400)], False, 1, []),
    "non-json-200": (lambda: [b"not json"], False, 1, []),
}


@pytest.mark.parametrize("row", sorted(RETRY_ROWS))
def test_urllib_transport_retry_budget(row):
    script, succeeds, calls, expected_sleeps = RETRY_ROWS[row]
    opener = Opener(*script())
    t, sleeps = _transport(opener)
    if succeeds:
        assert _post(t) == {"ok": True}
    else:
        with pytest.raises(providers.TransportError):
            _post(t)
    assert len(opener.requests) == calls
    assert sleeps == expected_sleeps


def test_urllib_transport_retry_budget_matches_llm():
    assert providers.TRANSPORT_MAX_ATTEMPTS == 1 + llm.TRANSPORT_MAX_RETRIES == 3
    assert providers.INITIAL_BACKOFF_SECONDS == llm.INITIAL_BACKOFF_SECONDS
    assert providers.MAX_BACKOFF_SECONDS == llm.MAX_BACKOFF_SECONDS


# --- AC6 ------------------------------------------------------------------------------------

SENTINEL = "pxg37-sentinel-" + uuid.uuid4().hex
SURL = f"https://u:{SENTINEL}@h.invalid/v1/x?key={SENTINEL}"


def _leaky(code):
    return _http(
        code, reason=SENTINEL, body=SENTINEL.encode(), headers={"X-Leak": SENTINEL}, url=SURL
    )


MAPPING_ROWS = {
    "503": (lambda: _leaky(503), ["HTTP 503", "gave up after 3 attempts"]),
    "401": (lambda: _leaky(401), ["HTTP 401", "check the API key"]),
    "404": (lambda: _leaky(404), ["HTTP 404"]),
    "302": (lambda: _leaky(302), ["HTTP 302"]),
    "timeout": (lambda: TimeoutError(SENTINEL), ["timed out"]),
    "urlerror-timeout": (lambda: urllib.error.URLError(TimeoutError(SENTINEL)), ["timed out"]),
    "urlerror": (lambda: urllib.error.URLError(SENTINEL), ["could not reach h.invalid"]),
    "remote-disconnected": (
        lambda: http.client.RemoteDisconnected(SENTINEL),
        ["could not reach h.invalid"],
    ),
    "invalid-header-control-char": (
        lambda: ValueError(f"Invalid header value b'Bearer {SENTINEL}\\rx'"),
        ["header or the URL is invalid"],
    ),
    "oserror": (lambda: OSError(SENTINEL), ["could not reach h.invalid"]),
    "not-json": (lambda: b"\xff\xfe" + SENTINEL.encode(), ["was not JSON"]),
    "not-object": (lambda: b'["' + SENTINEL.encode() + b'"]', ["not a JSON object"]),
}


@pytest.mark.parametrize("row", sorted(MAPPING_ROWS))
def test_urllib_transport_error_mapping(row):
    build, tokens = MAPPING_ROWS[row]
    t, _ = _transport(Opener(build()))
    with pytest.raises(providers.TransportError) as exc:
        t.post_json(
            SURL, headers={"Authorization": f"Bearer {SENTINEL}"}, body={"k": 1}, timeout=1.0
        )
    err = exc.value
    assert isinstance(err, PixelgenError)
    assert "\n" not in str(err)
    for token in tokens:
        assert token in str(err)
    for text in (str(err), repr(err)):
        assert SENTINEL not in text
        assert "/v1" not in text
        assert "?" not in text
    assert err.__cause__ is None
    assert err.__context__ is None


def test_urllib_transport_error_mapping_real_header_validation():
    # No injected opener: http.client itself rejects the control character, before any socket.
    t = providers.UrllibTransport(sleep=lambda s: None)
    with pytest.raises(providers.TransportError) as exc:
        t.post_json(
            "https://h.invalid/v1/images",
            headers={"Authorization": f"Bearer {SENTINEL}\rx"},
            body={},
            timeout=1.0,
        )
    assert SENTINEL not in str(exc.value) + repr(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


# --- AC7 ------------------------------------------------------------------------------------


def test_urllib_transport_refuses_scheme_and_redirects():
    for url in ("file:///etc/hostname", "ftp://h.invalid/x", "http://h.invalid/x"):
        opener = Opener(OK)
        t, _ = _transport(opener)
        with pytest.raises(providers.TransportError) as exc:
            _post(t, url)
        assert "refusing to send a request" in str(exc.value)
        assert opener.requests == []
    opener = Opener(OK)
    t, _ = _transport(opener)
    assert _post(t, "http://127.0.0.1:7860/x") == {"ok": True}
    assert len(opener.requests) == 1

    req = urllib.request.Request("https://h.invalid/x", data=b"{}", method="POST")
    handler = providers._NoRedirectHandler()
    assert handler.redirect_request(req, None, 302, "Found", {}, "https://other.invalid/") is None
    handlers = providers.build_opener().handlers
    assert any(isinstance(h, providers._NoRedirectHandler) for h in handlers)
    assert not any(type(h) is urllib.request.HTTPRedirectHandler for h in handlers)


# --- AC8 ------------------------------------------------------------------------------------


def test_unpatched_urllib_transport_is_stopped_by_socket_guard():
    sleeps = []
    t = providers.UrllibTransport(sleep=sleeps.append)
    caught = None
    try:
        t.post_json("https://fake-image.invalid/v1/images", headers={}, body={}, timeout=1.0)
    except BaseException as exc:  # noqa: BLE001 - the guard may raise a BaseException
        caught = exc
    assert caught is not None, "the unpatched call returned instead of being stopped"
    seen, stack, found = set(), [caught], False
    while stack:
        exc = stack.pop()
        if exc is None or id(exc) in seen:
            continue
        seen.add(id(exc))
        if "pixelgen test guard: outbound network blocked" in str(exc):
            found = True
        stack += [exc.__context__, exc.__cause__, getattr(exc, "reason", None)]
    assert found, repr(caught)
    assert len(sleeps) <= 2


# --- AC9 ------------------------------------------------------------------------------------


def test_providers_imports_stdlib_only():
    path = ROOT / "src" / "pixelgen" / "providers.py"
    assert path.is_file(), "src/pixelgen/providers.py is missing"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.split(".")[0] in sys.stdlib_module_names:
                continue
            if name in ("pixelgen.errors", "pixelgen.backends.base"):
                continue
            offenders.append(name)
    assert offenders == []
    for node in tree.body:  # no module-level import of base / llm / anthropic
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            assert not mod.startswith(("pixelgen.backends", "pixelgen.llm", "anthropic"))
