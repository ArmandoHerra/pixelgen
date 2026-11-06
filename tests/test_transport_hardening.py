"""Provider transport hardening (PXG-45, AC1-AC8).

Offline: every transport call goes through an injected opener or a patched
``http.client.HTTPConnection.connect`` (no socket is opened). New symbols are fetched with
``getattr`` so each test fails by assertion, never by an ImportError, against the base tree.
Assertions use product text and test-owned sentinels only, never ``urlsplit``'s or Typer's.
"""

import ast
import http.client
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NETGUARD = str(Path(__file__).resolve().parent / "netguard")
SENTINEL = "pxg45-dummy-key-7f3a"
MALFORMED = "refusing to send a request: the provider URL is malformed"
BASE_URL_MSG = "base_url must be a well-formed https URL (http only to localhost)"
BAD_URLS = {
    "ipv6-unclosed": "https://[::1/x",
    "bracket-echo": "https://[secretvalue]/v1",
}
PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")


def _providers():
    return importlib.import_module("pixelgen.providers")


def _max():
    value = getattr(_providers(), "MAX_RESPONSE_BYTES", None)
    assert value == 16 * 1024 * 1024, f"MAX_RESPONSE_BYTES is {value!r}"
    return value


def _call(fn):
    """Run ``fn``; return (result, exception) so a wrong exception type is an assertion."""
    try:
        return fn(), None
    except BaseException as exc:  # noqa: BLE001 - the test asserts on the type
        return None, exc


def _body(n: int, amt: int | None = None) -> bytes:
    """A JSON object ``{"p":"aaa..."}`` of ``n`` bytes, cut to ``amt`` bytes (built lazily)."""
    size = n if amt is None else min(n, amt)
    if size == n:
        return b'{"p":"' + b"a" * (n - 8) + b'"}'
    return b'{"p":"' + b"a" * max(size - 6, 0)


class _Resp:
    def __init__(self, total, reads):
        self._total = total
        self._reads = reads

    def read(self, amt=None):
        self._reads.append(amt)
        return _body(self._total, amt)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    def __init__(self, total):
        self.total = total
        self.requests = []
        self.reads = []

    def open(self, req, timeout=None):
        self.requests.append(req)
        return _Resp(self.total, self.reads)


class _EmptyObjResp(_Resp):
    def read(self, amt=None):
        self._reads.append(amt)
        return b"{}"


class _EmptyOpener(_Opener):
    def open(self, req, timeout=None):
        self.requests.append(req)
        return _EmptyObjResp(2, self.reads)


def _post(url, opener, sleeps):
    transport = _providers().UrllibTransport(opener=opener, sleep=sleeps.append)
    return transport.post_json(
        url, headers={"Authorization": f"Bearer {SENTINEL}"}, body={}, timeout=1.0
    )


@pytest.mark.parametrize("name", ["ipv6-unclosed", "bracket-echo"])
def test_pxg45_malformed_url_post_json_is_one_transport_error(name):
    providers = _providers()
    opener, sleeps = _EmptyOpener(2), []
    _, exc = _call(lambda: _post(BAD_URLS[name], opener, sleeps))
    assert type(exc) is providers.TransportError, repr(type(exc))
    assert isinstance(exc, providers.PixelgenError)
    assert str(exc) == MALFORMED
    text = str(exc) + repr(exc)
    assert "secretvalue" not in text and SENTINEL not in text
    assert exc.__cause__ is None and exc.__context__ is None
    assert opener.requests == []
    assert sleeps == []


@pytest.mark.parametrize("name", ["ipv6-unclosed", "bracket-echo"])
def test_pxg45_malformed_base_url_config_is_key_free(name):
    providers = _providers()
    _, exc = _call(
        lambda: providers.ProviderConfig("p", "P_KEY", "m", "P_MODEL", BAD_URLS[name], "images")
    )
    assert type(exc) is ValueError, repr(type(exc))
    assert str(exc) == BASE_URL_MSG
    assert "secretvalue" not in str(exc) + repr(exc)
    assert exc.__context__ is None


def test_pxg45_cli_app_pins_show_locals_false():
    tree = ast.parse((ROOT / "src" / "pixelgen" / "cli.py").read_text(encoding="utf-8"))
    calls = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "app" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_NoColorTyper"
    ]
    assert len(calls) == 1
    kws = {kw.arg: kw.value for kw in calls[0].keywords}
    value = kws.get("pretty_exceptions_show_locals")
    assert isinstance(value, ast.Constant) and value.value is False


_DRIVER = """
import pixelgen, pixelgen.cli as cli
print(pixelgen.__file__)

class Pxg45Boom(Exception):
    pass

def boom():
    key = "pxg45-loc-sentinel"
    headers = {"Authorization": "Bearer " + key}
    raise Pxg45Boom("boom")

cli._console = boom
cli.app(["palettes"], prog_name="pixelgen")
"""


def test_pxg45_escaping_exception_prints_no_frame_locals(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    for name in ("TYPER_STANDARD_TRACEBACK", "_TYPER_STANDARD_TRACEBACK"):
        env.pop(name, None)
    env["PYTHONPATH"] = NETGUARD
    env["PIXELGEN_HOME"] = str(tmp_path / "home")
    env["NO_COLOR"] = "1"
    env["COLUMNS"] = "200"
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0, (proc.stdout, proc.stderr)
    assert "Pxg45Boom" in proc.stderr, proc.stderr
    assert "pxg45-loc-sentinel" not in proc.stdout + proc.stderr
    printed = Path(proc.stdout.splitlines()[0]).resolve()
    assert printed.is_relative_to(ROOT), printed


def test_pxg45_reply_at_read_ceiling_parses():
    mx = _max()
    opener, sleeps = _Opener(mx), []
    data, exc = _call(lambda: _post("https://h.invalid/v1/x", opener, sleeps))
    assert exc is None, repr(exc)
    assert len(data["p"]) == mx - 8
    assert opener.reads == [mx + 1]


@pytest.mark.parametrize("arm", ["plus-one", "double"])
def test_pxg45_reply_over_read_ceiling_refused(arm):
    providers = _providers()
    mx = _max()
    opener, sleeps = _Opener(mx + 1 if arm == "plus-one" else 2 * mx), []
    _, exc = _call(lambda: _post("https://h.invalid/v1/x", opener, sleeps))
    assert type(exc) is providers.TransportError, repr(exc)
    assert str(exc) == "the response from h.invalid was larger than 16 MiB"
    assert SENTINEL not in str(exc) + repr(exc)
    assert exc.__context__ is None
    assert len(opener.requests) == 1
    assert sleeps == []
    assert opener.reads == [mx + 1]


def _proxy_env(monkeypatch, *, proxied):
    for name in ("no_proxy", "NO_PROXY", "REQUEST_METHOD"):
        monkeypatch.delenv(name, raising=False)
    for name in PROXY_VARS:
        if proxied:
            monkeypatch.setenv(name, "http://proxy.invalid:3128")
        else:
            monkeypatch.delenv(name, raising=False)
    targets = []

    def fake_connect(self):
        targets.append((self.host, self.port))
        raise ConnectionRefusedError

    monkeypatch.setattr(http.client.HTTPConnection, "connect", fake_connect)
    return targets


def _drive_real_transport(url):
    providers = _providers()
    sleeps = []
    transport = providers.UrllibTransport(sleep=sleeps.append)
    _, exc = _call(
        lambda: transport.post_json(
            url, headers={"Authorization": f"Bearer {SENTINEL}"}, body={}, timeout=1.0
        )
    )
    assert type(exc) is providers.TransportError, repr(exc)
    assert SENTINEL not in str(exc) + repr(exc)


def test_pxg45_loopback_http_ignores_env_proxy(monkeypatch):
    targets = _proxy_env(monkeypatch, proxied=True)
    _drive_real_transport("http://127.0.0.1:9/x")
    assert targets == [("127.0.0.1", 9)] * 3


def test_pxg45_https_keeps_env_proxy(monkeypatch):
    targets = _proxy_env(monkeypatch, proxied=True)
    _drive_real_transport("https://h.invalid/v1/x")
    assert targets == [("proxy.invalid", 3128)] * 3
    targets = _proxy_env(monkeypatch, proxied=False)
    _drive_real_transport("https://h.invalid/v1/x")
    assert targets == [("h.invalid", 443)] * 3
