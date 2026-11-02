"""PXG-32: the harness blocks every outbound socket, in-process and in children."""

import http.client
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parent.parent
_VENDOR_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_BASE_URL",
    "PIXELGEN_MODEL",
)


@pytest.fixture(scope="module", autouse=True)
def _outer_keys_set():
    """Simulate a developer shell with every key exported."""
    mp = pytest.MonkeyPatch()
    for name in _VENDOR_KEYS:
        mp.setenv(name, "dummy-outer")
    yield
    mp.undo()


@pytest.fixture
def netguard():
    module = sys.modules.get("pixelgen_netguard")
    assert module is not None, "tests/conftest.py did not load the netguard module"
    assert getattr(module, "PIXELGEN_NETGUARD", False)
    return module


def _closed_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
@pytest.mark.parametrize("family, host", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")])
def test_inet_connect_is_blocked(netguard, method, family, host):
    port = _closed_port()
    sock = socket.socket(family, socket.SOCK_STREAM)
    address = (host, port) if family == socket.AF_INET else (host, port, 0, 0)

    with pytest.raises(pytest.fail.Exception) as exc:
        getattr(sock, method)(address)

    assert exc.value.msg.startswith(netguard.GUARD_PREFIX)
    assert sock.fileno() == -1


def test_unix_socket_is_allowed(netguard, tmp_path):
    a, b = socket.socketpair(socket.AF_UNIX)
    a.close()
    b.close()
    path = str(tmp_path / "s.sock")
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(path)
        server.listen(1)
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(path)
            conn, _ = server.accept()
            conn.close()


def _http_client(port):
    http.client.HTTPConnection("127.0.0.1", port).request("GET", "/")


def _urlopen(port):
    urllib.request.urlopen(f"http://127.0.0.1:{port}/")


@pytest.mark.parametrize(
    "call",
    [lambda port: socket.create_connection(("127.0.0.1", port)), _http_client, _urlopen],
    ids=["create_connection", "http_client", "urlopen"],
)
def test_stdlib_http_paths_are_blocked(netguard, call):
    port = _closed_port()

    with pytest.raises(pytest.fail.Exception) as exc:
        call(port)

    assert exc.value.msg.startswith(netguard.GUARD_PREFIX)


@pytest.mark.parametrize("name", ["getaddrinfo", "gethostbyname", "gethostbyname_ex"])
def test_name_resolution_is_blocked(netguard, name):
    args = ("localhost", 80) if name == "getaddrinfo" else ("localhost",)

    with pytest.raises(pytest.fail.Exception) as exc:
        getattr(socket, name)(*args)

    assert exc.value.msg.startswith(netguard.GUARD_PREFIX)


def test_numeric_and_none_hosts_resolve(netguard):
    assert socket.getaddrinfo("127.0.0.1", 80)
    assert socket.getaddrinfo(None, 80)


@pytest.mark.parametrize(
    "family, address, expected",
    [
        (socket.AF_INET, ("127.0.0.1", 80), True),
        (socket.AF_INET, ("192.0.2.1", 80), True),
        (socket.AF_INET6, ("::1", 80, 0, 0), True),
        (socket.AF_UNIX, "/tmp/x.sock", False),
    ],
)
def test_policy_table_connect(netguard, family, address, expected):
    assert netguard.is_blocked_connect(family, address) is expected


@pytest.mark.parametrize(
    "host, expected",
    [
        (None, False),
        ("::1", False),
        ("127.0.0.1", False),
        ("localhost", True),
        ("example.com", True),
    ],
)
def test_policy_table_host(netguard, host, expected):
    assert netguard.is_blocked_host(host) is expected


def test_urlopen_example_com_fails_with_guard_message(netguard, monkeypatch):
    # D8 precondition: fail before any lookup if the guard is not in place.
    assert netguard.installed()
    assert socket.getaddrinfo is not netguard._ORIGINALS["getaddrinfo"]
    assert socket.socket.connect is not netguard._ORIGINALS["connect"]
    for name in list(netguard._ORIGINALS):

        def tripwire(*a, _name=name, **k):
            raise AssertionError(f"netguard tripwire: delegated {_name}")

        monkeypatch.setitem(netguard._ORIGINALS, name, tripwire)

    with pytest.raises(pytest.fail.Exception) as exc:
        urllib.request.urlopen("https://example.com", timeout=5)

    assert exc.value.msg.startswith(netguard.GUARD_PREFIX)
    assert "example.com" in exc.value.msg


_PROBE = """
import sys
print(sorted(m for m in sys.modules if m.split(".")[0] in ("anthropic", "pixelgen", "pytest")))
import pixelgen
print(pixelgen.__file__)
sys.stdout.flush()
import urllib.request
urllib.request.urlopen("http://127.0.0.1:%d/")
"""


def test_child_process_is_guarded(netguard):
    port = _closed_port()

    result = subprocess.run(
        [sys.executable, "-c", _PROBE % port],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    lines = result.stdout.splitlines()
    assert result.returncode == netguard.CHILD_EXIT_CODE
    assert netguard.GUARD_PREFIX in result.stderr
    assert "Traceback" not in result.stderr
    assert lines[0] == "[]"
    assert Path(lines[1]).resolve().is_relative_to(TREE / "src" / "pixelgen")


def test_unguarded_child_with_key_is_refused():
    path = os.environ["PATH"]

    with pytest.raises(pytest.fail.Exception) as exc:
        subprocess.Popen(
            [sys.executable, "-c", "pass"], env={"PATH": path, "OPENAI_API_KEY": "dummy"}
        )
    assert "OPENAI_API_KEY" in exc.value.msg

    ok = subprocess.run([sys.executable, "-c", "pass"], env={"PATH": path}, timeout=60)
    assert ok.returncode == 0

    with pytest.raises(pytest.fail.Exception) as exc:
        subprocess.Popen(
            [sys.executable, "-I", "-c", "pass"],
            env=dict(os.environ, ANTHROPIC_API_KEY="dummy"),
        )
    assert "ANTHROPIC_API_KEY" in exc.value.msg


@pytest.mark.parametrize("flag", ["-I", "-E"])
def test_unguarded_child_shell_string_is_refused(flag):
    keyed = dict(os.environ, OPENAI_API_KEY="x")

    with pytest.raises(pytest.fail.Exception) as exc:
        subprocess.Popen(f"{sys.executable} {flag} -c pass", shell=True, env=keyed)

    assert "OPENAI_API_KEY" in exc.value.msg


def test_guarded_shell_string_child_is_allowed():
    keyed = dict(os.environ, OPENAI_API_KEY="x")

    result = subprocess.run(f"{sys.executable} -c pass", shell=True, env=keyed, timeout=60)

    assert result.returncode == 0


def test_vendor_key_variables_are_cleared():
    leaked = [name for name in _VENDOR_KEYS if name in os.environ]
    assert leaked == []

    code = f"import os; print([k for k in {_VENDOR_KEYS!r} if k in os.environ])"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "[]"
