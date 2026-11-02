"""Outbound-network guard for the pixelgen test harness (PXG-32).

Stdlib only; never import pytest, anthropic or pixelgen here. Loaded two ways:
by tests/conftest.py by path (in-process, with a pytest.fail handler) and as
``sitecustomize`` in child Pythons whose PYTHONPATH starts with this directory
(they exit with CHILD_EXIT_CODE on any socket use).
"""

import ipaddress
import os
import socket

GUARD_PREFIX = "pixelgen test guard: outbound network blocked"
CHILD_EXIT_CODE = 97
PIXELGEN_NETGUARD = True

_ORIGINALS = {}
_STATE = {"on_block": None}
_INET = (socket.AF_INET, socket.AF_INET6)


def is_blocked_connect(family, address):
    """True for every AF_INET/AF_INET6 connect, loopback included."""
    return family in _INET


def is_blocked_host(host):
    """True for every name that is not None and not a numeric IP literal."""
    if host is None:
        return False
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    try:
        ipaddress.ip_address(host)
    except (ValueError, TypeError):
        return True
    return False


def installed():
    return bool(_ORIGINALS) and _STATE["on_block"] is not None


def _block(api, target):
    msg = f"{GUARD_PREFIX} ({api} {target}); use tests/fakes.py fakes, never a real endpoint"
    _STATE["on_block"](msg)
    # A handler that returns would let the call through; never allow that.
    raise RuntimeError(msg)


def _make_connect(name):
    def wrapper(self, address):
        if is_blocked_connect(self.family, address):
            self.close()
            _block(name, address)
        return _ORIGINALS[name](self, address)

    wrapper.__name__ = name
    return wrapper


def _make_lookup(name):
    def wrapper(host, *args, **kwargs):
        if is_blocked_host(host):
            _block(name, host)
        return _ORIGINALS[name](host, *args, **kwargs)

    wrapper.__name__ = name
    return wrapper


def install(on_block):
    """Install the guard; a second call only swaps the handler."""
    _STATE["on_block"] = on_block
    if _ORIGINALS:
        return
    _ORIGINALS["connect"] = socket.socket.connect
    _ORIGINALS["connect_ex"] = socket.socket.connect_ex
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        _ORIGINALS[name] = getattr(socket, name)
    socket.socket.connect = _make_connect("connect")
    socket.socket.connect_ex = _make_connect("connect_ex")
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        setattr(socket, name, _make_lookup(name))


def _child_exit(message):
    os.write(2, (message + "\n").encode("utf-8", "replace"))
    os._exit(CHILD_EXIT_CODE)


if __name__ == "sitecustomize":
    install(_child_exit)
