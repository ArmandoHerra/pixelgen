"""Test harness: no test may spend money or touch the network (X-PXG-18)."""

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

import anthropic
import anthropic._base_client as _base_client
import pytest

NETGUARD_DIR = str(Path(__file__).parent / "netguard")

_CLEARED_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)

_ORIG_SYNC_INIT = anthropic.Anthropic.__init__
_ORIG_ASYNC_INIT = anthropic.AsyncAnthropic.__init__

_CONSTRUCT_MSG = (
    "pixelgen test guard: tests must never construct a real anthropic client; "
    "use tests/fakes.py::FakeLLM or the allow_anthropic_client marker"
)
_REQUEST_MSG = "pixelgen test guard: SDK request blocked; tests must never call the Anthropic API"


def _blocked_init(self, *args, **kwargs):
    pytest.fail(_CONSTRUCT_MSG, pytrace=False)


def _blocked_request(self, *args, **kwargs):
    pytest.fail(_REQUEST_MSG, pytrace=False)


async def _blocked_async_request(self, *args, **kwargs):
    pytest.fail(_REQUEST_MSG, pytrace=False)


# Installed at import time so module-level construction is caught too.
anthropic.Anthropic.__init__ = _blocked_init
anthropic.AsyncAnthropic.__init__ = _blocked_init
_base_client.SyncAPIClient.request = _blocked_request
_base_client.AsyncAPIClient.request = _blocked_async_request


def _load_netguard():
    """Reuse the child-installed guard (child pytest) or load it by path."""
    existing = sys.modules.get("sitecustomize")
    if existing is not None and getattr(existing, "PIXELGEN_NETGUARD", False):
        sys.modules["pixelgen_netguard"] = existing
        return existing
    spec = importlib.util.spec_from_file_location(
        "pixelgen_netguard", Path(NETGUARD_DIR) / "sitecustomize.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pixelgen_netguard"] = module
    spec.loader.exec_module(module)
    return module


_netguard = _load_netguard()
_netguard.install(lambda message: pytest.fail(message, pytrace=False))


def _is_guarded_child(argv, env):
    """A child is guarded when it loads the guard dir and keeps site/env on."""
    entries = (env.get("PYTHONPATH") or "").split(os.pathsep)
    if NETGUARD_DIR not in entries:
        return False
    if isinstance(argv, (str, bytes, os.PathLike)):
        try:
            argv = shlex.split(os.fsdecode(argv))
        except ValueError:
            return False  # unparseable command: fail closed
    skip_next = False
    for arg in list(argv)[1:]:
        arg = os.fsdecode(arg)
        if skip_next:
            skip_next = False
            continue
        if not arg.startswith("-") or arg == "-" or arg.startswith("--"):
            break
        for letter in arg[1:]:
            if letter in "cm":
                return True
            if letter in "IES":
                return False
            if letter in "WX":
                skip_next = arg.endswith(letter)
                break
    return True


_ORIG_POPEN_INIT = subprocess.Popen.__init__


def _audited_popen_init(self, args, *pos, **kwargs):
    env = kwargs.get("env")
    if env is None and len(pos) > 9:
        env = pos[9]
    if env is None:
        env = os.environ
    if not _is_guarded_child(args, env):
        for name in _CLEARED_KEYS:
            if env.get(name):
                pytest.fail(
                    "pixelgen test guard: child process started without the network "
                    f"guard while an API key variable is set ({name})",
                    pytrace=False,
                )
    _ORIG_POPEN_INIT(self, args, *pos, **kwargs)


subprocess.Popen.__init__ = _audited_popen_init


@pytest.fixture(autouse=True)
def _pixelgen_hermetic_env(monkeypatch, tmp_path, request):
    for key in (*_CLEARED_KEYS, "ANTHROPIC_BASE_URL", "PIXELGEN_MODEL"):
        monkeypatch.delenv(key, raising=False)
    existing_path = os.environ.get("PYTHONPATH", "")
    if existing_path.split(os.pathsep)[0] != NETGUARD_DIR:
        monkeypatch.setenv(
            "PYTHONPATH", os.pathsep.join(p for p in (NETGUARD_DIR, existing_path) if p)
        )
    # Plain, wide output for every CliRunner test regardless of CI colour variables:
    # the app honours NO_COLOR per invocation (PXG-22); tests/test_ci_env.py guards it.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("PIXELGEN_HOME", str(tmp_path / "pixelgen-home"))
    if request.node.get_closest_marker("allow_anthropic_client"):
        monkeypatch.setattr(anthropic.Anthropic, "__init__", _ORIG_SYNC_INIT)
        monkeypatch.setattr(anthropic.AsyncAnthropic, "__init__", _ORIG_ASYNC_INIT)
