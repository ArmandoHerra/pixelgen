"""Test harness: no test may spend money or touch the network (X-PXG-18)."""

import anthropic
import anthropic._base_client as _base_client
import pytest

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


@pytest.fixture(autouse=True)
def _pixelgen_hermetic_env(monkeypatch, tmp_path, request):
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "PIXELGEN_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    # Plain, wide output for every CliRunner test regardless of CI colour variables:
    # the app honours NO_COLOR per invocation (PXG-22); tests/test_ci_env.py guards it.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("PIXELGEN_HOME", str(tmp_path / "pixelgen-home"))
    if request.node.get_closest_marker("allow_anthropic_client"):
        monkeypatch.setattr(anthropic.Anthropic, "__init__", _ORIG_SYNC_INIT)
        monkeypatch.setattr(anthropic.AsyncAnthropic, "__init__", _ORIG_ASYNC_INIT)
