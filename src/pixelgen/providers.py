"""Provider seam (PXG-37): per-backend provider config and a stdlib HTTP transport.

Stdlib and ``pixelgen.errors`` only at import time (never ``anthropic`` or
``pixelgen.llm``). No concrete provider ships here (OR-24); a provider is a registry
entry in ``pixelgen.backends.base`` plus one generator module. Key *names* are
vendor-standard (OR-26); key *values* are read from the environment and are never
formatted into a message, a log line or an error.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pixelgen.errors import MissingAPIKeyError, PixelgenError

log = logging.getLogger(__name__)

USAGE_UNITS = ("tokens", "images", "calls")

TRANSPORT_MAX_ATTEMPTS = 3
INITIAL_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0
DEFAULT_TIMEOUT_SECONDS = 120.0
RETRYABLE_STATUS = frozenset({408, 409, 429})
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALIAS_KEY_RE = re.compile(r"^PIXELGEN_.*_API_KEY$")


def _scheme_problem(url: str) -> str | None:
    """None when ``url`` may be requested; else the refusal message (names no path/query)."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname or "the provider"
    if scheme == "https":
        return None
    if scheme == "http" and parts.hostname in LOOPBACK_HOSTS:
        return None
    return f"refusing to send a request to {host} over {scheme}: use https (http only to localhost)"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    key_env: str | None
    default_model: str | None
    model_env: str | None
    base_url: str | None
    usage_units: str

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid provider name {self.name!r}")
        for label, var in (("key_env", self.key_env), ("model_env", self.model_env)):
            if var is not None and not _ENV_RE.match(var):
                raise ValueError(f"invalid {label} {var!r}")
        if self.key_env is not None and _ALIAS_KEY_RE.match(self.key_env):
            raise ValueError("key_env must be the vendor-standard name, not a PIXELGEN_ alias")
        if self.usage_units not in USAGE_UNITS:
            raise ValueError(f"usage_units must be one of {USAGE_UNITS}")
        if self.base_url is not None and _scheme_problem(self.base_url) is not None:
            raise ValueError("base_url must be https (http only to localhost)")


# Duplicates pixelgen.llm on purpose so this module never imports ``anthropic``.
ANTHROPIC = ProviderConfig(
    "anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-5-5", "PIXELGEN_MODEL", None, "tokens"
)
SPEC_LLM_PROVIDER = ANTHROPIC
DIFFUSION_STUB = ProviderConfig("diffusion", None, None, None, None, "images")


def resolve_model(config: ProviderConfig, environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    if config.model_env is not None:
        override = env.get(config.model_env, "").strip()
        if override:
            return override
    return config.default_model


def resolve_key(config: ProviderConfig, environ: Mapping[str, str] | None = None) -> str | None:
    if config.key_env is None:
        return None
    env = os.environ if environ is None else environ
    value = env.get(config.key_env, "").strip()
    if not value:
        raise MissingAPIKeyError(missing_key_message(config.key_env))
    return value


def missing_key_message(var: str, backend: str | None = None) -> str:
    if var == SPEC_LLM_PROVIDER.key_env:
        return MissingAPIKeyError.default_message
    if backend is None:
        return f"{var} is not set. Export it in your shell and try again."
    return (
        f"{var} is not set; the '{backend}' backend needs it. "
        "Export it in your shell and try again."
    )


def is_header_safe_key(value: str) -> bool:
    """True when ``value`` is printable ASCII (0x20-0x7e): safe in an HTTP header.

    Stricter than latin-1 on purpose: a key pasted with a smart quote or a control
    character is refused before any spend. Never echo the value.
    """
    return all(0x20 <= ord(ch) < 0x7F for ch in value)


def required_keys(backend: str) -> tuple[str, ...]:
    from pixelgen.backends.base import provider_for  # base imports this module

    keys = (SPEC_LLM_PROVIDER.key_env, provider_for(backend).key_env)
    return tuple(dict.fromkeys(k for k in keys if k))


class TransportError(PixelgenError):
    default_message = "The provider request failed."


@runtime_checkable
class Transport(Protocol):
    def post_json(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float
    ) -> dict[str, Any]: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: an Authorization header must not reach another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirectHandler())


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)


class UrllibTransport:
    def __init__(
        self,
        *,
        opener: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._opener = opener
        self._sleep = sleep

    def __repr__(self) -> str:
        return "UrllibTransport()"

    def post_json(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float
    ) -> dict[str, Any]:
        problem = _scheme_problem(url)
        if problem is not None:
            raise TransportError(problem)
        host = urlsplit(url).hostname or "the provider"
        req = None
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={**headers, "Content-Type": "application/json"},
                method="POST",
            )
        except (ValueError, TypeError):
            pass  # raised below, outside the handler, so no __context__ carries the values
        if req is None:
            raise TransportError(
                f"the request to {host} could not be built: the URL, headers or body are invalid"
            )
        raw = b""
        for attempt in range(1, TRANSPORT_MAX_ATTEMPTS + 1):
            log.debug(
                "provider request attempt %d/%d host=%s", attempt, TRANSPORT_MAX_ATTEMPTS, host
            )
            failure: str
            fatal: str | None = None
            try:
                with (self._opener or build_opener()).open(req, timeout=timeout) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                log.debug("provider error class=%s status=%s", "HTTPError", code)
                if code in RETRYABLE_STATUS or code >= 500:
                    failure = (
                        f"the request to {host} failed (HTTP {code}); gave up after {{n}} attempts"
                    )
                elif code in (401, 403):
                    failure = ""
                    fatal = f"the request to {host} was rejected (HTTP {code}); check the API key"
                else:
                    failure = ""
                    fatal = f"the request to {host} was rejected (HTTP {code})"
            except ValueError:
                # e.g. http.client rejects a header value with a control character; the
                # exception text can echo the value (an API key), so it is never kept.
                failure = ""
                fatal = f"the request to {host} could not be sent: a header or the URL is invalid"
            except (TimeoutError, urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                log.debug("provider error class=%s status=%s", type(exc).__name__, None)
                if _is_timeout(exc):
                    failure = f"the request to {host} timed out; gave up after {{n}} attempts"
                else:
                    failure = (
                        f"could not reach {host}; gave up after {{n}} attempts. Check your network."
                    )
            if fatal is not None:  # raised outside every handler: __context__ stays None
                raise TransportError(fatal)
            if attempt >= TRANSPORT_MAX_ATTEMPTS:
                raise TransportError(failure.replace("{n}", str(attempt)))
            delay = min(INITIAL_BACKOFF_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
            log.debug("retrying in %.1fs", delay)
            self._sleep(delay)
        data: Any = None
        decoded = False
        try:
            data = json.loads(raw.decode("utf-8"))
            decoded = True
        except (ValueError, RecursionError):
            pass
        if not decoded:
            raise TransportError(f"the response from {host} was not JSON")
        if not isinstance(data, dict):
            raise TransportError(f"the response from {host} was not a JSON object")
        return data
