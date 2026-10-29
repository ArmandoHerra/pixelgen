"""Thin, JSON-only wrapper around the Anthropic SDK (PXG-06).

Every LLM consumer talks to the ``LLMClient`` protocol: ``complete_json(system, user)``
returns an ``LLMResult`` (a JSON object plus token usage). ``AnthropicLLM`` is the one
real adapter; ``tests/fakes.py::FakeLLM`` is the scriptable fake used by tests.

Retry budgets (kept separate on purpose):

* The transport budget, ``TRANSPORT_MAX_RETRIES`` (2, this module), covers network and
  API failures inside one ``complete_json`` call: at most 3 attempts, with backoff of
  0.5 s then 1.0 s. The SDK's own retries are switched off (``max_retries=0``) so the
  count is product-owned and cannot multiply with ours.
* The validation budget (at most 2, PXG-08 ``backends/llm_grid.py``) re-asks the model
  when the JSON is well-formed but the grid is invalid; each re-ask is a fresh
  ``complete_json`` call.

Worst case per backend call: (1+2) x (1+2) = 9 requests.

SDK exceptions never escape: each is mapped to a ``PixelgenError`` subclass with a
one-line message, raised ``from None``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import anthropic

from pixelgen.errors import LLMError, LLMOutputError, MissingAPIKeyError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 8192
LLM_TIMEOUT_SECONDS = 120.0
TRANSPORT_MAX_RETRIES = 2
INITIAL_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0
JSON_ONLY_INSTRUCTION = "Respond with exactly one JSON object and nothing else."

_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)
_RETRYABLE_STATUS = frozenset({408, 409})


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResult:
    data: dict[str, Any]
    usage: Usage
    model: str
    attempts: int


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete_json(
        self, system: str, user: str, *, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMResult: ...


def resolve_model(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    override = env.get("PIXELGEN_MODEL", "").strip()
    return override or DEFAULT_MODEL


def _not_json(reason: str) -> LLMOutputError:
    return LLMOutputError(f"The model's response was not a single JSON object ({reason}).")


def extract_json(text: str) -> dict[str, Any]:
    """Parse one JSON object from model text; raises ``LLMOutputError`` otherwise."""
    stripped = text.strip()
    if not stripped:
        raise _not_json("empty response")
    try:
        whole = json.loads(stripped)
    except ValueError:
        whole = None
        whole_ok = False
    else:
        whole_ok = True
    if whole_ok:
        if isinstance(whole, dict):
            return whole
        raise _not_json(f"top-level value is {type(whole).__name__}, not an object")
    fences = _FENCE_RE.findall(stripped)
    if not fences:
        raise _not_json("no JSON found")
    if len(fences) > 1:
        raise _not_json("more than one fenced block")
    try:
        body = json.loads(fences[0].strip())
    except ValueError:
        raise _not_json("fenced block is not valid JSON") from None
    if not isinstance(body, dict):
        raise _not_json(f"fenced value is {type(body).__name__}, not an object")
    return body


def _with_usage(err: LLMOutputError, usage: Usage, raw_text: str) -> LLMOutputError:
    err.usage = usage  # type: ignore[attr-defined]
    err.raw_text = raw_text  # type: ignore[attr-defined]
    return err


def _map_sdk_error(exc: BaseException, model: str, attempts: int) -> tuple[LLMError, bool]:
    """Return (pixelgen error, retryable) for an ``anthropic.AnthropicError``."""
    code = getattr(exc, "status_code", None)
    if isinstance(exc, anthropic.AuthenticationError):
        msg = (
            "Anthropic rejected the key in ANTHROPIC_API_KEY (authentication failed); "
            "check the key."
        )
        return LLMError(msg), False
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LLMError(
            f"Your Anthropic key may not use model '{model}' (permission denied)."
        ), False
    if isinstance(exc, anthropic.NotFoundError):
        return LLMError(f"Model '{model}' was not found; check PIXELGEN_MODEL."), False
    if isinstance(exc, anthropic.RateLimitError):
        msg = f"Anthropic rate limit hit; gave up after {attempts} attempts. Wait and try again."
        return LLMError(msg), True
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMError(f"The Anthropic API timed out; gave up after {attempts} attempts."), True
    if isinstance(exc, anthropic.APIConnectionError):
        msg = (
            f"Could not reach the Anthropic API; gave up after {attempts} attempts. "
            "Check your network."
        )
        return LLMError(msg), True
    if isinstance(exc, anthropic.APIStatusError):
        server_side = isinstance(exc, anthropic.InternalServerError) or (
            isinstance(code, int) and (code >= 500 or code in _RETRYABLE_STATUS)
        )
        if server_side:
            msg = (
                f"The Anthropic API is unavailable (HTTP {code}); "
                f"gave up after {attempts} attempts."
            )
            return LLMError(msg), True
        return LLMError(f"Anthropic rejected the request (HTTP {code})."), False
    return LLMError(f"Unexpected Anthropic SDK error ({type(exc).__name__})."), False


class AnthropicLLM:
    """``LLMClient`` backed by an injected Anthropic SDK client."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self.model = model
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"AnthropicLLM(model={self.model!r})"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> AnthropicLLM:
        env = os.environ if environ is None else environ
        key = env.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not set. Set it with "
                "`export ANTHROPIC_API_KEY=<your key>` (get one at https://console.anthropic.com/)."
            )
        model = resolve_model(env)
        client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=LLM_TIMEOUT_SECONDS)
        return cls(client, model, sleep=sleep)

    def complete_json(
        self, system: str, user: str, *, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMResult:
        max_attempts = 1 + TRANSPORT_MAX_RETRIES
        attempt = 0
        while True:
            attempt += 1
            log.debug("anthropic request attempt %d/%d model=%s", attempt, max_attempts, self.model)
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=f"{system}\n\n{JSON_ONLY_INSTRUCTION}",
                    messages=[{"role": "user", "content": user}],
                )
            except anthropic.AnthropicError as exc:
                log.debug(
                    "anthropic error class=%s status=%s",
                    type(exc).__name__,
                    getattr(exc, "status_code", None),
                )
                err, retryable = _map_sdk_error(exc, self.model, attempt)
                if retryable and attempt < max_attempts:
                    delay = min(INITIAL_BACKOFF_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
                    log.debug("retrying in %.1fs", delay)
                    self._sleep(delay)
                    continue
                raise err from None
            return self._parse(response, attempt)

    def _parse(self, response: Any, attempts: int) -> LLMResult:
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        log.debug("anthropic usage in=%d out=%d", usage.input_tokens, usage.output_tokens)
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        stop = getattr(response, "stop_reason", None)
        if stop == "max_tokens":
            raise _with_usage(
                LLMOutputError("The model's response was truncated (max_tokens); try again."),
                usage,
                text,
            )
        if stop == "refusal":
            raise _with_usage(LLMOutputError("The model refused the request."), usage, text)
        try:
            data = extract_json(text)
        except LLMOutputError as err:
            raise _with_usage(err, usage, text) from None
        return LLMResult(data=data, usage=usage, model=self.model, attempts=attempts)
