"""Scriptable LLM fake shared by the PXG-07..PXG-12 tests (owned by PXG-06, X-PXG-18).

Canonical import (pytest ``prepend`` mode, ``tests/`` is not a package)::

    from fakes import FakeLLM, FakeReply

Script items: a ``dict`` is returned as a deep copy; a ``str`` goes through
``pixelgen.llm.extract_json`` (malformed text raises the real ``LLMOutputError`` carrying
``.usage``); a ``BaseException`` instance is raised; a ``FakeReply`` overrides the usage
for that call; a callable receives the ``FakeCall`` and its return value is processed by
the same rules. Consumers import this module and never edit it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pixelgen.errors import LLMOutputError
from pixelgen.llm import DEFAULT_MAX_TOKENS, LLMResult, Usage, extract_json

_DEFAULT_USAGE = Usage(10, 20)


@dataclass(frozen=True)
class FakeCall:
    system: str
    user: str
    max_tokens: int


@dataclass(frozen=True)
class FakeReply:
    payload: dict | str
    usage: Usage


class FakeLLM:
    def __init__(self, *script: Any, model: str = "fake-model", usage: Usage = _DEFAULT_USAGE):
        self.model = model
        self.calls: list[FakeCall] = []
        self._script: list[Any] = list(script)
        self._usage = usage

    def queue(self, *items: Any) -> FakeLLM:
        self._script.extend(items)
        return self

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> FakeCall:
        assert self.calls, "FakeLLM: no calls recorded"
        return self.calls[-1]

    def complete_json(
        self, system: str, user: str, *, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMResult:
        call = FakeCall(system, user, max_tokens)
        self.calls.append(call)
        n = len(self.calls)
        if n > len(self._script):
            raise AssertionError(f"FakeLLM: no scripted response for call #{n}")
        return self._resolve(self._script[n - 1], call, self._usage)

    def _resolve(self, item: Any, call: FakeCall, usage: Usage) -> LLMResult:
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, FakeReply):
            return self._resolve(item.payload, call, item.usage)
        if callable(item) and not isinstance(item, dict):
            fn: Callable[[FakeCall], Any] = item
            return self._resolve(fn(call), call, usage)
        if isinstance(item, str):
            try:
                data = extract_json(item)
            except LLMOutputError as err:
                err.usage = usage  # type: ignore[attr-defined]
                err.raw_text = item  # type: ignore[attr-defined]
                raise
        elif isinstance(item, dict):
            data = copy.deepcopy(item)
        else:
            raise AssertionError(f"FakeLLM: unsupported script item {type(item).__name__}")
        return LLMResult(data=data, usage=usage, model=self.model, attempts=1)
