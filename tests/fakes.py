"""Scriptable LLM fake shared by the PXG-07..PXG-12 tests (owned by PXG-06, X-PXG-18).

Canonical import (pytest ``prepend`` mode, ``tests/`` is not a package)::

    from fakes import FakeLLM, FakeReply

Script items: a ``dict`` is returned as a deep copy; a ``str`` goes through
``pixelgen.llm.extract_json`` (malformed text raises the real ``LLMOutputError`` carrying
``.usage``); a ``BaseException`` instance is raised; a ``FakeReply`` overrides the usage
for that call; a callable receives the ``FakeCall`` and its return value is processed by
the same rules. Consumers import this module and never edit it (PXG-35 adds the raster fake).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pixelgen import providers
from pixelgen.backends import base
from pixelgen.backends.base import GenerationResult, GenerationUsage, Generator
from pixelgen.errors import BackendUnavailableError, LLMOutputError
from pixelgen.llm import DEFAULT_MAX_TOKENS, LLMResult, Usage, extract_json
from pixelgen.providers import ProviderConfig

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


def raster_gradient(width: int = 512, height: int = 512, channels: int = 3) -> np.ndarray:
    """Horizontal grey ramp ``x * 255 // (width - 1)``; alpha 255 when ``channels == 4``."""
    ramp = (np.arange(width, dtype=np.int64) * 255) // max(width - 1, 1)
    img = np.empty((height, width, channels), dtype=np.uint8)
    img[..., :3] = ramp[None, :, None].astype(np.uint8)
    if channels == 4:
        img[..., 3] = 255
    return img


def raster_noise(seed: int, width: int = 512, height: int = 512, channels: int = 3) -> np.ndarray:
    """Seeded uniform noise; the alpha channel is random too when ``channels == 4``."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)


class FakeRasterGenerator(Generator):
    """Image backend fake: returns ``image_factory(spec)`` as a raster result (PXG-35)."""

    image_factory: Callable[[Any], np.ndarray] = staticmethod(lambda spec: raster_gradient())
    log: list[Any] = []

    def __init__(self, llm: Any = None):
        self.llm = llm

    def generate(self, spec: Any, palette: np.ndarray) -> GenerationResult:
        type(self).log.append(spec)
        img = type(self).image_factory(spec)
        return GenerationResult(
            image=img, generated_size=(img.shape[1], img.shape[0]), usage=GenerationUsage(calls=1)
        )

    def refine(self, previous_spec, previous_grid, palette, instruction):
        raise AssertionError("FakeRasterGenerator.refine must never be called (X-PXG-97)")


# --- provider seam fakes (PXG-37) ------------------------------------------------------------


@dataclass(frozen=True)
class TransportCall:
    url: str
    headers: dict
    body: dict
    timeout: float


class FakeTransport:
    """Scriptable ``Transport``: dict -> deep copy returned, exception -> raised,
    callable -> called with the ``TransportCall``. An exhausted script is an AssertionError."""

    def __init__(self, *script: Any):
        self.calls: list[TransportCall] = []
        self._script: list[Any] = list(script)

    def queue(self, *items: Any) -> FakeTransport:
        self._script.extend(items)
        return self

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def post_json(self, url: str, *, headers: Any, body: Any, timeout: float) -> dict:
        call = TransportCall(url, copy.deepcopy(dict(headers)), copy.deepcopy(dict(body)), timeout)
        self.calls.append(call)
        n = len(self.calls)
        if n > len(self._script):
            raise AssertionError(f"FakeTransport: no scripted response for call #{n}")
        return self._resolve(self._script[n - 1], call)

    def _resolve(self, item: Any, call: TransportCall) -> dict:
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, dict):
            return self._resolve(item(call), call)
        if isinstance(item, dict):
            return copy.deepcopy(item)
        raise AssertionError(f"FakeTransport: unsupported script item {type(item).__name__}")


FAKE_PROVIDER = ProviderConfig(
    "fake-image",
    "FAKEIMG_TEST_KEY",
    "fake-image-1",
    "FAKEIMG_TEST_MODEL",
    "https://fake-image.invalid/v1",
    "images",
)


def fake_image_reply(w: int, h: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return {"width": w, "height": h, "rgb": rgb.tolist()}


class FakeImageGenerator(Generator):
    """Test-only image provider backend: one ``transport.post_json`` per generate."""

    def __init__(self, llm: Any = None, transport: Any = None, provider: Any = None):
        self.llm = llm
        self.transport = transport
        self.provider = provider

    def generate(self, spec: Any, palette: np.ndarray) -> GenerationResult:
        key = providers.resolve_key(self.provider)
        model = providers.resolve_model(self.provider)
        reply = self.transport.post_json(
            f"{self.provider.base_url}/images",
            headers={"Authorization": f"Bearer {key}"},
            body={
                "model": model,
                "prompt": spec.subject,
                "width": spec.width,
                "height": spec.height,
            },
            timeout=providers.DEFAULT_TIMEOUT_SECONDS,
        )
        w, h = int(reply["width"]), int(reply["height"])
        image = np.asarray(reply["rgb"], dtype=np.uint8).reshape(h, w, 3)
        return GenerationResult(image=image, generated_size=(w, h), usage=GenerationUsage(calls=1))

    def refine(self, previous_spec, previous_grid, palette, instruction):
        raise AssertionError("FakeImageGenerator.refine must never be called (X-PXG-97)")


class FakeUnavailableGenerator(Generator):
    def __init__(self, llm: Any = None, **_: Any):
        raise BackendUnavailableError("the 'fake-down' backend is not available in this test")

    def generate(self, spec, palette):  # pragma: no cover - unreachable
        raise AssertionError

    def refine(self, previous_spec, previous_grid, palette, instruction):  # pragma: no cover
        raise AssertionError


def register_fake_provider(
    monkeypatch: Any,
    name: str = "fake-image",
    target: str = "fakes:FakeImageGenerator",
    provider: ProviderConfig = FAKE_PROVIDER,
) -> None:
    monkeypatch.setitem(base._REGISTRY, name, base.BackendEntry(target, provider, transport=True))
