"""Registry, result types and the diffusion stub (PXG-08)."""

import pytest
from fakes import FakeLLM

from pixelgen.backends import base
from pixelgen.backends.base import (
    GenerationUsage,
    Generator,
    available_backends,
    create_generator,
)
from pixelgen.backends.diffusion import (
    DIFFUSION_UNAVAILABLE_MESSAGE,
    DiffusionUnavailableError,
)
from pixelgen.backends.llm_grid import LLMGridGenerator
from pixelgen.errors import BackendUnavailableError, PixelgenError


def test_diffusion_backend_is_unavailable():
    fake = FakeLLM()
    with pytest.raises(NotImplementedError) as info:
        create_generator("diffusion", llm=fake)
    e = info.value
    assert isinstance(e, PixelgenError)
    assert isinstance(e, BackendUnavailableError)
    assert str(e) == DIFFUSION_UNAVAILABLE_MESSAGE
    assert "llm-grid" in str(e)
    assert "\n" not in str(e)
    assert fake.call_count == 0


def test_registry_lists_and_creates_backends():
    fake = FakeLLM()
    assert available_backends() == ["diffusion", "llm-grid"]
    assert isinstance(create_generator("llm-grid", llm=fake), LLMGridGenerator)
    assert fake.call_count == 0


def test_unknown_backend_lists_available():
    with pytest.raises(BackendUnavailableError) as info:
        create_generator("dall-e", llm=FakeLLM())
    assert not isinstance(info.value, DiffusionUnavailableError)
    assert "dall-e" in str(info.value)
    assert "available: diffusion, llm-grid" in str(info.value)


def test_registry_entries_resolve():
    for name in available_backends():
        assert issubclass(base._resolve(name), Generator)
    with pytest.raises(TypeError):
        Generator()


def test_generation_usage_adds():
    total = GenerationUsage(1, 2, 3) + GenerationUsage(1, 10, 20)
    assert total == GenerationUsage(2, 12, 23)
