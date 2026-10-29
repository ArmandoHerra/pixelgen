"""The diffusion backend stub: refuses at construction (PXG-08)."""

from __future__ import annotations

from typing import Any

from pixelgen.backends.base import Generator
from pixelgen.errors import BackendUnavailableError

DIFFUSION_UNAVAILABLE_MESSAGE = (
    "The diffusion backend is not implemented in this version of pixelgen; "
    "use --backend llm-grid (the default)."
)


class DiffusionUnavailableError(BackendUnavailableError, NotImplementedError):
    pass


class DiffusionGenerator(Generator):
    def __init__(self, llm: Any = None) -> None:
        raise DiffusionUnavailableError(DIFFUSION_UNAVAILABLE_MESSAGE)

    def generate(self, spec, palette):  # pragma: no cover - unreachable
        raise DiffusionUnavailableError(DIFFUSION_UNAVAILABLE_MESSAGE)

    def refine(self, previous_spec, previous_grid, palette, instruction):  # pragma: no cover
        raise DiffusionUnavailableError(DIFFUSION_UNAVAILABLE_MESSAGE)
