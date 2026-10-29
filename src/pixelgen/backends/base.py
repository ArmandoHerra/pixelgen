"""Generator interface, result types and the lazy backend registry (PXG-08)."""

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pixelgen.errors import BackendUnavailableError

if TYPE_CHECKING:
    from pixelgen.llm import LLMClient
    from pixelgen.spec import ArtSpec

MAX_GENERATION_SIDE = 32


@dataclass(frozen=True)
class GenerationUsage:
    """Calls and tokens spent by one generate/refine (not ``pixelgen.llm.Usage``)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: GenerationUsage) -> GenerationUsage:
        if not isinstance(other, GenerationUsage):
            return NotImplemented
        return GenerationUsage(
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class GenerationResult:
    grid: list[list[int]]  # exactly height rows of width ints
    generated_size: tuple[int, int]  # (gw, gh) as requested from the model
    usage: GenerationUsage


@dataclass(frozen=True)
class RefineResult:
    spec: ArtSpec
    grid: list[list[int]]
    generated_size: tuple[int, int]
    usage: GenerationUsage
    locked_overrides: tuple[str, ...]


class Generator(abc.ABC):
    @abc.abstractmethod
    def generate(self, spec: ArtSpec, palette: np.ndarray) -> GenerationResult: ...

    @abc.abstractmethod
    def refine(
        self,
        previous_spec: ArtSpec,
        previous_grid: list[list[int]],
        palette: np.ndarray,
        instruction: str,
    ) -> RefineResult: ...


def generation_size(width: int, height: int) -> tuple[int, int]:
    """Size to request from the model: longer side capped at 32, aspect preserved."""
    long_side = max(width, height)
    if long_side <= MAX_GENERATION_SIDE:
        return width, height
    short = min(width, height)
    scaled = max(1, (short * MAX_GENERATION_SIDE + long_side // 2) // long_side)
    if width >= height:
        return MAX_GENERATION_SIDE, scaled
    return scaled, MAX_GENERATION_SIDE


def resample_nearest(grid: list[list[int]], out_w: int, out_h: int) -> list[list[int]]:
    """Nearest-neighbour resample: output (r, c) <- source (r*src_h//out_h, c*src_w//out_w)."""
    src_h = len(grid)
    src_w = len(grid[0])
    return [
        [grid[r * src_h // out_h][c * src_w // out_w] for c in range(out_w)] for r in range(out_h)
    ]


_REGISTRY: dict[str, str] = {
    "llm-grid": "pixelgen.backends.llm_grid:LLMGridGenerator",
    "diffusion": "pixelgen.backends.diffusion:DiffusionGenerator",
}


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def _resolve(name: str) -> type[Generator]:
    module_name, _, class_name = _REGISTRY[name].partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def create_generator(name: str, *, llm: LLMClient) -> Generator:
    if name not in _REGISTRY:
        raise BackendUnavailableError(
            f"unknown backend '{name}'; available: {', '.join(available_backends())}"
        )
    return _resolve(name)(llm=llm)
