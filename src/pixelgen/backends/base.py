"""Generator interface, result types and the lazy backend registry (PXG-08)."""

from __future__ import annotations

import abc
import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pixelgen import providers
from pixelgen.errors import BackendUnavailableError
from pixelgen.providers import ProviderConfig

if TYPE_CHECKING:
    from pixelgen.llm import LLMClient
    from pixelgen.providers import Transport
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
    """Exactly one payload: an index ``grid`` or an RGB(A) ``image`` raster (PXG-35)."""

    generated_size: tuple[int, int]  # (gw, gh): requested grid size, or the raster's (W, H)
    usage: GenerationUsage
    grid: list[list[int]] | None = field(default=None, kw_only=True)  # height rows of width ints
    image: np.ndarray | None = field(default=None, kw_only=True, compare=False, repr=False)

    def __post_init__(self) -> None:
        if (self.grid is None) == (self.image is None):
            raise ValueError("GenerationResult needs exactly one of grid or image")
        img = self.image
        if img is not None and not (
            isinstance(img, np.ndarray)
            and img.dtype == np.uint8
            and img.ndim == 3
            and img.shape[2] in (3, 4)
            and img.shape[0] >= 1
            and img.shape[1] >= 1
        ):
            raise ValueError(
                "GenerationResult.image must be a uint8 array of shape (H, W, 3) or (H, W, 4)"
            )


@dataclass(frozen=True)
class RefineResult:
    spec: ArtSpec
    grid: list[list[int]]
    generated_size: tuple[int, int]
    usage: GenerationUsage
    locked_overrides: tuple[str, ...]


class Generator(abc.ABC):
    @classmethod
    def check_env(cls, environ: Mapping[str, str] | None = None) -> None:
        """Usage-phase check of backend-specific environment (default: nothing to check)."""
        return None

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


@dataclass(frozen=True)
class BackendEntry:
    """A registered backend: its ``module:Class`` target, its provider, and whether its
    generator takes a transport (PXG-37)."""

    target: str
    provider: ProviderConfig
    transport: bool = False


# A bare ``"module:Class"`` string is still accepted as a registry value (a keyless,
# transport-less entry), so tests that register a fake by string keep working.
_KEYLESS = ProviderConfig("unconfigured", None, None, None, None, "images")

_REGISTRY: dict[str, BackendEntry] = {
    "llm-grid": BackendEntry("pixelgen.backends.llm_grid:LLMGridGenerator", providers.ANTHROPIC),
    "diffusion": BackendEntry(
        "pixelgen.backends.diffusion:DiffusionGenerator", providers.DIFFUSION_STUB
    ),
    "openrouter": BackendEntry(
        "pixelgen.backends.openrouter:OpenRouterGenerator", providers.OPENROUTER, transport=True
    ),
}


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def _unknown(name: str) -> BackendUnavailableError:
    return BackendUnavailableError(
        f"unknown backend '{name}'; available: {', '.join(available_backends())}"
    )


def _entry(name: str) -> BackendEntry:
    if name not in _REGISTRY:
        raise _unknown(name)
    value: BackendEntry | str = _REGISTRY[name]
    return BackendEntry(value, _KEYLESS) if isinstance(value, str) else value


def provider_for(name: str) -> ProviderConfig:
    return _entry(name).provider


def uses_transport(name: str) -> bool:
    return _entry(name).transport


def _resolve(name: str) -> type[Generator]:
    module_name, _, class_name = _entry(name).target.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def create_generator(name: str, *, llm: LLMClient, transport: Transport | None = None) -> Generator:
    entry = _entry(name)
    cls = _resolve(name)
    if entry.transport:
        return cls(llm=llm, transport=transport, provider=entry.provider)  # type: ignore[call-arg]
    return cls(llm=llm)


def check_backend_env(name: str, environ: Mapping[str, str] | None = None) -> None:
    """Run the backend's usage-phase environment check (raises a PixelgenError)."""
    _resolve(name).check_env(environ)
