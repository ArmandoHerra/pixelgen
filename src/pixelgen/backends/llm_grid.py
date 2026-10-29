"""The ``llm-grid`` backend: ask the model for a palette-index grid, validate, retry (PXG-08).

Sizes above 32: the model is asked for at most 32 px on the longer side (aspect
preserved, round-half-up on the shorter side); the grid is then upscaled with
nearest-neighbour resampling to exactly width x height, non-integer ratios included.
Reason: one paid call per attempt and a single validation/retry contract, no seams
between chunks, and the deterministic ``pixelate()`` still runs on the final grid.
Cost: above 32 the output carries at most 32-px detail (2x2 blocks at 64; uneven 1/2-px
blocks for non-integer ratios). Generating in row chunks was rejected: N calls per
image, seams between chunks, and an unclear retry budget per chunk.

Retry budgets (kept separate): this module owns the validation retries.
``MAX_ATTEMPTS`` = 3 calls (1 + at most 2 validation retries, each carrying the previous
error); a non-JSON answer (``LLMOutputError``) consumes one of those attempts. The
transport retries belong to ``pixelgen.llm`` alone and are never repeated here.

Outline (X-PXG-27): the prompt asks for a transparent margin, not an outline; the
deterministic pipeline is the single owner of the outline. Dither (X-PXG-28): the prompt
hint is the only dither influence on this path (the caller quantizes with dither none).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from pydantic import ValidationError

from pixelgen.backends.base import (
    GenerationResult,
    GenerationUsage,
    Generator,
    RefineResult,
    generation_size,
    resample_nearest,
)
from pixelgen.errors import GridValidationError, LLMOutputError
from pixelgen.spec import ArtSpec

if TYPE_CHECKING:
    from pixelgen.llm import LLMClient, LLMResult

MAX_ATTEMPTS = 3
GUIDANCE = (
    "Aim for a readable silhouette.",
    "Use 2-3 tones per material.",
    "Light source from the top-left.",
    "Use no anti-aliasing: hard pixel edges only.",
)
OUTLINE_ON = (
    "Leave at least 1px of transparent margin around the subject on every side. "
    "Do not draw an outline; pixelgen adds it."
)
OUTLINE_OFF = "Do not draw an outline."
DITHER_HINTS = {
    "bayer": "Shade with an ordered (Bayer-style) dither pattern.",
    "floyd-steinberg": "Shade with a diffused (Floyd-Steinberg-style) dither pattern.",
}
RETRY_SUFFIX = (
    "\n\nYour previous response was rejected: {error}\n"
    "Return the complete corrected JSON object only."
)

T = TypeVar("T")


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def validate_grid(data: Any, width: int, height: int, n_colors: int) -> list[list[int]]:
    if not isinstance(data, dict) or "grid" not in data:
        raise GridValidationError('response must be a JSON object with a "grid" key')
    grid = data["grid"]
    if not isinstance(grid, list):
        raise GridValidationError('"grid" must be a list of rows')
    if len(grid) != height:
        raise GridValidationError(f"grid has {len(grid)} rows, expected {height}")
    for i, row in enumerate(grid):
        if not isinstance(row, list):
            raise GridValidationError(f"row {i} is not a list")
        if len(row) != width:
            raise GridValidationError(f"row {i} has {len(row)} values, expected {width}")
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, int):
                raise GridValidationError(f"value {v!r} at row {i}, column {j} is not an integer")
            if v < -1 or v > n_colors - 1:
                raise GridValidationError(
                    f"value {v} at row {i}, column {j} is out of range -1..{n_colors - 1}"
                )
    return grid


def _palette_lines(palette: np.ndarray) -> str:
    return "\n".join(
        f"{i}: #{int(r):02X}{int(g):02X}{int(b):02X}" for i, (r, g, b) in enumerate(palette)
    )


def _system_prompt(spec: ArtSpec, shape: str) -> str:
    lines = [
        "You are a pixel artist who draws sprites as grids of palette indices.",
        *GUIDANCE,
        OUTLINE_ON if spec.outline else OUTLINE_OFF,
    ]
    if spec.dither != "none":
        lines.append(DITHER_HINTS[spec.dither])
    lines += [
        "-1 means transparent.",
        "Return JSON only, no prose and no code fences.",
        f"Shape: {shape}",
    ]
    return "\n".join(lines)


def _size_lines(spec: ArtSpec, gw: int, gh: int, n: int) -> list[str]:
    lines = [f"valid indices: -1..{n - 1}"]
    size = f"Return exactly {gh} rows of {gw} integers ({gw}x{gh})."
    if (gw, gh) != (spec.width, spec.height):
        size += f" (final image {spec.width}x{spec.height}, upscaled by the tool)"
    lines.append(size)
    return lines


class LLMGridGenerator(Generator):
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def _call(self, system: str, user: str) -> LLMResult:
        return self._llm.complete_json(system, user)

    def _attempts(
        self, system: str, user: str, check: Callable[[dict[str, Any]], T]
    ) -> tuple[T, GenerationUsage]:
        usage = GenerationUsage()
        error_text = ""
        last: Exception | None = None
        prompt = user
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                prompt = user + RETRY_SUFFIX.format(error=error_text)
            try:
                result = self._call(system, prompt)
                usage += GenerationUsage(1, result.usage.input_tokens, result.usage.output_tokens)
                return check(result.data), usage
            except LLMOutputError as err:
                spent = getattr(err, "usage", None)
                usage += GenerationUsage(
                    1,
                    getattr(spent, "input_tokens", 0),
                    getattr(spent, "output_tokens", 0),
                )
                error_text, last = _one_line(err), err
            except GridValidationError as err:
                error_text, last = _one_line(err), err
        failure = GridValidationError(
            f"the model returned invalid output after {MAX_ATTEMPTS} attempts: {error_text}"
        )
        failure.usage = usage  # type: ignore[attr-defined]
        raise failure from last

    def generate(self, spec: ArtSpec, palette: np.ndarray) -> GenerationResult:
        n = len(palette)
        gw, gh = generation_size(spec.width, spec.height)
        system = _system_prompt(spec, '{"grid": [[int, ...], ...]}')
        user = "\n".join(
            [
                spec.model_dump_json(),
                "Palette:",
                _palette_lines(palette),
                *_size_lines(spec, gw, gh, n),
            ]
        )
        grid, usage = self._attempts(system, user, lambda d: validate_grid(d, gw, gh, n))
        if (gw, gh) != (spec.width, spec.height):
            grid = resample_nearest(grid, spec.width, spec.height)
        return GenerationResult(grid=grid, generated_size=(gw, gh), usage=usage)

    def refine(
        self,
        previous_spec: ArtSpec,
        previous_grid: list[list[int]],
        palette: np.ndarray,
        instruction: str,
    ) -> RefineResult:
        n = len(palette)
        gw, gh = generation_size(previous_spec.width, previous_spec.height)
        src = resample_nearest(previous_grid, gw, gh)
        system = _system_prompt(previous_spec, '{"spec": {...}, "grid": [[int, ...], ...]}')
        user = "\n".join(
            [
                "Previous spec:",
                previous_spec.model_dump_json(),
                "Previous grid:",
                json.dumps(src, separators=(",", ":")),
                "Palette:",
                _palette_lines(palette),
                f"Edit instruction: {instruction}",
                "Return the updated spec and a full new grid.",
                *_size_lines(previous_spec, gw, gh, n),
            ]
        )
        locked = {
            "width": previous_spec.width,
            "height": previous_spec.height,
            "palette": previous_spec.palette,
        }

        def check(data: dict[str, Any]) -> tuple[ArtSpec, list[list[int]], tuple[str, ...]]:
            llm_spec = data.get("spec")
            if not isinstance(llm_spec, dict):
                raise GridValidationError('response must include a "spec" object')
            try:
                new_spec = ArtSpec.model_validate(
                    {**previous_spec.model_dump(), **llm_spec, **locked}
                )
            except ValidationError as exc:
                detail = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                )
                raise GridValidationError("spec is invalid: " + detail) from exc
            grid = validate_grid(data, gw, gh, n)
            overrides = tuple(k for k, v in locked.items() if k in llm_spec and llm_spec[k] != v)
            return new_spec, grid, overrides

        (new_spec, grid, overrides), usage = self._attempts(system, user, check)
        if (gw, gh) != (previous_spec.width, previous_spec.height):
            grid = resample_nearest(grid, previous_spec.width, previous_spec.height)
        return RefineResult(
            spec=new_spec,
            grid=grid,
            generated_size=(gw, gh),
            usage=usage,
            locked_overrides=overrides,
        )
