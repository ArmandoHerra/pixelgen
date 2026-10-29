"""The UI-free generate pipeline: spec -> backend -> pixelate -> export (PXG-11).

No Typer or Rich in here: PXG-12 (refine) reuses ``render_grid`` and the helpers unchanged.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pixelgen.backends.base import Generator, create_generator
from pixelgen.config import pixelgen_home
from pixelgen.errors import OutputExistsError, PaletteError, PixelgenError
from pixelgen.export import ExportResult, TokenUsage, export, output_paths
from pixelgen.palettes import list_palettes, load_palette
from pixelgen.pixelate import grid_to_rgba, pixelate
from pixelgen.presets import load_preset
from pixelgen.spec import ArtSpec, InferConstraints, InferredSpec, infer_spec, resolve_spec

# Backends whose grid is already palette-exact: dithering it would only perturb the model's
# colours (X-PXG-28). The sidecar still records the user's dither.
INDEX_GRID_BACKENDS = frozenset({"llm-grid"})
SLUG_MAX = 48
FALLBACK_SLUG = "pixelgen"
MAX_COLLISION_INDEX = 9999
DEFAULT_OUT_DIR = Path("out")


@dataclass(frozen=True)
class PipelineResult:
    spec: ArtSpec
    grid: Any
    image: np.ndarray
    export: ExportResult


def slugify_prompt(prompt: str) -> str:
    text = unicodedata.normalize("NFKD", prompt).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = slug[:SLUG_MAX].rstrip("-")
    slug = re.sub(r"-v(\d+)$", r"-ver\1", slug)  # never parse as a refine version
    return slug or FALLBACK_SLUG


def next_free_output(out_dir: Path, slug: str, scale: int) -> Path:
    """First of ``slug``, ``slug_2`` ... whose three output files all do not exist."""
    for index in range(1, MAX_COLLISION_INDEX + 1):
        candidate = out_dir / (slug if index == 1 else f"{slug}_{index}")
        if not any(path.exists() for path in output_paths(candidate, scale)):
            return candidate
    raise OutputExistsError(
        f"no free output name for '{slug}' in {out_dir} (tried up to _{MAX_COLLISION_INDEX})"
    )


def ensure_backend_available(name: str) -> None:
    """Refuse an unavailable backend before any key check or client exists.

    Backends are constructed with ``llm=None``: an unavailable one refuses at construction
    (``BackendUnavailableError``); an available one just holds the ``None`` and is discarded.
    """
    get_generator(name, None)


def _nearest_existing(path: Path) -> Path:
    current = path
    while not os.path.lexists(current):
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current


def _check_writable_dir(path: Path, label: str) -> None:
    """Refuse when the nearest existing ancestor of ``path`` is no writable directory."""
    target = path.absolute()
    anchor = _nearest_existing(target)
    where = "" if anchor == target else f" ('{anchor}')"
    if not anchor.is_dir():
        raise PixelgenError(f"cannot write {label} '{path}': not a directory{where}")
    if not os.access(anchor, os.W_OK | os.X_OK):
        raise PixelgenError(f"cannot write {label} '{path}': not writable{where}")


def preflight_generate(
    prompt: str,
    output: Path | None,
    scale: int,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Path:
    """Everything knowable about the filesystem before the first paid call; writes nothing.

    Returns the final output path (the default name is resolved here, not after spend) and
    raises ``PixelgenError`` when the output or ``PIXELGEN_HOME`` cannot be written.
    """
    if output is None:
        _check_writable_dir(out_dir, "output directory")
        resolved = next_free_output(out_dir, slugify_prompt(prompt), scale)
    else:
        resolved = output
        _check_writable_dir(output.absolute().parent, "output")
    _check_writable_dir(pixelgen_home(), "pixelgen home")
    return resolved


def ensure_palette_known(name: str) -> None:
    if name not in list_palettes():
        raise PaletteError(f"unknown palette '{name}'; available: {', '.join(list_palettes())}")


def build_spec(
    prompt: str,
    preset_name: str | None,
    cli_overrides: Mapping[str, Any],
    llm: Any = None,
) -> tuple[ArtSpec, InferredSpec | None]:
    """defaults < preset < LLM-inferred < CLI flags; the LLM is called only when given."""
    preset = load_preset(preset_name) if preset_name else None
    inferred: InferredSpec | None = None
    if llm is not None:
        constraints = InferConstraints(palette_names=tuple(list_palettes()))
        inferred = infer_spec(prompt, llm, constraints)
    spec = resolve_spec(
        {"subject": prompt},
        preset,
        inferred.fields if inferred is not None else None,
        cli_overrides,
    )
    ensure_palette_known(spec.palette)
    return spec, inferred


def get_generator(name: str, llm: Any) -> Generator:
    return create_generator(name, llm=llm)


def to_token_usage(*usages: Any) -> TokenUsage:
    """Sum LLM ``Usage`` (one call each) and generation usage (``calls``), by attribute."""
    calls = inputs = outputs = 0
    for usage in usages:
        calls += int(getattr(usage, "calls", 1))
        inputs += int(usage.input_tokens)
        outputs += int(usage.output_tokens)
    return TokenUsage(input_tokens=inputs, output_tokens=outputs, calls=calls)


def render_grid(
    spec: ArtSpec,
    grid: Any,
    palette_colors: np.ndarray,
    *,
    output: Path,
    scale: int,
    overwrite: bool,
    backend: str,
    model: str | None,
    token_usage: TokenUsage,
) -> PipelineResult:
    """grid -> RGBA -> ``pixelate()`` -> export. PXG-12 calls this unchanged."""
    rgba = grid_to_rgba(np.asarray(grid), palette_colors)
    pixel_spec = spec
    if backend in INDEX_GRID_BACKENDS:
        pixel_spec = spec.model_copy(update={"dither": "none"})
    image = pixelate(rgba, pixel_spec, palette_colors)
    result = export(
        output,
        image,
        spec=spec,
        grid=grid,
        palette_colors=palette_colors,
        model=model,
        backend=backend,
        token_usage=token_usage,
        scale=scale,
        overwrite=overwrite,
    )
    return PipelineResult(spec=spec, grid=grid, image=image, export=result)


def run_generate(
    prompt: str,
    *,
    llm: Any,
    generator: Generator,
    backend_name: str,
    preset_name: str | None,
    cli_overrides: Mapping[str, Any],
    output: Path | None,
    overwrite: bool,
    scale: int,
    out_dir: Path = DEFAULT_OUT_DIR,
    on_spec: Callable[[ArtSpec], None] | None = None,
) -> PipelineResult:
    spec, inferred = build_spec(prompt, preset_name, cli_overrides, llm)
    palette = load_palette(spec.palette)
    if on_spec is not None:
        on_spec(spec)
    generated = generator.generate(spec, palette)
    usages = [generated.usage] if inferred is None else [inferred.usage, generated.usage]
    if output is None:  # callers that did not preflight resolve the default name here
        output = next_free_output(out_dir, slugify_prompt(prompt), scale)
        overwrite = False
    return render_grid(
        spec,
        generated.grid,
        palette,
        output=output,
        scale=scale,
        overwrite=overwrite,
        backend=backend_name,
        model=getattr(llm, "model", None),
        token_usage=to_token_usage(*usages),
    )
