"""The ``ArtSpec`` data model (PLAN.md section 1)."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from pixelgen.errors import SpecError


class ArtSpec(BaseModel):
    subject: str  # concise description of what to draw
    width: int = Field(32, ge=8, le=128)  # 8-128
    height: int = Field(32, ge=8, le=128)  # 8-128
    palette: str = "pico8"  # palette name from palettes/
    preset: str | None = None
    background: Literal["transparent", "solid"] = "transparent"
    background_color_index: int | None = Field(None, ge=0)
    outline: bool = True
    dither: Literal["none", "floyd-steinberg", "bayer"] = "none"
    view: Literal["front", "side", "top-down", "isometric"] = "front"
    seed: int | None = None
    notes: str | None = None  # extra style guidance for the generator


def _given(layer: Mapping[str, Any] | ArtSpec | None) -> dict[str, Any]:
    """Entries a layer actually sets; ``None`` means "not given"."""
    if layer is None:
        return {}
    if isinstance(layer, ArtSpec):
        layer = layer.model_dump(exclude_unset=True)
    return {key: value for key, value in layer.items() if value is not None}


def resolve_spec(
    defaults: Mapping[str, Any] | None = None,
    preset: Mapping[str, Any] | None = None,
    inferred: Mapping[str, Any] | ArtSpec | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ArtSpec:
    """Merge defaults < preset < LLM-inferred < CLI into one validated spec."""
    merged: dict[str, Any] = {}
    for layer in (defaults, preset, inferred, cli_overrides):
        merged.update(_given(layer))
    unknown = sorted(set(merged) - set(ArtSpec.model_fields))
    if unknown:
        raise SpecError(f"invalid spec: unknown field(s): {', '.join(unknown)}")
    if "subject" not in merged:
        raise SpecError("invalid spec: subject is required")
    try:
        return ArtSpec.model_validate(merged)
    except ValidationError as exc:
        parts = [f"{'.'.join(str(p) for p in err['loc'])}={err['input']!r}" for err in exc.errors()]
        raise SpecError(f"invalid spec: {'; '.join(parts)}") from exc
