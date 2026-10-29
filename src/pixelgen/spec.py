"""The ``ArtSpec`` data model (PLAN.md section 1)."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from pixelgen.errors import SpecError

if TYPE_CHECKING:  # annotation-only: keeps the anthropic SDK out of spec.py's import graph
    from pixelgen.llm import LLMClient, Usage


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


def _raw(layer: Mapping[str, Any] | ArtSpec | None) -> dict[str, Any]:
    """A layer's raw entries, ``None`` values included."""
    if layer is None:
        return {}
    if isinstance(layer, ArtSpec):
        return layer.model_dump(exclude_unset=True)
    return dict(layer)


def resolve_spec(
    defaults: Mapping[str, Any] | None = None,
    preset: Mapping[str, Any] | None = None,
    inferred: Mapping[str, Any] | ArtSpec | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ArtSpec:
    """Merge defaults < preset < LLM-inferred < CLI into one validated spec."""
    layers = [_raw(layer) for layer in (defaults, preset, inferred, cli_overrides)]
    # Unknown keys are rejected on the RAW keys; None only means "not given" for known fields.
    unknown = sorted({key for layer in layers for key in layer} - set(ArtSpec.model_fields))
    if unknown:
        raise SpecError(f"invalid spec: unknown field(s): {', '.join(unknown)}")
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update({key: value for key, value in layer.items() if value is not None})
    if "subject" not in merged:
        raise SpecError("invalid spec: subject is required")
    try:
        return ArtSpec.model_validate(merged)
    except ValidationError as exc:
        parts = [f"{'.'.join(str(p) for p in err['loc'])}={err['input']!r}" for err in exc.errors()]
        raise SpecError(f"invalid spec: {'; '.join(parts)}") from exc


# --- Prompt -> inferred spec layer (PXG-09) -----------------------------------------------------

# Fields the LLM may never set: ``preset`` is a lower layer, ``seed`` is user-owned, and
# ``background_color_index`` needs a palette size this module does not know.
_NON_INFERABLE = frozenset({"preset", "seed", "background_color_index"})
_INFERABLE = tuple(name for name in ArtSpec.model_fields if name not in _NON_INFERABLE)
_VALUE_MAX = 80


def _width_bound(attr: Literal["ge", "le"]) -> int:
    """Read the width bound from the model's own constraint metadata (one source)."""
    for item in ArtSpec.model_fields["width"].metadata:
        if hasattr(item, attr):
            return int(getattr(item, attr))
    raise RuntimeError("ArtSpec.width lost its bound")  # pragma: no cover


@dataclass(frozen=True)
class InferConstraints:
    palette_names: tuple[str, ...]
    size_min: int = _width_bound("ge")
    size_max: int = _width_bound("le")


@dataclass(frozen=True)
class DroppedField:
    field: str
    reason: str
    value: str  # repr(value), truncated


@dataclass(frozen=True)
class InferredSpec:
    fields: dict[str, Any]
    dropped: tuple[DroppedField, ...]
    subject_source: Literal["llm", "prompt"]
    usage: "Usage"

    def metadata(self) -> dict[str, Any]:
        """JSON-serializable report of what was kept and dropped (not persisted in cycle 1)."""
        return {
            "inferred_fields": sorted(self.fields),
            "dropped": [
                {"field": d.field, "reason": d.reason, "value": d.value} for d in self.dropped
            ],
            "subject_source": self.subject_source,
        }


def _drop(field: str, reason: str, value: Any) -> DroppedField:
    return DroppedField(field, reason, repr(value)[:_VALUE_MAX])


def _build_infer_prompt(prompt: str, constraints: InferConstraints) -> tuple[str, str]:
    schema = ArtSpec.model_json_schema()
    schema["properties"] = {k: v for k, v in schema["properties"].items() if k in _INFERABLE}
    schema["required"] = [k for k in schema.get("required", []) if k in _INFERABLE]
    system = (
        "You convert a pixel-art request into a JSON art spec. "
        "Reply with a single JSON object and nothing else: no prose, no markdown fences.\n"
        f"The object must follow this JSON schema: {json.dumps(schema)}\n"
        f"palette must be one of: {', '.join(constraints.palette_names)}.\n"
        f"width and height must each be between {constraints.size_min} and "
        f"{constraints.size_max} pixels.\n"
        "Omit any field the request does not imply."
    )
    return system, prompt


def infer_spec(prompt: str, llm: "LLMClient", constraints: InferConstraints) -> InferredSpec:
    """One LLM call; every returned field is validated alone and bad ones are dropped."""
    if not prompt.strip():
        raise SpecError("prompt is empty")
    system, user = _build_infer_prompt(prompt, constraints)
    result = llm.complete_json(system, user)  # LLM-layer errors propagate unchanged

    fields: dict[str, Any] = {}
    dropped: list[DroppedField] = []
    subject: str | None = None
    for key, value in result.data.items():
        if key not in ArtSpec.model_fields:
            dropped.append(_drop(key, "unknown field", value))
        elif key in _NON_INFERABLE:
            dropped.append(_drop(key, "not inferable", value))
        elif value is None:
            dropped.append(_drop(key, "null", value))
        elif key == "palette" and value not in constraints.palette_names:
            dropped.append(_drop(key, "unknown palette", value))
        else:
            try:
                model = ArtSpec.model_validate({"subject": "x", key: value})
            except ValidationError as exc:
                dropped.append(_drop(key, exc.errors()[0]["msg"], value))
                continue
            kept = getattr(model, key)
            if key == "subject":
                if not kept.strip():
                    dropped.append(_drop(key, "blank subject", value))
                    continue
                subject = kept
            else:
                fields[key] = kept
    if subject is None:
        fields["subject"] = prompt.strip()
        source: Literal["llm", "prompt"] = "prompt"
    else:
        fields["subject"] = subject
        source = "llm"
    return InferredSpec(fields, tuple(dropped), source, result.usage)
