"""Packaged YAML style presets (PXG-05)."""

from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

import yaml
from pydantic import ValidationError

from pixelgen.errors import PresetError
from pixelgen.spec import ArtSpec

PRESET_FORBIDDEN_KEYS = frozenset({"subject", "preset", "seed"})


def _preset_dir() -> Traversable:
    """The single place that knows where preset data lives (X-PXG-26 seam)."""
    return files("pixelgen") / "presets"


def list_presets() -> list[str]:
    return sorted(
        entry.name[:-5]
        for entry in _preset_dir().iterdir()
        if entry.is_file() and entry.name.endswith(".yaml")
    )


def parse_preset(text: str, name: str) -> dict[str, Any]:
    prefix = f"preset '{name}':"
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PresetError(f"{prefix} not valid YAML") from exc
    if data is None:
        raise PresetError(f"{prefix} the file is empty")
    if not isinstance(data, dict):
        raise PresetError(f"{prefix} must be a mapping of spec fields")
    keys = [str(key) for key in data]
    unknown = sorted(set(keys) - set(ArtSpec.model_fields))
    if unknown:
        raise PresetError(f"{prefix} unknown key(s): {', '.join(unknown)}")
    forbidden = sorted(set(keys) & PRESET_FORBIDDEN_KEYS)
    if forbidden:
        raise PresetError(f"{prefix} key(s) not allowed in a preset: {', '.join(forbidden)}")
    try:
        model = ArtSpec.model_validate({"subject": "<preset validation>", **data})
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in err["loc"]) for err in exc.errors()})
        raise PresetError(f"{prefix} invalid value for: {', '.join(fields)}") from exc
    return {key: getattr(model, key) for key in data}


def load_preset(name: str) -> dict[str, Any]:
    available = list_presets()
    if name not in available:
        raise PresetError(f"preset '{name}': unknown preset (available: {', '.join(available)})")
    text = (_preset_dir() / f"{name}.yaml").read_text(encoding="utf-8")
    return parse_preset(text, name)
