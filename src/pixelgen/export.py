"""Export one finished run: native PNG, upscaled preview, JSON sidecar, ``last.json``."""

import importlib.metadata
import io
import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from PIL import Image
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from pixelgen.config import pixelgen_home
from pixelgen.errors import NoPreviousResultError, OutputExistsError, SpecError
from pixelgen.spec import ArtSpec

SIDECAR_SCHEMA_VERSION: Final[int] = 1

_HEX6 = re.compile(r"^[0-9a-f]{6}$")


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    calls: NonNegativeInt = 0


class PaletteInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    colors: list[str]

    @field_validator("colors")
    @classmethod
    def _check_colors(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("colors must not be empty")
        for color in value:
            if not _HEX6.match(color):
                raise ValueError(f"color {color!r} is not lowercase rrggbb")
        return value


class OutputNames(BaseModel):
    model_config = ConfigDict(extra="forbid")

    png: str
    preview: str
    scale: PositiveInt

    @field_validator("png", "preview")
    @classmethod
    def _basename_only(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value:
            raise ValueError("must be a bare file name")
        return value


class Sidecar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    pixelgen_version: str
    created_at: AwareDatetime
    spec: ArtSpec
    grid: list[list[StrictInt]]
    palette: PaletteInfo
    model: str | None
    backend: str
    token_usage: TokenUsage
    outputs: OutputNames

    @field_validator("pixelgen_version")
    @classmethod
    def _version_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "Sidecar":
        if not self.grid or not self.grid[0]:
            raise ValueError("grid must be at least 1x1")
        width = len(self.grid[0])
        top = len(self.palette.colors) - 1
        for row in self.grid:
            if len(row) != width:
                raise ValueError("grid must be rectangular")
            for cell in row:
                if cell < -1 or cell > top:
                    raise ValueError(f"grid index {cell} outside -1..{top}")
        if self.palette.name != self.spec.palette:
            raise ValueError("palette.name must equal spec.palette")
        return self


@dataclass(frozen=True)
class ExportResult:
    png_path: Path
    preview_path: Path
    sidecar_path: Path
    last_path: Path


@dataclass(frozen=True)
class LastResult:
    sidecar: Sidecar
    sidecar_path: Path


def output_paths(output: Path, scale: int) -> tuple[Path, Path, Path]:
    """Return absolute ``(png, preview, sidecar)`` paths for ``output``."""
    output = Path(output)
    stem = output.with_suffix("") if output.suffix == ".png" else output
    stem = stem.absolute()
    name = stem.name
    return (
        stem.with_name(f"{name}.png"),
        stem.with_name(f"{name}@{scale}x.png"),
        stem.with_name(f"{name}.json"),
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="PNG")
    return buffer.getvalue()


def _normalise_grid(grid: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [[int(c) if isinstance(c, np.integer) else c for c in row] for row in grid]


def _validate_arrays(rgba: Any, palette_colors: Any, scale: Any, spec: ArtSpec) -> None:
    if type(scale) is not int or scale < 1:
        raise ValueError(f"scale must be an int >= 1, got {scale!r}")
    if (
        not isinstance(rgba, np.ndarray)
        or rgba.dtype != np.uint8
        or rgba.shape != (spec.height, spec.width, 4)
    ):
        raise ValueError(f"rgba must be a uint8 array of shape ({spec.height}, {spec.width}, 4)")
    if (
        not isinstance(palette_colors, np.ndarray)
        or palette_colors.dtype != np.uint8
        or palette_colors.ndim != 2
        or palette_colors.shape[1] != 3
        or palette_colors.shape[0] < 1
    ):
        raise ValueError("palette_colors must be a uint8 array of shape (N, 3) with N >= 1")


def export(
    output: Path,
    rgba: np.ndarray,
    *,
    spec: ArtSpec,
    grid: Sequence[Sequence[int]],
    palette_colors: np.ndarray,
    model: str | None,
    backend: str,
    token_usage: TokenUsage,
    scale: int,
    overwrite: bool = False,
    now: datetime | None = None,
) -> ExportResult:
    """Write the PNG, preview, sidecar and ``last.json`` for one run."""
    _validate_arrays(rgba, palette_colors, scale, spec)
    png_path, preview_path, sidecar_path = output_paths(output, scale)
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is not None:
        now = now.astimezone(UTC)
    sidecar = Sidecar(
        schema_version=1,
        pixelgen_version=importlib.metadata.version("pixelgen"),
        created_at=now,
        spec=spec,
        grid=_normalise_grid(grid),
        palette=PaletteInfo(
            name=spec.palette,
            colors=[f"{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in palette_colors],
        ),
        model=model,
        backend=backend,
        token_usage=token_usage,
        outputs=OutputNames(png=png_path.name, preview=preview_path.name, scale=scale),
    )

    targets = (png_path, preview_path, sidecar_path)
    for path in targets:
        if path.exists() and not overwrite:
            raise OutputExistsError(f"refusing to overwrite existing file: {path}")
    pre_existing = {path for path in targets if path.exists()}

    png_path.parent.mkdir(parents=True, exist_ok=True)
    preview = np.repeat(np.repeat(rgba, scale, axis=0), scale, axis=1)
    payloads = (
        (png_path, _png_bytes(rgba)),
        (preview_path, _png_bytes(preview)),
        (sidecar_path, (sidecar.model_dump_json(indent=2) + "\n").encode("utf-8")),
    )
    try:
        for path, data in payloads:
            _atomic_write_bytes(path, data)
    except BaseException:
        for path in targets:
            if path not in pre_existing:
                path.unlink(missing_ok=True)
        raise

    last_path = _write_last(sidecar, sidecar_path)
    return ExportResult(png_path, preview_path, sidecar_path, last_path)


def _write_last(sidecar: Sidecar, sidecar_path: Path) -> Path:
    home = pixelgen_home()
    home.mkdir(parents=True, exist_ok=True)
    last_path = home / "last.json"
    wrapper = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "sidecar_path": str(sidecar_path),
        "sidecar": json.loads(sidecar.model_dump_json()),
    }
    _atomic_write_bytes(last_path, (json.dumps(wrapper, indent=2) + "\n").encode("utf-8"))
    return last_path


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _check_version(obj: dict[str, Any], label: str, what: str) -> None:
    if "schema_version" not in obj:
        raise SpecError(f"{label}: {what} has no schema_version")
    version = obj["schema_version"]
    if type(version) is not int or version != 1:
        raise SpecError(
            f"{label}: unsupported {what} schema_version {version!r} (this pixelgen reads 1)"
        )


def _validate_sidecar(obj: Any, label: str) -> Sidecar:
    if not isinstance(obj, dict):
        raise SpecError(f"{label}: sidecar must be a JSON object")
    _check_version(obj, label, "sidecar")
    try:
        return Sidecar.model_validate(obj)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<root>"
        raise SpecError(_one_line(f"{label}: invalid sidecar: {loc}: {first['msg']}")) from None


def _read_json(path: Path, what: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        reason = getattr(exc, "strerror", None) or "undecodable"
        raise SpecError(_one_line(f"{path}: cannot read {what} ({reason})")) from None
    try:
        return json.loads(text)
    except ValueError:
        raise SpecError(f"{path}: {what} is not valid JSON") from None


def load_sidecar(path: Path) -> Sidecar:
    """Read and validate a v1 sidecar; every failure is a one-line ``SpecError``."""
    path = Path(path)
    return _validate_sidecar(_read_json(path, "sidecar"), str(path))


def load_last() -> LastResult:
    """Read ``$PIXELGEN_HOME/last.json`` (the self-contained wrapper)."""
    path = pixelgen_home() / "last.json"
    if not path.exists():
        raise NoPreviousResultError(f"no previous result: {path} does not exist")
    data = _read_json(path, "last.json")
    label = str(path)
    if not isinstance(data, dict):
        raise SpecError(f"{label}: last.json must be a JSON object")
    _check_version(data, label, "last.json")
    sidecar_path = data.get("sidecar_path")
    if not isinstance(sidecar_path, str) or not sidecar_path:
        raise SpecError(f"{label}: last.json has no sidecar_path")
    if "sidecar" not in data:
        raise SpecError(f"{label}: last.json has no sidecar")
    return LastResult(_validate_sidecar(data["sidecar"], label), Path(sidecar_path))
