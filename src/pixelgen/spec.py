"""The ``ArtSpec`` data model (PLAN.md section 1)."""

from typing import Literal

from pydantic import BaseModel, Field


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
