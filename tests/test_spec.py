import typing

import pytest
from pydantic import ValidationError

from pixelgen.spec import ArtSpec

FIELDS = {
    "subject",
    "width",
    "height",
    "palette",
    "preset",
    "background",
    "background_color_index",
    "outline",
    "dither",
    "view",
    "seed",
    "notes",
}


def test_artspec_fields_and_defaults_match_plan():
    assert set(ArtSpec.model_fields) == FIELDS
    s = ArtSpec(subject="x")
    assert (
        s.width,
        s.height,
        s.palette,
        s.preset,
        s.background,
        s.background_color_index,
        s.outline,
        s.dither,
        s.view,
        s.seed,
        s.notes,
    ) == (32, 32, "pico8", None, "transparent", None, True, "none", "front", None, None)
    hints = typing.get_type_hints(ArtSpec)
    assert typing.get_args(hints["background"]) == ("transparent", "solid")
    assert typing.get_args(hints["dither"]) == ("none", "floyd-steinberg", "bayer")
    assert typing.get_args(hints["view"]) == ("front", "side", "top-down", "isometric")


def test_artspec_bounds():
    for field in ("width", "height"):
        for ok in (8, 128):
            assert getattr(ArtSpec(subject="x", **{field: ok}), field) == ok
        for bad in (7, 129):
            with pytest.raises(ValidationError):
                ArtSpec(subject="x", **{field: bad})
    assert ArtSpec(subject="x", background_color_index=None).background_color_index is None
    assert ArtSpec(subject="x", background_color_index=0).background_color_index == 0
    with pytest.raises(ValidationError):
        ArtSpec(subject="x", background_color_index=-1)
    with pytest.raises(ValidationError):
        ArtSpec(subject="x", dither="ordered")
    with pytest.raises(ValidationError):
        ArtSpec(subject="x", background="gradient")
    with pytest.raises(ValidationError):
        ArtSpec(subject="x", view="back")
    with pytest.raises(ValidationError):
        ArtSpec()
