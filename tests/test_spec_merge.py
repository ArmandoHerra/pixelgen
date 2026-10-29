"""PXG-05: resolve_spec precedence (defaults < preset < LLM < CLI)."""

import ast
import copy
import inspect

import pytest

import pixelgen.spec
from pixelgen.errors import SpecError
from pixelgen.presets import load_preset
from pixelgen.spec import ArtSpec, resolve_spec


def test_boundary_defaults_lt_preset():
    spec = resolve_spec(
        defaults={"width": 48}, preset={"width": 64}, cli_overrides={"subject": "s"}
    )
    assert spec.width == 64
    assert resolve_spec(preset={"dither": "bayer"}, inferred={"subject": "s"}).dither == "bayer"


def test_boundary_preset_lt_llm():
    spec = resolve_spec(
        preset={"dither": "bayer"},
        inferred={"subject": "slime", "dither": "floyd-steinberg"},
        cli_overrides={"dither": None},
    )
    assert spec.dither == "floyd-steinberg"


def test_boundary_llm_lt_cli():
    spec = resolve_spec(
        inferred={"subject": "s", "palette": "nes", "dither": "floyd-steinberg"},
        cli_overrides={"palette": "gameboy", "dither": "none"},
    )
    assert spec.palette == "gameboy"
    assert spec.dither == "none"


def test_explicit_false_beats_preset_and_none_does_not():
    nes = load_preset("nes")
    inferred = {"subject": "s"}
    assert (
        resolve_spec(preset=nes, inferred=inferred, cli_overrides={"outline": False}).outline
        is False
    )
    assert (
        resolve_spec(preset=nes, inferred=inferred, cli_overrides={"outline": None}).outline is True
    )
    seeded = {"subject": "s", "seed": 7}
    assert resolve_spec(inferred=seeded, cli_overrides={"seed": 0}).seed == 0
    gb = load_preset("gameboy")
    spec = resolve_spec(
        preset=gb, inferred=inferred, cli_overrides={"outline": True, "dither": None}
    )
    assert spec.outline is True
    assert spec.dither == "bayer"


def test_none_is_not_given_at_every_layer():
    nones = {name: None for name in ArtSpec.model_fields}
    base = {"subject": "s"}
    assert resolve_spec(inferred=base, cli_overrides=nones) == resolve_spec(inferred=base)
    spec = resolve_spec(preset={"dither": "bayer"}, inferred={"subject": "s", "dither": None})
    assert spec.dither == "bayer"


def test_artspec_inferred_counts_only_set_fields():
    preset = {"dither": "bayer", "width": 64}
    spec = resolve_spec(preset=preset, inferred=ArtSpec(subject="slime"))
    assert spec.dither == "bayer"
    assert spec.width == 64
    assert (
        resolve_spec(preset=preset, inferred=ArtSpec(subject="slime", dither="none")).dither
        == "none"
    )


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"cli_overrides": {"subject": "s", "width": 200}}, "width=200"),
        ({"inferred": {"subject": "s", "dither": "sierra"}}, "dither='sierra'"),
        ({"cli_overrides": {"subject": "s", "colour": "red"}}, "unknown field(s): colour"),
        ({"preset": {"width": 16}}, "subject is required"),
    ],
)
def test_invalid_merged_spec_raises_spec_error(kwargs, needle):
    with pytest.raises(SpecError) as info:
        resolve_spec(**kwargs)
    message = str(info.value)
    assert message.startswith("invalid spec:")
    assert needle in message
    assert "\n" not in message
    assert "pydantic" not in message.lower()


def test_resolve_spec_does_no_palette_io():
    spec = resolve_spec(cli_overrides={"subject": "s", "palette": "no-such-palette"})
    assert spec.palette == "no-such-palette"
    banned = {"pixelgen.palettes", "pixelgen.presets", "palettes", "presets"}
    for node in ast.walk(ast.parse(inspect.getsource(pixelgen.spec))):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {module} | {f"{module}.{alias.name}".strip(".") for alias in node.names}
            names |= {alias.name for alias in node.names} if node.level else set()
        else:
            continue
        assert not (names & banned), names


def test_resolve_spec_is_pure():
    d = {"width": 48}
    p = {"width": 64, "dither": "bayer"}
    i = {"subject": "s", "palette": "nes"}
    c = {"palette": "gameboy", "outline": None}
    before = copy.deepcopy((d, p, i, c))
    first = resolve_spec(d, p, i, c)
    assert (d, p, i, c) == before
    assert resolve_spec(d, p, i, c) == first
