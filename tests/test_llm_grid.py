"""The llm-grid backend: validation, retries, sizing, refine (PXG-08). FakeLLM only."""

import json

import numpy as np
import pytest
from fakes import FakeLLM, FakeReply

from pixelgen.backends import llm_grid
from pixelgen.backends.base import (
    GenerationUsage,
    generation_size,
    resample_nearest,
)
from pixelgen.backends.llm_grid import LLMGridGenerator, validate_grid
from pixelgen.errors import GridValidationError, LLMError, LLMOutputError
from pixelgen.llm import Usage
from pixelgen.palettes import load_palette
from pixelgen.spec import ArtSpec

P4 = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [255, 255, 255]], np.uint8)


def ok(w, h):
    return {"grid": [[0] * w for _ in range(h)]}


def spec(w=16, h=16, **kw):
    return ArtSpec(subject=kw.pop("subject", "knight"), width=w, height=h, palette="p4", **kw)


def gen(*script):
    fake = FakeLLM(*script)
    return LLMGridGenerator(fake), fake


def _prompts(fake):
    return [c.system + "\n" + c.user for c in fake.calls]


# AC1
def test_validate_grid_accepts_valid_grid():
    g = ok(16, 16)
    assert validate_grid(g, 16, 16, 4) == g["grid"]
    mixed = {"grid": [[-1, 3], [3, -1]]}
    assert validate_grid(mixed, 2, 2, 4) == mixed["grid"]


REJECT = {
    "missing_key": ({}, 'with a "grid" key'),
    "not_a_list": ({"grid": "x"}, "must be a list of rows"),
    "too_few_rows": (ok(16, 15), "15 rows, expected 16"),
    "row_too_short": ({"grid": [[0] * 16] * 15 + [[0] * 15]}, "row 15 has 15 values, expected 16"),
    "value_equals_n": ({"grid": [[4] * 16] * 16}, "out of range -1..3"),
    "value_below_minus_one": ({"grid": [[-2] * 16] * 16}, "out of range -1..3"),
    "float_value": ({"grid": [[2.0] * 16] * 16}, "is not an integer"),
    "bool_value": ({"grid": [[True] * 16] * 16}, "is not an integer"),
    "string_value": ({"grid": [["1"] * 16] * 16}, "is not an integer"),
}


@pytest.mark.parametrize("case", list(REJECT))
def test_validate_grid_rejects(case):
    data, text = REJECT[case]
    with pytest.raises(GridValidationError) as info:
        validate_grid(data, 16, 16, 4)
    assert text in str(info.value)


BAD3 = [ok(16, 15), ok(16, 15), {"grid": [[4] * 16] * 16}]


# AC2 + AC3
def test_generate_exhausts_three_attempts_then_raises():
    g, fake = gen(*BAD3)
    with pytest.raises(GridValidationError) as info:
        g.generate(spec(), P4)
    assert fake.call_count == 3
    msg = str(info.value)
    assert "after 3 attempts" in msg
    assert "out of range -1..3" in msg
    assert "\n" not in msg
    assert info.value.usage.calls == 3


def test_retry_prompts_carry_previous_error():
    g, fake = gen(*BAD3)
    with pytest.raises(GridValidationError):
        g.generate(spec(), P4)
    assert "previous response was rejected" not in fake.calls[0].user
    for k in (1, 2):
        with pytest.raises(GridValidationError) as exp:
            validate_grid(BAD3[k - 1], 16, 16, 4)
        assert str(exp.value) in fake.calls[k].user


# AC4
@pytest.mark.parametrize("bad_count", [0, 1, 2], ids=["first", "second", "third"])
def test_generate_recovers_within_budget(bad_count):
    good = {"grid": [[1] * 16 for _ in range(16)]}
    g, fake = gen(*([ok(16, 15)] * bad_count), good)
    result = g.generate(spec(), P4)
    assert fake.call_count == bad_count + 1
    assert result.grid == good["grid"]


# AC5
def test_non_json_output_consumes_an_attempt():
    g, fake = gen(LLMOutputError("not json"), ok(16, 16))
    g.generate(spec(), P4)
    assert fake.call_count == 2
    assert "not json" in fake.calls[1].user

    script = [LLMOutputError(f"x{i}") for i in range(3)]
    g, fake = gen(*script)
    with pytest.raises(GridValidationError) as info:
        g.generate(spec(), P4)
    assert fake.call_count == 3
    assert info.value.__cause__ is script[2]
    assert info.value.__cause__ is not script[0]


# AC6
def test_transport_error_is_not_retried():
    scripted = LLMError("network down")
    g, fake = gen(scripted)
    with pytest.raises(LLMError) as info:
        g.generate(spec(), P4)
    assert info.value is scripted
    assert not isinstance(info.value, GridValidationError)
    assert fake.call_count == 1


# AC7
def test_generate_prompt_contents():
    palette = load_palette("pico8")
    g, fake = gen(ok(16, 16))
    g.generate(ArtSpec(subject="tiny knight", width=16, height=16, palette="pico8"), palette)
    text = _prompts(fake)[0]
    for i, (r, gg, b) in enumerate(palette):
        assert f"{i}: #{int(r):02X}{int(gg):02X}{int(b):02X}" in text
    for needle in (
        "16 rows of 16 integers",
        "-1 means transparent",
        "readable silhouette",
        "2-3 tones per material",
        "top-left",
        "no anti-aliasing",
        "Return JSON only",
        "tiny knight",
    ):
        assert needle in text


# AC8
def test_outline_guidance_pair():
    g, fake = gen(ok(16, 16), ok(16, 16))
    g.generate(spec(outline=True), P4)
    g.generate(spec(outline=False), P4)
    on, off = _prompts(fake)
    assert "transparent margin" in on
    assert "1px dark outline" not in on
    assert "Do not draw an outline" in off
    assert "transparent margin" not in off


def test_dither_guidance_pair():
    g, fake = gen(ok(16, 16), ok(16, 16), ok(16, 16))
    for d in ("bayer", "floyd-steinberg", "none"):
        g.generate(spec(dither=d), P4)
    bayer, fs, none = _prompts(fake)
    assert "ordered (Bayer-style) dither" in bayer
    assert "diffused (Floyd-Steinberg-style) dither" in fs
    assert "Bayer-style" not in none
    assert "Floyd-Steinberg-style" not in none
    assert "dither pattern" not in none


# AC9
SIZES = {
    "16x16": ((16, 16), (16, 16)),
    "32x32": ((32, 32), (32, 32)),
    "33x33": ((33, 33), (32, 32)),
    "40x40": ((40, 40), (32, 32)),
    "64x48": ((64, 48), (32, 24)),
    "100x30": ((100, 30), (32, 10)),
    "128x8": ((128, 8), (32, 2)),
    "8x128": ((8, 128), (2, 32)),
}


@pytest.mark.parametrize("case", list(SIZES))
def test_generation_size_rule(case):
    (w, h), expected = SIZES[case]
    assert generation_size(w, h) == expected


# AC10
@pytest.mark.parametrize("case", ["16x16", "33x33", "40x40", "64x48", "128x8", "8x128"])
def test_generate_returns_exact_requested_size(case):
    (w, h), (gw, gh) = SIZES[case]
    g, fake = gen(ok(gw, gh))
    result = g.generate(spec(w, h), P4)
    assert len(result.grid) == h
    assert all(len(row) == w for row in result.grid)
    assert result.generated_size == generation_size(w, h)
    assert f"{gh} rows of {gw} integers" in fake.calls[0].user


# AC11
def test_resample_nearest_mapping():
    src = [[r * 32 + c for c in range(32)] for r in range(32)]
    out = resample_nearest(src, 40, 40)
    for r in range(40):
        for c in range(40):
            assert out[r][c] == src[r * 32 // 40][c * 32 // 40]
    assert {v for row in out for v in row} == {v for row in src for v in row}
    small = [[r * 32 + c for c in range(32)] for r in range(24)]
    big = resample_nearest(small, 64, 48)
    for r in range(48):
        for c in range(64):
            assert big[r][c] == small[r // 2][c // 2]


# AC12
def prev_grid(w=16, h=16):
    return [[(r + c) % 4 for c in range(w)] for r in range(h)]


def test_refine_prompt_and_retry_contract():
    pg = prev_grid()
    bad = [{"spec": {}, **p} for p in BAD3]
    g, fake = gen(*bad)
    with pytest.raises(GridValidationError):
        g.refine(spec(), pg, P4, "make the cape blue")
    assert fake.call_count == 3
    first = _prompts(fake)[0]
    for needle in (
        "make the cape blue",
        "knight",
        json.dumps(pg, separators=(",", ":")),
        *(f"{i}: #" for i in range(4)),
    ):
        assert needle in first
    for k in (1, 2):
        with pytest.raises(GridValidationError) as exp:
            validate_grid(bad[k - 1], 16, 16, 4)
        assert str(exp.value) in fake.calls[k].user


# AC13
def test_refine_locks_geometry_and_palette():
    reply = {"spec": {"subject": "knight, blue cape", "width": 32, "palette": "nes"}, **ok(16, 16)}
    g, fake = gen(reply)
    result = g.refine(spec(), prev_grid(), P4, "blue cape")
    assert fake.call_count == 1
    assert result.spec.width == 16
    assert result.spec.palette == "p4"
    assert result.spec.subject == "knight, blue cape"
    assert set(result.locked_overrides) == {"width", "palette"}

    bad = {"spec": {"dither": "sparkle"}, **ok(16, 16)}
    good = {"spec": {}, **ok(16, 16)}
    g, fake = gen(bad, good)
    g.refine(spec(), prev_grid(), P4, "x")
    assert fake.call_count == 2
    assert "spec is invalid: dither" in fake.calls[1].user


# AC14
def test_refine_above_32_resamples_previous_grid():
    pg = prev_grid(64, 48)
    g, fake = gen({"spec": {}, **ok(32, 24)})
    result = g.refine(spec(64, 48), pg, P4, "x")
    prompt = fake.calls[0].user
    assert json.dumps(resample_nearest(pg, 32, 24), separators=(",", ":")) in prompt
    assert "24 rows of 32 integers" in prompt
    assert len(result.grid) == 48
    assert all(len(r) == 64 for r in result.grid)


# AC15
def test_usage_is_aggregated_across_attempts():
    replies = [
        FakeReply(ok(16, 15), Usage(11, 21)),
        FakeReply(ok(16, 15), Usage(12, 22)),
        FakeReply(ok(16, 16), Usage(13, 23)),
    ]
    g, _ = gen(*replies)
    result = g.generate(spec(), P4)
    assert isinstance(result.usage, GenerationUsage)
    assert result.usage == GenerationUsage(3, 36, 66)

    g, _ = gen(*BAD3)
    with pytest.raises(GridValidationError) as info:
        g.generate(spec(), P4)
    assert info.value.usage.calls == 3


# AC18
def test_module_docstring_documents_size_rule_and_budgets():
    doc = llm_grid.__doc__
    for needle in (
        "Sizes above 32",
        "nearest-neighbour",
        "row chunks",
        "Reason:",
        "validation retries",
        "transport retries",
    ):
        assert needle in doc
