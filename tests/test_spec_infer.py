"""PXG-09: infer an ArtSpec layer from a prompt via a fake LLM (no network)."""

import json
import re
import typing

import pytest
from fakes import FakeLLM

from pixelgen.errors import LLMOutputError, SpecError
from pixelgen.presets import load_preset
from pixelgen.spec import (
    ArtSpec,
    DroppedField,
    InferConstraints,
    infer_spec,
    resolve_spec,
)

CONSTRAINTS = InferConstraints(palette_names=("pico8", "gameboy"))
SLIME = {"subject": "slime", "width": 999, "dither": "bayer", "view": "diagonal"}


def test_infer_all_valid_fields_kept():
    payload = {
        "subject": "tiny knight",
        "width": "16",
        "height": 24,
        "palette": "gameboy",
        "dither": "bayer",
        "view": "side",
        "outline": False,
        "background": "solid",
        "notes": "red cape",
    }
    result = infer_spec("a knight", FakeLLM(payload), CONSTRAINTS)
    assert result.fields == {**payload, "width": 16}
    assert type(result.fields["width"]) is int
    assert result.dropped == ()
    assert result.subject_source == "llm"


def test_infer_drops_invalid_field_keeps_rest():
    result = infer_spec("slime", FakeLLM(SLIME), CONSTRAINTS)
    assert result.fields == {"subject": "slime", "dither": "bayer"}
    assert {d.field for d in result.dropped} == {"width", "view"}


def test_infer_propagates_non_object_rejection():
    llm = FakeLLM("[1, 2]")
    with pytest.raises(LLMOutputError, match="was not a single JSON object"):
        infer_spec("x", llm, CONSTRAINTS)
    assert llm.call_count == 1


def test_infer_palette_checked_against_constraints():
    ok = infer_spec("x", FakeLLM({"subject": "s", "palette": "gameboy"}), CONSTRAINTS)
    assert ok.fields["palette"] == "gameboy"
    bad = infer_spec("x", FakeLLM({"subject": "s", "palette": "vaporwave"}), CONSTRAINTS)
    assert "palette" not in bad.fields
    assert bad.dropped == (DroppedField("palette", "unknown palette", "'vaporwave'"),)
    empty = infer_spec(
        "x", FakeLLM({"subject": "s", "palette": "pico8"}), InferConstraints(palette_names=())
    )
    assert "palette" not in empty.fields
    assert [d.reason for d in empty.dropped] == ["unknown palette"]


def test_infer_drops_non_inferable_unknown_and_null():
    payload = {
        "subject": "s",
        "seed": 7,
        "preset": "nes",
        "background_color_index": 2,
        "foo": 1,
        "notes": None,
    }
    result = infer_spec("x", FakeLLM(payload), CONSTRAINTS)
    assert result.fields == {"subject": "s"}
    reasons = {d.field: d.reason for d in result.dropped}
    assert reasons == {
        "seed": "not inferable",
        "preset": "not inferable",
        "background_color_index": "not inferable",
        "foo": "unknown field",
        "notes": "null",
    }
    assert "notes" not in result.fields


def test_infer_subject_fallback_to_prompt():
    prompt = "  a red apple  "
    llm_subject = infer_spec(prompt, FakeLLM({"subject": "apple, glossy"}), CONSTRAINTS)
    assert llm_subject.fields["subject"] == "apple, glossy"
    assert llm_subject.subject_source == "llm"

    omitted = infer_spec(prompt, FakeLLM({"width": 16}), CONSTRAINTS)
    assert omitted.fields["subject"] == "a red apple"
    assert omitted.subject_source == "prompt"
    assert not [d for d in omitted.dropped if d.field == "subject"]

    for bad in (123, "   "):
        result = infer_spec(prompt, FakeLLM({"subject": bad}), CONSTRAINTS)
        assert result.fields["subject"] == "a red apple"
        assert result.subject_source == "prompt"
        assert len([d for d in result.dropped if d.field == "subject"]) == 1


def test_infer_blank_prompt_raises_without_llm_call():
    llm = FakeLLM({"subject": "x"})
    with pytest.raises(SpecError, match="prompt is empty"):
        infer_spec("   ", llm, CONSTRAINTS)
    assert llm.call_count == 0


def test_infer_single_call_no_retry():
    scripted = LLMOutputError("boom")
    llm = FakeLLM(scripted)
    with pytest.raises(LLMOutputError) as info:
        infer_spec("x", llm, CONSTRAINTS)
    assert info.value is scripted
    assert llm.call_count == 1

    drops = FakeLLM(SLIME)
    infer_spec("slime", drops, CONSTRAINTS)
    assert drops.call_count == 1


def test_infer_prompt_asks_json_with_allowed_values():
    llm = FakeLLM({"subject": "s"})
    constraints = InferConstraints(palette_names=("gameboy", "nes"), size_min=12, size_max=96)
    infer_spec("a tiny dragon", llm, constraints)
    system = llm.calls[0].system
    assert "JSON" in system and "nothing else" in system
    for name in ("background", "dither", "view"):
        for value in typing.get_args(ArtSpec.model_fields[name].annotation):
            assert value in system
    assert "floyd-steinberg" in system
    assert "nes" in system and "gameboy" in system
    assert re.search(r"(?<!\d)12(?!\d)", system)
    assert re.search(r"(?<!\d)96(?!\d)", system)
    assert '"seed":' not in system and '"preset":' not in system
    assert "a tiny dragon" in llm.calls[0].user


def test_infer_metadata_is_json_and_truncated():
    result = infer_spec("slime", FakeLLM(SLIME), CONSTRAINTS)
    meta = json.loads(json.dumps(result.metadata()))
    assert meta["subject_source"] == "llm"
    assert meta["inferred_fields"] == ["dither", "subject"]
    assert {d["field"] for d in meta["dropped"]} == {"width", "view"}
    assert all(set(d) == {"field", "reason", "value"} for d in meta["dropped"])

    long = infer_spec("x", FakeLLM({"subject": "s", "view": "x" * 500}), CONSTRAINTS)
    assert len(long.dropped[0].value) <= 80
    assert json.loads(json.dumps(long.metadata()))["dropped"][0]["value"] == long.dropped[0].value

    mixed = infer_spec(
        "x", FakeLLM({"subject": "s", "seed": 7, "foo": 1, "notes": None}), CONSTRAINTS
    )
    assert json.loads(json.dumps(mixed.metadata()))["subject_source"] == "llm"


def test_inferred_fields_feed_resolve_spec_below_cli():
    payload = {"subject": "k", "width": 64, "palette": "pico8"}
    result = infer_spec("k", FakeLLM(payload), CONSTRAINTS)
    spec = resolve_spec(preset=load_preset("gameboy"), inferred=result.fields, cli_overrides={})
    assert (spec.width, spec.palette) == (64, "pico8")
    spec = resolve_spec(
        preset=load_preset("gameboy"), inferred=result.fields, cli_overrides={"width": 16}
    )
    assert (spec.width, spec.palette) == (16, "pico8")
