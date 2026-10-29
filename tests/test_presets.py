"""PXG-05: packaged presets and their loader."""

import pytest

from pixelgen import presets
from pixelgen.errors import PresetError
from pixelgen.presets import PRESET_FORBIDDEN_KEYS, list_presets, load_preset, parse_preset

PALETTES = {"pico8", "gameboy", "nes", "endesga32", "grayscale4"}

TABLE = {
    "gameboy": {
        "palette": "gameboy",
        "width": 32,
        "height": 32,
        "dither": "bayer",
        "outline": False,
        "notes": "Classic handheld look, high contrast, simple shapes",
    },
    "nes": {
        "palette": "nes",
        "width": 16,
        "height": 16,
        "dither": "none",
        "outline": True,
        "notes": (
            "8-bit console sprite: chunky readable shapes, flat colour areas, "
            "at most three colours plus outline per sprite"
        ),
    },
    "16bit-rpg": {
        "palette": "endesga32",
        "width": 32,
        "height": 32,
        "dither": "floyd-steinberg",
        "outline": True,
        "notes": (
            "16-bit console RPG sprite: three to four shading tones per material, "
            "light from the top-left, dark outline"
        ),
    },
    "icon": {
        "palette": "pico8",
        "width": 16,
        "height": 16,
        "dither": "none",
        "outline": True,
        "background": "transparent",
        "view": "front",
        "notes": (
            "Single centred object that reads at a glance at small size, "
            "bold silhouette, minimal detail"
        ),
    },
}


def test_list_presets_returns_the_four_sorted():
    assert list_presets() == ["16bit-rpg", "gameboy", "icon", "nes"]
    assert list_presets() == list_presets()


def test_gameboy_preset_is_the_plan_example():
    assert load_preset("gameboy") == {
        "palette": "gameboy",
        "width": 32,
        "height": 32,
        "dither": "bayer",
        "outline": False,
        "notes": "Classic handheld look, high contrast, simple shapes",
    }


@pytest.mark.parametrize("name", ["nes", "16bit-rpg", "icon", "gameboy"])
def test_shipped_presets_match_the_table(name):
    assert load_preset(name) == TABLE[name]


def test_shipped_presets_use_known_palettes():
    for name in list_presets():
        data = load_preset(name)
        assert data["palette"] in PALETTES, (name, data["palette"])
        assert not (set(data) & PRESET_FORBIDDEN_KEYS)


def test_unknown_key_is_rejected():
    with pytest.raises(PresetError) as info:
        parse_preset("colour: red\nwidth: 32\n", "x")
    assert "preset 'x':" in str(info.value)
    assert "colour" in str(info.value)


@pytest.mark.parametrize(
    "text,key",
    [("subject: knight\n", "subject"), ("seed: 7\n", "seed"), ("preset: nes\n", "preset")],
)
def test_forbidden_keys_are_rejected(text, key):
    with pytest.raises(PresetError) as info:
        parse_preset(text, "x")
    assert key in str(info.value)


def test_notes_is_allowed():
    assert parse_preset('notes: "soft"\n', "x") == {"notes": "soft"}


@pytest.mark.parametrize(
    "text",
    ["width: 200\n", "dither: sierra\n", "- a\n- b\n", "", "width: [\n"],
)
def test_bad_preset_text_raises_preset_error(text):
    with pytest.raises(PresetError) as info:
        parse_preset(text, "x")
    message = str(info.value)
    assert "\n" not in message
    assert message.startswith("preset 'x':")
    assert "pydantic" not in message.lower()


def test_python_tags_are_refused():
    with pytest.raises(PresetError):
        parse_preset('width: !!python/object/apply:builtins.int ["32"]\n', "x")


@pytest.mark.parametrize("name", ["nope", "../spec"])
def test_unknown_preset_names_available(name):
    with pytest.raises(PresetError) as info:
        load_preset(name)
    message = str(info.value)
    assert "unknown preset" in message
    for available in ("16bit-rpg", "gameboy", "icon", "nes"):
        assert available in message


def test_dropped_preset_is_discovered_through_the_seam(tmp_path, monkeypatch):
    (tmp_path / "zz.yaml").write_text("dither: bayer\n", encoding="utf-8")
    (tmp_path / "aa.yaml").write_text("width: 48\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    monkeypatch.setattr(presets, "_preset_dir", lambda: tmp_path)
    assert list_presets() == ["aa", "zz"]
    assert load_preset("aa") == {"width": 48}
    with pytest.raises(PresetError) as info:
        load_preset("gameboy")
    assert "aa, zz" in str(info.value)
