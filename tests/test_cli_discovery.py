"""PXG-10: the `palettes` and `presets` discovery commands."""

import re

import pytest
from typer.testing import CliRunner

from pixelgen import palettes, presets
from pixelgen.cli import app
from pixelgen.spec import ArtSpec

runner = CliRunner()

PINNED_COUNTS = {"pico8": 16, "gameboy": 4, "nes": 54, "endesga32": 32, "grayscale4": 4}
PINNED_PRESETS = ["gameboy", "nes", "16bit-rpg", "icon"]


def _env(**overrides):
    env = {
        "NO_COLOR": None,
        "FORCE_COLOR": None,
        "TTY_COMPATIBLE": None,
        "TTY_INTERACTIVE": None,
        "COLUMNS": "80",
    }
    env.update(overrides)
    return env


def _piped_env():
    return _env()


def _truecolor_env():
    return _env(FORCE_COLOR="1", COLORTERM="truecolor", TERM="xterm-256color")


def _no_color_env():
    return _env(FORCE_COLOR="1", COLORTERM="truecolor", TERM="xterm-256color", NO_COLOR="1")


def _hex(colour) -> str:
    r, g, b = (int(c) for c in colour)
    return f"#{r:02x}{g:02x}{b:02x}"


def _blocks(output: str) -> dict[str, tuple[int, list[str]]]:
    """name -> (count read from the row line, hex tokens in order from its block)."""
    names = palettes.list_palettes()
    lines = output.splitlines()
    starts: list[tuple[str, int, int]] = []
    for name in names:
        pattern = re.compile(rf"^\W*\b{re.escape(name)}\b\W+(\d+)\b")
        matches = [(i, m) for i, line in enumerate(lines) if (m := pattern.match(line))]
        assert len(matches) == 1, f"{name}: expected exactly one row line, got {len(matches)}"
        i, m = matches[0]
        starts.append((name, i, int(m.group(1))))
    ordered = sorted(starts, key=lambda s: s[1])
    result = {}
    for idx, (name, start, count) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(lines)
        tokens = re.findall(r"#[0-9a-f]{6}", "\n".join(lines[start:end]))
        result[name] = (count, tokens)
    return result


def _assert_rows_and_hex(output: str) -> None:
    blocks = _blocks(output)
    positions = [output.index(name) for name in palettes.list_palettes()]
    assert positions == sorted(positions)
    for name in palettes.list_palettes():
        colours = palettes.load_palette(name)
        count, tokens = blocks[name]
        assert count == len(colours)
        assert tokens == [_hex(c) for c in colours]


def test_palettes_lists_every_palette_with_its_loaded_count():
    assert {n: len(palettes.load_palette(n)) for n in palettes.list_palettes()} == PINNED_COUNTS
    result = runner.invoke(app, ["palettes"], env=_piped_env())
    assert result.exit_code == 0, result.output
    blocks = _blocks(result.output)
    assert {n: c for n, (c, _) in blocks.items()} == PINNED_COUNTS
    order = [n for n, _ in sorted(blocks.items(), key=lambda kv: result.output.index(kv[0]))]
    assert order == palettes.list_palettes()
    for name, (count, _) in blocks.items():
        assert count == len(palettes.load_palette(name))


def test_palettes_piped_output_lists_every_hex_per_palette_without_escapes():
    result = runner.invoke(app, ["palettes"], env=_piped_env())
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    _assert_rows_and_hex(result.output)
    assert sum(len(t) for _, t in _blocks(result.output).values()) == 110


def test_palettes_piped_output_ignores_ambient_tty_compatible(monkeypatch):
    plain = runner.invoke(app, ["palettes"], env=_piped_env())
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    again = runner.invoke(app, ["palettes"], env=_piped_env())
    assert again.exit_code == 0, again.output
    assert "\x1b" not in again.output
    assert again.output == plain.output


def test_palettes_forced_truecolor_paints_every_swatch_background():
    result = runner.invoke(app, ["palettes"], env=_truecolor_env())
    assert result.exit_code == 0, result.output
    seen = set()
    for seq in re.findall(r"\x1b\[[0-9;]*m", result.output):
        for m in re.finditer(r"48;2;(\d+);(\d+);(\d+)", seq):
            seen.add(tuple(int(g) for g in m.groups()))
    expected = {
        tuple(int(c) for c in colour)
        for name in palettes.list_palettes()
        for colour in palettes.load_palette(name)
    }
    assert expected
    assert expected <= seen, f"missing backgrounds: {sorted(expected - seen)[:5]}"


def test_palettes_no_color_strips_colour_but_keeps_names_counts_hex():
    result = runner.invoke(app, ["palettes"], env=_no_color_env())
    assert result.exit_code == 0, result.output
    assert "38;2;" not in result.output
    assert "48;2;" not in result.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    _assert_rows_and_hex(plain)


def _expected_lines(name: str, settings: dict) -> list[str]:
    def fmt(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    return [f"  {k}: {fmt(settings[k])}" for k in ArtSpec.model_fields if k in settings]


def test_presets_lists_every_preset_with_all_its_settings():
    assert presets.list_presets() == sorted(PINNED_PRESETS)
    result = runner.invoke(app, ["presets"], env=_piped_env())
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    heads = [i for i, line in enumerate(lines) if line and not line.startswith(" ")]
    assert [lines[i] for i in heads] == presets.list_presets()
    for n, i in enumerate(heads):
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        body = [line for line in lines[i + 1 : end] if line.strip()]
        settings = presets.load_preset(lines[i])
        assert settings
        assert body == _expected_lines(lines[i], settings)


def test_presets_prints_values_verbatim_in_artspec_field_order(monkeypatch):
    notes = "[red]x[/red] " + "y" * 150
    monkeypatch.setattr(presets, "list_presets", lambda: ["probe"])
    monkeypatch.setattr(
        presets, "load_preset", lambda name: {"notes": notes, "outline": False, "width": 16}
    )
    result = runner.invoke(app, ["presets"], env=_piped_env())
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert lines == ["probe", "  width: 16", "  outline: false", f"  notes: {notes}"]


@pytest.mark.parametrize("cmd", ["palettes", "presets"])
def test_discovery_commands_exit_zero_when_piped(cmd):
    assert runner.invoke(app, [cmd], env=_piped_env()).exit_code == 0
