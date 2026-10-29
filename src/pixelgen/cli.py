"""Command-line entry point. Later specs add verbs with ``@app.command()``."""

from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pixelgen import __version__, palettes, presets
from pixelgen.spec import ArtSpec

app = typer.Typer(
    name="pixelgen",
    help="Turn a natural-language prompt into true pixel art.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pixelgen {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Turn a natural-language prompt into true pixel art."""


def _console() -> Console:
    """Built per invocation so Rich reads NO_COLOR/FORCE_COLOR/COLUMNS at call time."""
    return Console()


def _swatch_text(colours: Any) -> Text:
    """One `#rrggbb` token per colour, painted on its own colour, then a plain space."""
    text = Text()
    for red, green, blue in colours:
        r, g, b = int(red), int(green), int(blue)
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        luma = (299 * r + 587 * g + 114 * b) / 1000
        fg = "black" if luma >= 128 else "white"
        text.append(hex_code, style=f"{fg} on {hex_code}")
        text.append(" ")
    return text


@app.command("palettes")
def palettes_command() -> None:
    """List the built-in palettes with colour counts and swatches."""
    table = Table()
    table.add_column("Name", no_wrap=True)
    table.add_column("Colours", no_wrap=True, justify="right")
    table.add_column("Swatches", min_width=8)
    for name in palettes.list_palettes():
        colours = palettes.load_palette(name)
        table.add_row(name, str(len(colours)), _swatch_text(colours))
    _console().print(table)


def _preset_settings(name: str) -> dict[str, Any]:
    loaded = presets.load_preset(name)
    if hasattr(loaded, "model_dump"):
        return dict(loaded.model_dump(exclude_unset=True))
    return dict(loaded)


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@app.command("presets")
def presets_command() -> None:
    """List the built-in style presets and every setting each one applies."""
    console = _console()
    for name in presets.list_presets():
        settings = _preset_settings(name)
        console.print(name, markup=False, highlight=False, soft_wrap=True)
        for key in ArtSpec.model_fields:
            if key in settings:
                line = f"  {key}: {_format_value(settings[key])}"
                console.print(line, markup=False, highlight=False, soft_wrap=True)
        console.print()
