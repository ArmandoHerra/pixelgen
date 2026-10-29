"""Command-line entry point. Later specs add verbs with ``@app.command()``."""

import typer

from pixelgen import __version__

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
