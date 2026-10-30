"""Command-line entry point. Later specs add verbs with ``@app.command()``."""

import functools
import os
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, NoReturn, ParamSpec, TypeVar

import typer
import typer.core
import typer.main
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pixelgen import __version__, palettes, pipeline, presets
from pixelgen.backends.base import available_backends
from pixelgen.errors import MissingAPIKeyError, PixelgenError, SpecError
from pixelgen.llm import AnthropicLLM
from pixelgen.spec import SIZE_MAX, SIZE_MIN, ArtSpec

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2

SCALE_MIN = 1
SCALE_MAX = 64

P = ParamSpec("P")
R = TypeVar("R")

# The LLM seam: tests monkeypatch ``pixelgen.cli.make_llm``.
make_llm = AnthropicLLM.from_env


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _error_line(exc: PixelgenError) -> str:
    text = _collapse(str(exc)) or _collapse(getattr(exc, "default_message", ""))
    return f"pixelgen: error: {text or type(exc).__name__}"


def _oserror_line(exc: OSError) -> str:
    reason = _collapse(exc.strerror or "") or type(exc).__name__
    path = exc.filename2 if exc.filename2 is not None else exc.filename
    if path is not None:
        return f"pixelgen: error: cannot access '{_collapse(str(path))}': {reason}"
    return f"pixelgen: error: {reason}"


def _fail(line: str, code: int) -> NoReturn:
    typer.echo(line, err=True)
    raise typer.Exit(code)


def usage_error(message: str) -> NoReturn:
    """Reject the user's input (exit 2) before anything ran or any money was spent."""
    _fail(f"pixelgen: error: {_collapse(message)}", EXIT_USAGE)


@contextmanager
def usage_phase() -> Iterator[None]:
    """Any PixelgenError raised inside is a usage error (exit 2): the phase decides."""
    try:
        yield
    except PixelgenError as exc:
        _fail(_error_line(exc), EXIT_USAGE)


# The Click usage-error base class, resolved through typer's public BadParameter so that
# cli.py never imports click or typer._click (absent as a declared dependency in Typer 0.27).
_USAGE_ERROR = next(c for c in typer.BadParameter.__mro__ if c.__name__ == "UsageError")


def _parse_error_line(exc: Any) -> str:
    text = _collapse(exc.format_message()) or type(exc).__name__
    ctx = getattr(exc, "ctx", None)
    hint = f" (see '{_collapse(ctx.command_path)} --help')" if ctx is not None else ""
    return f"pixelgen: error: {text}{hint}"


@contextmanager
def _parse_errors_as_usage() -> Iterator[None]:
    """Turn a Click/Typer parse error into one product line and exit 2 (X-PXG-41)."""
    try:
        yield
    except _USAGE_ERROR as exc:
        if type(exc).__name__ == "NoArgsIsHelpError":
            raise  # the help is already on stdout; Typer prints nothing more
        _fail(_parse_error_line(exc), EXIT_USAGE)


@contextmanager
def _attach_parse_ctx(ctx: Any) -> Iterator[None]:
    """Give a context-less Click usage error the context of the command being parsed."""
    try:
        yield
    except _USAGE_ERROR as exc:
        if getattr(exc, "ctx", None) is None:
            exc.ctx = ctx
        raise


class PixelgenCommand(typer.core.TyperCommand):
    def parse_args(self, ctx: Any, args: Any) -> Any:
        with _attach_parse_ctx(ctx):
            return super().parse_args(ctx, args)


class PixelgenGroup(typer.core.TyperGroup):
    def parse_args(self, ctx: Any, args: Any) -> Any:
        with _attach_parse_ctx(ctx):
            return super().parse_args(ctx, args)

    def make_context(self, info_name: Any, args: Any, parent: Any = None, **extra: Any) -> Any:
        with _parse_errors_as_usage():
            return super().make_context(info_name, args, parent=parent, **extra)

    def invoke(self, ctx: Any) -> Any:
        with _parse_errors_as_usage():
            return super().invoke(ctx)


class _NoColorTyper(typer.Typer):
    """Typer forces a terminal at import under CI variables; NO_COLOR must win per invocation."""

    @property
    def rich_markup_mode(self) -> Any:
        if os.environ.get("NO_COLOR", ""):
            return None
        return self._pixelgen_markup_mode

    @rich_markup_mode.setter
    def rich_markup_mode(self, value: Any) -> None:
        self._pixelgen_markup_mode = value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # On Typer's plain path a bare invocation writes the help to stderr; keep it on
        # stdout, exit 2 (X-PXG-52).
        if self.rich_markup_mode is None:
            argv = args[0] if args else kwargs.get("args")
            if argv is None:
                argv = sys.argv[1:]
            if not list(argv):
                prog = kwargs.get("prog_name") or os.path.basename(sys.argv[0])
                ctx = typer.main.get_command(self).make_context(prog, [], resilient_parsing=True)
                typer.echo(ctx.get_help())
                sys.exit(EXIT_USAGE)
        return super().__call__(*args, **kwargs)


app = _NoColorTyper(
    name="pixelgen",
    help="Turn a natural-language prompt into true pixel art.",
    no_args_is_help=True,
    add_completion=False,
    cls=PixelgenGroup,
)


def error_boundary(func: Callable[P, R]) -> Callable[P, R]:
    """Turn a PixelgenError or OSError into one stderr line and exit 1; never a traceback.

    Other exceptions are programming bugs and stay visible.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except PixelgenError as exc:
            _fail(_error_line(exc), EXIT_RUNTIME)
        except OSError as exc:
            _fail(_oserror_line(exc), EXIT_RUNTIME)

    return wrapper


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise MissingAPIKeyError()


def print_summary(result: pipeline.PipelineResult) -> None:
    out = result.export
    spec = result.spec
    typer.echo(
        f"Wrote {out.png_path} ({spec.width}\u00d7{spec.height}), "
        f"preview {out.preview_path}, sidecar {out.sidecar_path}"
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


@app.command("palettes", cls=PixelgenCommand)
@error_boundary
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


@app.command("presets", cls=PixelgenCommand)
@error_boundary
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


def _print_spec(spec: ArtSpec) -> None:
    table = Table(title="Resolved spec")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value")
    for key in ArtSpec.model_fields:
        table.add_row(key, _format_value(getattr(spec, key)))
    _console().print(table)


@app.command("generate", cls=PixelgenCommand)
@error_boundary
def generate(
    prompt: str = typer.Argument(..., help="What to draw, in plain language."),
    size: int | None = typer.Option(
        None, "--size", help=f"Width and height in pixels ({SIZE_MIN}-{SIZE_MAX})."
    ),
    width: int | None = typer.Option(
        None, "--width", help=f"Width in pixels ({SIZE_MIN}-{SIZE_MAX})."
    ),
    height: int | None = typer.Option(
        None, "--height", help=f"Height in pixels ({SIZE_MIN}-{SIZE_MAX})."
    ),
    palette: str | None = typer.Option(None, "--palette", help="Palette name."),
    preset: str | None = typer.Option(None, "--preset", help="Style preset name."),
    dither: str | None = typer.Option(None, "--dither", help="none, floyd-steinberg or bayer."),
    outline: bool | None = typer.Option(None, "--outline/--no-outline", help="Draw an outline."),
    background: str | None = typer.Option(None, "--background", help="transparent or solid."),
    view: str | None = typer.Option(None, "--view", help="front, side, top-down or isometric."),
    seed: int | None = typer.Option(None, "--seed", help="Seed recorded in the spec."),
    scale: int = typer.Option(8, "--scale", help="Preview upscale factor (1-64)."),
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Output path (overwrites).")
    ] = None,
    backend: str = typer.Option("llm-grid", "--backend", help="Generator backend."),
    show_spec: bool = typer.Option(False, "--show-spec", help="Print the resolved spec first."),
) -> None:
    """Generate pixel art from PROMPT."""
    with usage_phase():
        if not prompt.strip():
            usage_error("prompt is empty")
        if size is not None and (width is not None or height is not None):
            usage_error("--size cannot be combined with --width or --height")
        for flag, value in (("--size", size), ("--width", width), ("--height", height)):
            if value is not None and not SIZE_MIN <= value <= SIZE_MAX:
                usage_error(f"{flag} must be between {SIZE_MIN} and {SIZE_MAX}")
        if not SCALE_MIN <= scale <= SCALE_MAX:
            usage_error(f"--scale must be between {SCALE_MIN} and {SCALE_MAX}")
        if backend not in available_backends():
            usage_error(
                f"unknown backend '{backend}'; available: {', '.join(available_backends())}"
            )
        if output is not None and output.is_dir():
            usage_error(f"--output '{output}' is a directory; pass a file path")
        if size is not None:
            width = height = size
        cli_overrides: dict[str, Any] = {
            "width": width,
            "height": height,
            "palette": palette,
            "preset": preset,
            "dither": dither,
            "outline": outline,
            "background": background,
            "view": view,
            "seed": seed,
        }
        pipeline.build_spec(prompt, preset, cli_overrides, llm=None)

    pipeline.ensure_backend_available(backend)
    resolved_output = pipeline.preflight_generate(prompt, output, scale)

    _require_api_key()
    llm = make_llm()
    generator = pipeline.get_generator(backend, llm)
    result = pipeline.run_generate(
        prompt,
        llm=llm,
        generator=generator,
        backend_name=backend,
        preset_name=preset,
        cli_overrides=cli_overrides,
        output=resolved_output,
        overwrite=output is not None,
        scale=scale,
        on_spec=_print_spec if show_spec else None,
    )
    print_summary(result)


@app.command("refine", cls=PixelgenCommand)
@error_boundary
def refine(
    instruction: str = typer.Argument(..., help="The edit to make, in plain language."),
    from_: Annotated[
        Path | None,
        typer.Option("--from", help="Sidecar .json to refine (default: the last result)."),
    ] = None,
) -> None:
    """Edit the last result (or --from a sidecar) and write the next -vN version."""
    source = None
    with usage_phase():
        instruction = instruction.strip()
        if not instruction:
            raise SpecError("refine needs a non-empty INSTRUCTION")
        if from_ is not None:
            source = pipeline.load_refine_source(from_)
    if source is None:
        source = pipeline.load_refine_source(None)
    pipeline.preflight_refine(source)

    _require_api_key()
    llm = make_llm()
    outcome = pipeline.run_refine(source, instruction, llm=llm)
    print_summary(outcome.result)
    if outcome.ignored:
        typer.echo(
            f"pixelgen: note: refine keeps the source's {', '.join(outcome.ignored)}; "
            "that requested change was ignored",
            err=True,
        )
