from importlib.metadata import entry_points, version

import typer
import typer.core
import typer.main
from typer.testing import CliRunner

from pixelgen.cli import app

ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def test_help_exits_zero_and_names_the_product():
    result = CliRunner().invoke(app, ["--help"], env=ENV)
    assert result.exit_code == 0
    assert "pixel art" in result.output
    assert "--version" in result.output


def test_version_prints_package_metadata_version():
    result = CliRunner().invoke(app, ["--version"], env=ENV)
    assert result.exit_code == 0
    assert result.output.strip() == f"pixelgen {version('pixelgen')}"


def test_console_script_targets_cli_app():
    eps = [e for e in entry_points(group="console_scripts") if e.name == "pixelgen"]
    assert len(eps) == 1
    assert eps[0].value == "pixelgen.cli:app"


def test_app_is_a_command_group():
    assert isinstance(app, typer.Typer)
    assert isinstance(typer.main.get_command(app), typer.core.TyperGroup)
