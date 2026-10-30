import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _names(reqs):
    return {re.match(r"^[A-Za-z0-9._-]+", r).group(0).lower().replace("_", "-") for r in reqs}


def test_runtime_dependencies():
    deps = DATA["project"]["dependencies"]
    assert _names(deps) == {"typer", "rich", "pillow", "numpy", "anthropic", "pyyaml", "pydantic"}
    pyd = [d for d in deps if d.lower().startswith("pydantic")][0]
    assert ">=2" in pyd


def test_dev_extra_and_group_are_pytest_and_ruff_only():
    assert _names(DATA["project"]["optional-dependencies"]["dev"]) == {"pytest", "ruff"}
    assert _names(DATA["dependency-groups"]["dev"]) == {"pytest", "ruff"}


def test_build_system_is_hatchling_only():
    assert _names(DATA["build-system"]["requires"]) == {"hatchling"}


def test_license():
    assert DATA["project"]["license"] == "MIT"
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Armando Herra" in text


def test_build_config_ships_no_tests_in_the_wheel():
    # Hatchling auto-detects src/pixelgen, so the wheel carries only pixelgen/** today. A
    # [tool.hatch...] table (force-include, sources, ...) is how tests/ content could join it.
    assert set(DATA["tool"]) == {"pytest", "ruff"}
