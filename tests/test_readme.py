"""Doc test for README.md (PXG-13): every check is derived from the CLI and the code.

Runs in-process through ``typer.testing.CliRunner`` under the conftest guard; it never builds
a real LLM client (the model is the scripted ``FakeLLM`` behind the ``make_llm`` seam).
"""

import importlib.resources
import re
import shlex
from pathlib import Path

import pytest
import typer.main
import yaml
from fakes import FakeLLM
from typer.testing import CliRunner

import pixelgen.backends.llm_grid as llm_grid
import pixelgen.cli as cli
import pixelgen.config as config
import pixelgen.llm as llm
import pixelgen.palettes as palettes
import pixelgen.presets as presets
from pixelgen.cli import app

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
README_LINES = README.splitlines()
runner = CliRunner()

_DASH = "—"
_FENCE = re.compile(r"^```([\w-]*)[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_SHELL_FENCES = {"bash", "sh", "shell", "console", "zsh"}
_FLAG_SPAN = re.compile(r"-{1,2}[A-Za-z][\w-]*(?:/-{1,2}[A-Za-z][\w-]*)*")
_NUMBER = re.compile(r"-?\d+(\.\d+)?")


# --- README parsing helpers -------------------------------------------------------------------


def _h2_names() -> list[str]:
    return [
        m.group(1).strip() for m in re.finditer(r"^## (.+)$", _mask_fences(README), re.MULTILINE)
    ]


def _mask_fences(text: str) -> str:
    """Same-length copy of ``text`` with fenced bodies blanked, so ``# comment`` is no heading."""
    return _FENCE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _section(heading: str, level: int = 2, text: str = README) -> str:
    """Body of ``<level> heading`` up to the next heading of the same or a higher level."""
    marks = "#" * level
    masked = _mask_fences(text)
    start = re.search(rf"^{marks} {re.escape(heading)}\s*$", masked, re.MULTILINE)
    assert start, f"README has no '{marks} {heading}' heading"
    end = re.compile(rf"^#{{1,{level}}} ", re.MULTILINE).search(masked, start.end())
    return text[start.end() : end.start() if end else len(text)]


def _fences(text: str) -> list[tuple[str, str, int]]:
    """(info string, body, 1-based line of the opening fence) for every fenced block."""
    return [
        (m.group(1), m.group(2), text.count("\n", 0, m.start()) + 1) for m in _FENCE.finditer(text)
    ]


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text)


def _commands_in(text: str, first_line: int = 1) -> list[tuple[int, list[str]]]:
    """Every fenced ``pixelgen ...`` shell line as (README line, argv without ``pixelgen``)."""
    found: list[tuple[int, list[str]]] = []
    for info, body, fence_line in _fences(text):
        if info not in _SHELL_FENCES:
            continue
        joined: list[tuple[int, str]] = []
        pending = ""
        pending_line = 0
        for offset, raw in enumerate(body.splitlines(), start=1):
            if not pending:
                pending_line = fence_line + offset
            line = raw.rstrip()
            if line.endswith("\\"):
                pending += line[:-1] + " "
                continue
            joined.append((pending_line, pending + line))
            pending = ""
        for number, line in joined:
            line = line.strip().removeprefix("$ ").strip()
            if not line or line.startswith("#"):
                continue
            tokens = shlex.split(line, comments=True)
            if tokens[:2] == ["uv", "run"]:
                tokens = tokens[2:]
            if tokens and tokens[0] == "pixelgen":
                found.append((first_line - 1 + number, tokens[1:]))
    return found


def _readme_commands() -> list[tuple[int, list[str]]]:
    return _commands_in(README)


def _table(section: str) -> list[list[str]]:
    """Body rows of the first markdown table in ``section`` (header and rule row dropped)."""
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    assert len(rows) >= 2, "no markdown table found"
    return rows[2:]


def _unticked(cell: str) -> str:
    return cell.replace("`", "").strip()


def _marker_block(kind: str) -> tuple[str, str]:
    """(name, body) of the fence following ``<!-- readme-test: <kind> name=<name> -->``."""
    match = re.search(
        rf"<!-- readme-test: {kind} name=(\S+) -->\n```\w*\n(.*?)^```", README, re.M | re.S
    )
    assert match, f"README has no '{kind}' marker followed by a fenced block"
    return match.group(1), match.group(2)


# --- CLI introspection helpers -----------------------------------------------------------------


def _group():
    return typer.main.get_command(app)


def _command_names() -> set[str]:
    return set(_group().commands)


def _option_params(cmd_name: str | None):
    command = _group() if cmd_name is None else _group().commands[cmd_name]
    return [p for p in command.params if p.param_type_name == "option"]


def _options(cmd_name: str | None) -> set[str]:
    found = {"--help"}
    for param in _option_params(cmd_name):
        found.update(param.opts)
        found.update(param.secondary_opts)
    return found


def _all_options() -> set[str]:
    found = _options(None)
    for name in _command_names():
        found |= _options(name)
    return found


def _primary_long(param) -> str:
    return next(opt for opt in param.opts if opt.startswith("--"))


def _flag_tokens(tokens: list[str]) -> list[str]:
    flags = []
    for token in tokens:
        if token.startswith("-") and token != "-" and not _NUMBER.fullmatch(token):
            flags.append(token.split("=", 1)[0])
    return flags


def _render_default(default) -> str:
    if default is None:
        return _DASH
    if isinstance(default, bool):
        return "true" if default else "false"
    return str(default)


# --- AC1 ---------------------------------------------------------------------------------------

ORDER = [
    "Install",
    "Quick start",
    "Examples",
    "Commands",
    "Configuration",
    "Palettes and presets",
    "Sizes above 32",
    "Costs",
    "Errors",
    "Out of scope",
    "Development",
    "License",
]


def test_readme_sections_in_order():
    names = _h2_names()
    missing = [n for n in ORDER if n not in names]
    assert not missing, f"missing H2 sections: {missing}"
    positions = [names.index(n) for n in ORDER]
    assert positions == sorted(positions), f"H2 sections out of order: {names}"
    commands = _section("Commands")
    assert [h for h in re.findall(r"^### (.+)$", _mask_fences(commands), re.M)] == [
        "generate",
        "refine",
        "palettes",
        "presets",
    ]
    how_to = _section("Palettes and presets")
    assert re.findall(r"^### (.+)$", how_to, re.M) == ["Adding a palette", "Adding a preset"]


def test_readme_has_no_placeholders():
    assert "Placeholder" not in README
    assert "TBD" not in README


# --- AC2 / AC3 ---------------------------------------------------------------------------------


def test_every_fenced_pixelgen_command_accepts_help():
    commands = _readme_commands()
    assert commands, "no fenced pixelgen commands found"
    covered = {argv[0] for _, argv in commands if argv}
    assert _command_names() <= covered, f"no fenced example for {_command_names() - covered}"
    failures = []
    for line, argv in commands:
        result = runner.invoke(app, [*argv, "--help"])
        if result.exit_code != 0:
            failures.append(
                f"README line {line}: pixelgen {shlex.join(argv)} --help -> exit {result.exit_code}"
            )
    assert not failures, "\n".join(failures)


def test_every_flag_on_a_fenced_command_exists_on_that_command():
    bad = []
    for line, argv in _readme_commands():
        name = argv[0] if argv and argv[0] in _command_names() else None
        known = _options(name)
        for flag in _flag_tokens(argv[1:] if name else argv):
            if flag not in known:
                bad.append(f"README line {line}: {flag} is not an option of {name or 'pixelgen'}")
    assert not bad, "\n".join(bad)


def test_every_inline_flag_exists_on_some_command():
    known = _all_options()
    bad = []
    for span in re.findall(r"`([^`\n]+)`", _strip_fences(README)):
        if _FLAG_SPAN.fullmatch(span):
            bad.extend(f"{part} (in `{span}`)" for part in span.split("/") if part not in known)
    assert not bad, f"inline flags that exist on no command: {bad}"


# --- AC4 ---------------------------------------------------------------------------------------


def test_every_command_has_an_example_and_every_option_is_documented():
    covered = {argv[0] for _, argv in _readme_commands() if argv}
    assert _command_names() <= covered, f"no example for {_command_names() - covered}"
    undocumented = []
    for name in [None, *sorted(_command_names())]:
        for param in _option_params(name):
            flag = _primary_long(param)
            if flag != "--help" and flag not in README:
                undocumented.append(f"{flag} ({name or 'pixelgen'})")
    assert not undocumented, f"options missing from the README: {undocumented}"


def test_generate_flag_table_defaults_match_the_cli():
    rows = _table(_section("generate", level=3))
    params = {_primary_long(p): p for p in _option_params("generate")}
    known = _options("generate")
    seen = set()
    for cells in rows:
        flags = re.findall(r"-{1,2}[A-Za-z][\w-]*", cells[0].replace("`", ""))
        assert flags, f"no flag in table row {cells}"
        for flag in flags:
            assert flag in known, f"{flag} is not an option of generate"
        primary = next(f for f in flags if f.startswith("--"))
        assert primary in params, f"{primary} is not a generate option"
        seen.add(primary)
        assert _unticked(cells[1]) == _render_default(params[primary].default), (
            f"{primary}: {_unticked(cells[1])!r} != {_render_default(params[primary].default)!r}"
        )
    assert seen == set(params), f"table rows differ from the CLI: {seen ^ set(params)}"


# --- AC5 ---------------------------------------------------------------------------------------


def test_install_section_covers_pip_pipx_and_uv():
    install = _section("Install")
    lines = [ln.strip() for _, body, _ in _fences(install) for ln in body.splitlines()]
    assert any(ln.startswith("pip install -e .") for ln in lines)
    assert any(ln.startswith("pipx install") for ln in lines)
    assert any(ln.startswith(("uv sync", "uv run")) for ln in lines)
    assert "pip install pixelgen" not in README, "PyPI claim: pip install pixelgen"
    assert "pipx install pixelgen" not in README, "PyPI claim: pipx install pixelgen"


# --- AC6 ---------------------------------------------------------------------------------------


def test_env_var_table_matches_code(monkeypatch, tmp_path):
    rows = {_unticked(c[0]): c for c in _table(_section("Configuration"))}
    assert set(rows) == {"ANTHROPIC_API_KEY", "PIXELGEN_MODEL", "PIXELGEN_HOME"}
    assert _unticked(rows["PIXELGEN_MODEL"][2]) == llm.DEFAULT_MODEL
    assert _unticked(rows["ANTHROPIC_API_KEY"][1]).startswith("yes")
    home = tmp_path / "h"
    monkeypatch.delenv("PIXELGEN_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    assert _unticked(rows["PIXELGEN_HOME"][2]) == "~/" + str(
        config.pixelgen_home().relative_to(home)
    )
    assert config.pixelgen_home() == home / ".pixelgen"


# --- AC7 ---------------------------------------------------------------------------------------


def test_costs_section_states_api_calls_without_prices():
    costs = _section("Costs")
    for needle in ("Anthropic API", "generate", "refine"):
        assert needle in costs, f"## Costs does not mention {needle!r}"
    for pattern in (r"\$\s?\d", r"\bUSD\b", r"\bcents?\b"):
        assert not re.search(pattern, README), f"README matches the price pattern {pattern}"


@pytest.mark.parametrize("command", ["palettes", "presets"])
def test_discovery_commands_need_no_api_key(monkeypatch, command):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, [command])
    assert result.exit_code == 0, result.output


# --- AC8 ---------------------------------------------------------------------------------------


def _names_on_line(prefix: str) -> set[str]:
    line = next((ln for ln in README_LINES if ln.startswith(prefix)), None)
    assert line, f"README has no line starting {prefix!r}"
    return set(re.findall(r"`([^`]+)`", line))


def test_shipped_names_match_loaders_and_no_hand_counts():
    assert _names_on_line("Shipped palettes:") == set(palettes.list_palettes())
    assert _names_on_line("Shipped presets:") == set(presets.list_presets())
    for pattern in (
        r"\b\d+\s+colou?rs?\b",
        r"\b(\d+|two|three|four|five|six)\s+(palettes|presets|commands|specs)\b",
    ):
        assert not re.search(pattern, README), f"hand-written count matches {pattern}"


# --- AC9 / AC10 --------------------------------------------------------------------------------


def _copy_package_data(sub: str, dest: Path) -> None:
    dest.mkdir(parents=True)
    for entry in (importlib.resources.files("pixelgen") / sub).iterdir():
        if entry.is_file():
            (dest / entry.name).write_bytes(entry.read_bytes())


def _documented_dir(section: str, name: str, ext: str, sub: str) -> Path:
    match = re.search(rf"(src/pixelgen/{sub})/<name>\.{ext}", section)
    assert match, f"section does not document src/pixelgen/{sub}/<name>.{ext}"
    directory = ROOT / match.group(1)
    assert directory.is_dir(), f"{match.group(1)} is not a directory in the repo"
    return directory


def _grid(side: int, colours: int) -> list[list[int]]:
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5) % colours
            for c in range(side)
        ]
        for r in range(side)
    ]


def test_add_a_palette_steps_work(monkeypatch, tmp_path):
    section = _section("Adding a palette", level=3)
    name, block = _marker_block("palette-example")
    real_dir = _documented_dir(section, name, "hex", "palettes")
    assert sorted(p.stem for p in real_dir.glob("*.hex")) == palettes.list_palettes()
    assert "pipx install --force ." in section
    colours = [
        [int(line[i : i + 2], 16) for i in (0, 2, 4)] for line in block.split() if line.strip()
    ]
    data_dir = tmp_path / "palettes"
    _copy_package_data("palettes", data_dir)
    (data_dir / f"{name}.hex").write_text(block, encoding="utf-8")
    generate_argv = next(
        argv
        for _, argv in _commands_in(section)
        if argv[0] == "generate" and argv[argv.index("--palette") + 1] == name
    )
    side = int(generate_argv[generate_argv.index("--size") + 1])
    fake = FakeLLM({}, {"grid": _grid(side, len(colours))})
    with monkeypatch.context() as mp:
        mp.setattr(palettes, "_palette_dir", lambda: data_dir)
        mp.chdir(tmp_path)
        assert name in palettes.list_palettes()
        assert palettes.load_palette(name).tolist() == colours
        listed = runner.invoke(app, ["palettes"])
        assert listed.exit_code == 0 and name in listed.output, listed.output
        mp.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
        mp.setattr(cli, "make_llm", lambda: fake)
        out = tmp_path / "out" / "x.png"
        result = runner.invoke(app, [*generate_argv, "-o", str(out)])
        assert result.exit_code == 0, (result.output, result.exception)
        assert out.is_file() and fake.call_count == 2
    assert name not in palettes.list_palettes()
    assert not (real_dir / f"{name}.hex").exists()


def test_add_a_preset_steps_work(monkeypatch, tmp_path):
    section = _section("Adding a preset", level=3)
    name, block = _marker_block("preset-example")
    real_dir = _documented_dir(section, name, "yaml", "presets")
    assert sorted(p.stem for p in real_dir.glob("*.yaml")) == presets.list_presets()
    data_dir = tmp_path / "presets"
    _copy_package_data("presets", data_dir)
    (data_dir / f"{name}.yaml").write_text(block, encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.setattr(presets, "_preset_dir", lambda: data_dir)
        assert name in presets.list_presets()
        assert presets.load_preset(name) == yaml.safe_load(block)
        listed = runner.invoke(app, ["presets"])
        assert listed.exit_code == 0 and name in listed.output, listed.output
    assert name not in presets.list_presets()
    assert not (real_dir / f"{name}.yaml").exists()


# --- AC11 --------------------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("`", "").replace("*", "")).strip()


def test_gt32_reason_is_quoted_from_backend_docstring():
    match = re.search(
        r"<!-- readme-test: gt32-reason -->\n(.*?)\n<!-- /readme-test -->", README, re.S
    )
    assert match, "README has no gt32-reason marker span"
    span = _normalize(match.group(1))
    assert len(span.split()) >= 20
    assert "Reason" in span
    assert span in _normalize(llm_grid.__doc__ or "")


# --- AC12 --------------------------------------------------------------------------------------

_BAD_GRID = {"grid": [[0] * 15 for _ in range(16)]}
_SRC_TEXT = "\n".join(
    p.read_text(encoding="utf-8") for p in (ROOT / "src" / "pixelgen").rglob("*.py")
)

SCENARIOS = {
    "No API key": {
        "argv": ["generate", "a red apple"],
        "key": False,
        "script": [],
        "token": "ANTHROPIC_API_KEY",
    },
    "Unknown palette": {
        "argv": ["generate", "a red apple", "--palette", "nope"],
        "key": True,
        "script": [{}],
        "token": "unknown palette",
    },
    "Unknown preset": {
        "argv": ["generate", "a red apple", "--preset", "nope"],
        "key": True,
        "script": [{}],
        "token": "unknown preset",
    },
    "Model output stays invalid": {
        "argv": ["generate", "a red apple", "--size", "16"],
        "key": True,
        "script": [{}, _BAD_GRID, _BAD_GRID, _BAD_GRID],
        "token": "after 3 attempts",
    },
    "Diffusion backend": {
        "argv": ["generate", "a red apple", "--backend", "diffusion"],
        "key": True,
        "script": [],
        "token": "diffusion",
    },
    "Nothing to refine": {
        "argv": ["refine", "add eyes"],
        "key": True,
        "script": [],
        "token": "no previous result",
        "empty_home": True,
    },
}


def _error_rows() -> dict[str, tuple[str, str]]:
    rows = {}
    for cells in _table(_section("Errors")):
        rows[_unticked(cells[0])] = (_unticked(cells[1]), cells[2].strip())
    return rows


def _in_src(excerpt: str) -> bool:
    """True when the excerpt occurs in the source; a number may stand for a formatted value."""
    parts = re.split(r"(\d+)", excerpt)
    pattern = "".join(
        r"(?:\d+|\{[^}]*\})" if part.isdigit() else re.escape(part) for part in parts if part
    )
    return re.search(pattern, _SRC_TEXT) is not None


def test_error_table_rows_match_scenarios():
    assert set(_error_rows()) == set(SCENARIOS)


@pytest.mark.parametrize("label", list(SCENARIOS))
def test_error_table_quotes_real_messages(label, monkeypatch, tmp_path):
    scenario = SCENARIOS[label]
    excerpt, exit_cell = _error_rows()[label]
    assert scenario["token"] in excerpt, f"{label}: excerpt lacks {scenario['token']!r}"
    assert excerpt.replace("pixelgen: error:", "").strip(), f"{label}: prefix-only excerpt"

    monkeypatch.chdir(tmp_path)
    if scenario.get("empty_home"):
        home = tmp_path / "empty-home"
        home.mkdir()
        monkeypatch.setenv("PIXELGEN_HOME", str(home))
    fake = FakeLLM(*scenario["script"])
    calls = []

    def make():
        calls.append(1)
        return fake

    monkeypatch.setattr(cli, "make_llm", make)
    if scenario["key"]:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, scenario["argv"])
    try:
        output = result.output + result.stderr
    except ValueError:  # runner without a separate stderr stream
        output = result.output
    if not scenario["key"]:
        assert not calls, "make_llm must not run without a key"
    assert "Traceback" not in output
    assert result.exit_code != 0
    assert int(exit_cell) == result.exit_code, f"{label}: README says {exit_cell}"
    assert excerpt in output, f"{label}: {excerpt!r} not in {output!r}"
    assert _in_src(excerpt), f"{label}: {excerpt!r} is not pixelgen-owned text"


# --- AC13 - AC15 -------------------------------------------------------------------------------


@pytest.mark.parametrize("item", ["diffusion", "animation", "sprite sheet", "GUI", "in-terminal"])
def test_out_of_scope_lists_the_five_items(item):
    assert item.lower() in _section("Out of scope").lower()


def test_readme_documents_aseprite_for_generate_and_refine():
    assert "--aseprite" in _section("generate", level=3)
    assert "--aseprite" in _section("refine", level=3)
    assert "aseprite" not in _section("Out of scope").lower()


def test_readme_examples_section_links_resolve():
    """PXG-14 D8: every ``examples/`` path lives in ``## Examples`` and exists on disk."""
    section = _section("Examples")
    outside = README.replace(section, "")
    assert "examples/" not in outside, "README references examples/ outside '## Examples'"
    paths = set(re.findall(r"examples/[\w@./-]+\.(?:png|json|md)", section))
    assert paths, "## Examples references no example files"
    missing = sorted(p for p in paths if not (ROOT / p).is_file())
    assert not missing, f"missing example files: {missing}"
    dirs = {m for m in re.findall(r"examples/([\w-]+)/", section)}
    assert len(dirs) >= 3, f"## Examples references only {sorted(dirs)}"


def test_license_section_is_mit_and_matches_license_file():
    body = _section("License")
    assert "MIT" in body and "LICENSE" in body
    first = (ROOT / "LICENSE").read_text(encoding="utf-8").splitlines()[0]
    assert first == "MIT License"


# --- PXG-19 ---

_POC = re.compile(r"proof[- ]of[- ]concept|\bpoc\b", re.IGNORECASE)


def _intro() -> str:
    masked = _mask_fences(README)
    m = re.search(r"^## ", masked, re.MULTILINE)
    return README[: m.start()] if m else README


def test_readme_has_no_proof_of_concept_framing():
    found = _POC.findall(README)
    assert not found, f"README carries proof-of-concept wording: {found}"


def test_readme_intro_matches_the_repository_description():
    intro = _intro()
    for claim in ("natural-language prompt", "true pixel art", "exact grid size", "exact palette"):
        assert claim in intro, f"README intro lost the product claim {claim!r}"
