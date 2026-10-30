# Changelog

All notable changes are recorded here in the Keep a Changelog format; entries are written per cycle.

## [Unreleased]

### Cycle 3 — Aseprite export

- PXG-25 — an option given no value (`--size`, `-o`, `--from`) or a flag given one (`--show-spec=1`, `--version=1`) now carries the same `(see 'pixelgen <verb> --help')` hint as every other parse error.

### Cycle 2 — CLI contract hardening

- PXG-15 — `-o` drops only a lowercase trailing `.png`, so `-o x.PNG` writes `x.PNG.png` and never overwrites `x.png`.
- PXG-16 — the examples checker now catches keys hidden in JSON-escaped strings, inflated PNG text chunks, huge integers and key-named slug directories, and its no-LLM import check inspects runtime imports.
- PXG-17 — CI runs `ruff check` and `ruff format --check`, and a standard-library test fails when `uv.lock` drifts from `pyproject.toml`.
- PXG-18 — every Click/Typer parse error prints one `pixelgen: error:` line with a `--help` hint and exits 2; help and `--version` output is unchanged.
- PXG-19 — the README intro no longer describes the project as a proof of concept.
- PXG-20 — the PXG-08 transport-error retry test scripts a single FakeLLM item, so widening the retry catch now fails it.
- PXG-21 — write errors name the path the user asked for, and size errors read `--size must be between 8 and 128`.
- PXG-22 — `NO_COLOR` is honoured under `GITHUB_ACTIONS`, `FORCE_COLOR` and `PY_COLORS` without patching private Typer state, and the no-args help stays on stdout with exit 2.
- PXG-23 — contract tests pinning the sweep-proven CLI points; this cycle-2 changelog section.

### Cycle 1 — proof of concept

- PXG-01 — installable `pixelgen` package, the `ArtSpec` model, the no-network test harness and the ruff lint gate.
- PXG-02 — CI runs the suite on Python 3.11 to 3.14 with no API secret, plus an install smoke job.
- PXG-03 — load, list and validate `.hex` palettes; five palettes shipped.
- PXG-04 — deterministic pixelation pipeline (size, palette, dithering, outline, cleanup) as pure NumPy functions.
- PXG-05 — YAML style presets and the spec merge precedence (defaults < preset < inferred < flags); unknown preset keys rejected.
- PXG-06 — thin Anthropic JSON client with bounded retries, safe errors and the default model `claude-sonnet-5-5`.
- PXG-07 — export of the native PNG, the upscaled preview, the JSON sidecar and `last.json` under `PIXELGEN_HOME`.
- PXG-08 — generator interface, the `llm-grid` backend with validated retries and the above-32 strategy, and the diffusion stub.
- PXG-09 — inference of an `ArtSpec` from the natural-language prompt via the model.
- PXG-10 — `pixelgen palettes` and `pixelgen presets` discovery commands.
- PXG-11 — `pixelgen generate` end to end with every flag, one-line errors, phase-based exit codes and pre-spend output checks.
- PXG-12 — `pixelgen refine` with `--from`, the `last.json` fallback and `-vN` versioned outputs.
- PXG-13 — README rewrite with a doc test; cycle-1 changelog entry.
- PXG-14 — offline examples checker and three live sample generations (plus one refine) in `examples/`, with the README `## Examples` section.
