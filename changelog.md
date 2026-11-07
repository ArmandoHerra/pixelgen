# Changelog

All notable changes are recorded here in the Keep a Changelog format; entries are written per cycle.

## [Unreleased]

### Cycle 4 — cheaper base images, seam first

- PXG-32 — the test harness refuses every outbound socket connection and name lookup, child Python processes included, and fails a keyed child that skips the guard; vendor key variables are cleared for tests.
- PXG-33 — `generate` and `refine` refuse an output target that is an existing directory with one `cannot write output` line, before the key check and before any paid call.
- PXG-34 — the test oracle gains `--png`: it compares a `.aseprite` file with its sibling PNG pixel for pixel and names the first differing pixel.
- PXG-35 — a raster base image is pixelated into a palette-exact sprite (a majority vote per block, then the usual quantize, dither and outline steps); `llm-grid` output is unchanged.
- PXG-36 — the examples checker reports a deeply nested `run.json` as one `FAIL` line instead of a traceback.
- PXG-37 — the provider seam: a per-provider config, a standard-library transport with bounded retries, and a fake transport for tests. Every key a backend needs is checked before any spend, and a key that is not printable ASCII is refused without echoing it. No provider ships yet.
- PXG-38 — a raster base image with an opaque backdrop is keyed to transparency before pixelation when the background is transparent.
- PXG-39 — sidecars are written as `schema_version` 2 with `provider`, `provider_model`, a per-step `usage` list and `dropped_fields`; `token_usage` stays; version 1 sidecars and `last.json` still load.
- PXG-40 — after a successful `generate` or `refine`, one `pixelgen: usage:` line on stderr lists the per-step usage; it shows no money figure and no colour.
- PXG-41 — the README documents the usage line, sidecar v2 and the missing-key message; this cycle-4 changelog section.
- PXG-42 — `--backend openrouter`, the first image provider (OpenRouter Images API, `openai/gpt-image-2` by default, `PIXELGEN_OPENROUTER_MODEL` to change it): one low-quality 1024×1024 image per run, cropped and pixelated; `refine` stays on the spec LLM.
- PXG-43 — a deeply nested `--from` sidecar or `last.json` is one `pixelgen: error:` line (exit 2 for `--from`, exit 1 for `last.json`), not a traceback, and nothing is spent.
- PXG-44 — `examples/` gains a live side-by-side of `--backend llm-grid` and `--backend openrouter` on three prompts at 16 and 32 px, with the measured cost per image; the examples checker prices each provider's usage separately and scans for OpenRouter keys.
- PXG-45 — the provider transport refuses a malformed URL with one error line that echoes nothing from it, reads at most 16 MiB of a reply, and never sends the localhost http request through `HTTP_PROXY`; the CLI never prints frame locals in a traceback.
- PXG-46 — CI installs from `uv.lock` with `uv sync --locked` on every test leg, the lint job and the LibreSprite check, so a lock that disagrees with `pyproject.toml` fails CI; `install-smoke` stays a plain `pip install .`; checkouts no longer keep the token; Dependabot proposes weekly Action and lock updates.
- Compatibility — pixelgen from cycles 1–3 rejects a `schema_version` 2 sidecar ("this pixelgen reads 1"), and so rejects `last.json` too; cycle 4 reads versions 1 and 2.

### Cycle 3 — Aseprite export

- PXG-25 — an option given no value (`--size`, `-o`, `--from`) or a flag given one (`--show-spec=1`, `--version=1`) now carries the same `(see 'pixelgen <verb> --help')` hint as every other parse error.
- PXG-24 — an independent standard-library `.aseprite` reader serves as the test oracle, anchored to vendored MIT Aseprite fixtures.
- PXG-26 — a standard-library writer for palette-indexed `.aseprite` files: one layer, one frame, the transparent slot after the palette, at most 255 colours.
- PXG-27 — `.aseprite` joins the all-or-nothing export set, and the sidecar records it as the optional `outputs.aseprite`; `schema_version` stays 1.
- PXG-29 — `generate` and `refine` take `--aseprite/--no-aseprite` (default off); default names and `-vN` versions step past a stray `.aseprite`, and a palette over 255 colours is refused before any API call.
- Compatibility — pixelgen from cycles 1 and 2 rejects a sidecar or `last.json` that records `outputs.aseprite` (unknown keys are forbidden); sidecars written without `--aseprite` are unchanged.

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
