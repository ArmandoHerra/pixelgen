# Changelog

All notable changes are recorded here in the Keep a Changelog format; entries are written per cycle.

## [Unreleased]

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
