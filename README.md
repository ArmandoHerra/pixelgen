# Pixelgen

`pixelgen` is a proof-of-concept Python CLI that turns a natural-language prompt into true pixel art, with an exact grid size and an exact palette. Claude turns the prompt into a structured spec and draws a grid of palette indices. Deterministic code then enforces the size and palette, applies dithering, outline and cleanup, and exports the PNG.

It needs only an Anthropic API key, no GPU. The generator backends are pluggable, so a diffusion backend can be added later.

## Getting Started

> Placeholder — setup instructions land with the first implementation wave.

```bash
pip install -e .
export ANTHROPIC_API_KEY=...
pixelgen generate "a red apple" --size 16 --palette pico8
```

## Development

> Placeholder — the development workflow lands with the first implementation wave.

```bash
pip install -e ".[dev]"
pytest
```

## License

TBD
