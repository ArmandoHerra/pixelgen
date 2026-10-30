# Aseprite-made fixtures: provenance (PXG-24, OR-20)

These files are Aseprite's own output, made from `generate.lua` by a personal source
build on the operator's machine. Aseprite itself is never committed and never runs in
CI.
This file is written by `tests/regen_aseprite_made.py`, never by hand.

- aseprite --version: Aseprite 1.3.18.6-dev
- aseprite binary sha256: 41e10491190759eda326e8be114618e2247ce04e0831612ebc26209fc18a22ee

Regenerate (local only): `PIXELGEN_ASEPRITE=/path/to/aseprite uv run python -I tests/regen_aseprite_made.py --out tests/fixtures/aseprite/aseprite_made`

| file | sha256 |
|---|---|
| `black0.aseprite` | b409b8c4e771f9c024151ba6773efca079634b1d9879b1293f3c10495e032fed |
| `black0.png` | 1dcaa4ae6c73c7ec09ee03da816501fa5ff940c5bfb615ae8ffb9909d24a4801 |
| `nonsquare.aseprite` | c57b9287b711feef4d4e9a93e3ae7fcc20b0ebf78bfb288e1a6d68d45abc299c |
| `nonsquare.png` | 547136fadf93c1ea81d49feb90a34b8599873de8f71ad948c356af7635c850d7 |
| `grid128.aseprite` | 4bbbe8683781356d30a9943556370c33be2d65b0312ee4c9de4b2909f33ee781 |
| `grid128.png` | b6de42c5845f6a8bafa8f9388fc269b4e1d7f1443a43ab53bb2613ee57edfa9c |
| `offset_cel.aseprite` | dbb6703f9d52786c6e927e6169e3036d78754f21ad73e904db858a64f935829e |
| `offset_cel.png` | 153a599c29588ba07f9358eebf27918842849c6cb1513a75d3edca4122827f98 |
| `generate.lua` | 57f34b24df1547a1f5ce9a12177963f9e2925c9cde533db766b8cc5775e8f946 |
