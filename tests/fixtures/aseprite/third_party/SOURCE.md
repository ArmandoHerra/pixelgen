# Vendored third-party fixtures (PXG-24)

These files are byte-identical copies from asefile, an MIT-licensed Rust reader for Aseprite files.
They anchor `tests/aseprite_oracle.py` against `.aseprite` bytes that Aseprite itself wrote.

- Upstream: https://github.com/alpine-alpaca/asefile
- Commit: 2274c354cea6764f85597252a0d2228e64709348 (branch `main`, 2024-07-21)
- Fetched: 2026-10-30
- Licence: the upstream root `LICENSE` is "The MIT License (MIT)", "Copyright (c) 2020 Alpine Alpaca
  Games", and `Cargo.toml` declares `license = "MIT"`. `tests/data/` carries no separate licence or
  notice. The upstream `LICENSE` is vendored beside the fixtures, unmodified.
- The `.aseprite` files are read as test inputs only. The oracle never writes them.

## Files

| file | sha256 |
|---|---|
| `indexed.aseprite` | a068ac5fe641df53601c14c14850275bb94ca4407c2f5032cc688e917c9a109f |
| `indexed_01.png` | bf0b707888b34addfb69b07e32760fef3303b0dd5f0bd29d074c9c189dc1a620 |
| `palette.aseprite` | d8b904e12a76241f4470de8e639b86961ebfdaa9f1c06dfcc2a4f3e02c1a946e |
| `LICENSE` | f33b372b56866e612afc83c26781ff9102282b37ebf628ecec33d8e72dbbc1ed |

Upstream paths: `tests/data/indexed.aseprite`, `tests/data/indexed_01.png`,
`tests/data/palette.aseprite`, and the root `LICENSE`. `SOURCE.md` itself is not hashed.

## Re-fetch

```bash
d=$(mktemp -d) && git clone https://github.com/alpine-alpaca/asefile "$d/asefile"
git -C "$d/asefile" checkout 2274c354cea6764f85597252a0d2228e64709348
cp "$d"/asefile/tests/data/{indexed.aseprite,indexed_01.png,palette.aseprite} "$d/asefile/LICENSE" .
sha256sum indexed.aseprite indexed_01.png palette.aseprite LICENSE
```
