# Live sample generations (PXG-14)

Real `pixelgen` output from the default model `claude-sonnet-5-5`. Each slug directory holds the native PNG, the `@8x` preview, the sidecar and a `run.json` record with the exact command, prompt, token usage and cost. Re-check offline with `python scripts/check_examples.py examples --require-runs 3`.

## Approval, caps and prices

`PXG-14 SPEND APPROVED 2026-10-29: per-run cap $0.75, total cap $2.50, model claude-sonnet-5-5, prices confirmed 2026-10-29 (input $2.00/MTok, output $10.00/MTok)`

Recorded deviation (decision.md OR-10): the operator ran the four commands before relaying the confirmed prices, so the price confirmation came after the calls. The pre-agreed go-condition was input <= $4.00 and output <= $20.00 per MTok; the confirmed $2.00/$10.00 satisfies it, and every run is far within the caps.

Cost formula: `cost_usd = round(sum(usage x price) / 1e6, 6)`.

## Estimate vs actual

Estimate method: the D7 expected in/out tokens the operator was shown at the gate (decision.md section 9.9: 624 / 872 for red-apple, 624 / 2,408 for slime and knight, 1,792 / 2,258 for the refine; 3,664 / 7,946 in total), multiplied out at the confirmed prices.

| Run | Slug | Grid | Input tokens | Output tokens | Estimate (USD) | Actual (USD) | Per-run cap |
|-----|------|------|--------------|---------------|----------------|--------------|-------------|
| generate | red-apple | 16x16 pico8 | 977 | 674 | 0.009968 | 0.008694 | 0.75 |
| generate | slime | 32x32 gameboy | 1784 | 15559 | 0.025328 | 0.159158 | 0.75 |
| generate | knight | 32x32 pico8 | 992 | 7328 | 0.025328 | 0.075264 | 0.75 |
| refine | knight (knight-v2) | 32x32 pico8 | 3351 | 6285 | 0.026164 | 0.069552 | 0.75 |
| **Total** | | | 7104 | 29846 | **0.086788** | **0.312668** | total cap 2.50 |

Notes on the gap: slime needed 2 grid validation retries (4 calls), and the 32x32 outputs used far more output tokens than the 2 x W x H envelope. No run failed, so no cost is missing from the table.

## Refine

The refine ran live: `pixelgen refine "make the cape blue and add a sword" --from examples/knight/knight.json` wrote `knight-v2.png`, `knight-v2@8x.png` and `knight-v2.json` into `examples/knight/`, recorded as the second step of `knight/run.json`.
