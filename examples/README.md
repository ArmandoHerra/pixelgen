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

## Provider comparison (PXG-44)

The same three prompts at 16 and 32 px, each through `--backend llm-grid` (the spec LLM writes the grid) and `--backend openrouter` (`openai/gpt-image-2` makes a 1024 px image that pixelgen crops and pixelates). Every command is `pixelgen generate "<prompt>" --size <N> --palette pico8 --backend <arm> -o examples/<slug>/<slug>.png`, with the default models. Each slug directory holds the native PNG, its `@8x` preview, the v2 sidecar and a v2 `run.json` that records the usage, the cost lines, the worst case and the cap. This comparison does not rank the two arms; the notes below only describe what each preview shows.

### Approval lines

Recorded verbatim by the project manager in `decision.md` (OR-46, OR-47). Stage 1 ran after the first line and stage 2 after the second.

```
PXG-44 CALIBRATION APPROVED 2026-11-06: runs cmp-apple-16-llm-grid + cmp-apple-16-openrouter + probe, per-run cap llm-grid $1.05 / openrouter $1.00 / probe $0.25, stage cap $2.30, models claude-sonnet-5-5 + openai/gpt-image-2, prices confirmed 2026-11-06 (anthropic input $2/MTok output $10/MTok; openrouter image output $0.00003/token text input $0.000005/token), price per image $0.006, credit fee 8.5% read 2026-11-06, OpenRouter key limit $1.00
PXG-44 SPEND APPROVED 2026-11-06: remaining 10 runs, per-run cap llm-grid $1.05 / openrouter $0.30, total cap $2.00 (all PXG-44 spend, calibration included), models claude-sonnet-5-5 + openai/gpt-image-2, prices confirmed 2026-11-06 (anthropic input $2/MTok output $10/MTok; openrouter image output $0.00003/token text input $0.000005/token), price per image $0.006, credit fee 8.5% read 2026-11-06, measured calibration llm-grid $0.008786 (sidecar tokens) / openrouter $0.007789 (key usage, cross-check match)
```

### Prices and fee

- Anthropic `claude-sonnet-5-5`: input $2/MTok, output $10/MTok, confirmed 2026-11-06. The Anthropic cost of a run is its sidecar token counts times these prices.
- OpenRouter `openai/gpt-image-2` (low quality, 1024x1024): image output $0.00003/token and text input $0.000005/token, confirmed 2026-11-06; price per image $0.006.
- OpenRouter credit fee: 8.5%, read 2026-11-06 from the credit purchase screen ($0.85 on $10.00). The OpenRouter cost of a run is its key-usage delta times 1.085, cross-checked against price times calls.
- Stage 1 and stage 2 used two dedicated OpenRouter keys; each openrouter row's metered figure is the delta on the key that billed it.

### Runs

| Slug | Arm | Size | Approval | Anthropic in/out | OpenRouter metered | Cost (USD) | Worst (USD) | Cap (USD) | Cost source | Note |
|------|-----|------|----------|------------------|--------------------|------------|-------------|-----------|-------------|------|
| `cmp-apple-16-llm-grid` | llm-grid | 16 | calibration | 978 / 683 | - | 0.008786 | 1.03104 | 1.05 | sidecar-tokens-x-price |  |
| `cmp-apple-16-openrouter` | openrouter | 16 | calibration | 534 / 13 | 0.006075 | 0.007789 | 0.99465 | 1.0 | sidecar tokens + openrouter-key-usage | Both read as a red apple with a stem; llm-grid adds a green leaf, openrouter is shaded with a darker stem. Worst at OR-46 cap basis (f = 16%). |
| `cmp-apple-32-llm-grid` | llm-grid | 32 | spend | 977 / 4171 | - | 0.043664 | 1.03104 | 1.05 | sidecar-tokens-x-price |  |
| `cmp-apple-32-openrouter` | openrouter | 32 | spend | 534 / 14 | 0.006080 | 0.007805 | 0.27957 | 0.3 | sidecar tokens + openrouter-key-usage | Both show a red apple with a stem and a green leaf; llm-grid uses flat bands of colour, openrouter adds speckled shading. |
| `cmp-knight-16-llm-grid` | llm-grid | 16 | spend | 992 / 2000 | - | 0.021984 | 1.03104 | 1.05 | sidecar-tokens-x-price |  |
| `cmp-knight-16-openrouter` | openrouter | 16 | spend | 541 / 32 | 0.006100 | 0.008021 | 0.27957 | 0.3 | sidecar tokens + openrouter-key-usage | llm-grid shows a grey helmeted figure with a red cape; openrouter shows a darker, pointed-helmet figure with scattered colour. |
| `cmp-knight-32-llm-grid` | llm-grid | 32 | spend | 992 / 6194 | - | 0.063924 | 1.03104 | 1.05 | sidecar-tokens-x-price |  |
| `cmp-knight-32-openrouter` | openrouter | 32 | spend | 541 / 32 | 0.006100 | 0.008021 | 0.27957 | 0.3 | sidecar tokens + openrouter-key-usage | llm-grid shows a flat-coloured knight with a blue shield and a sword; openrouter shows a densely shaded armoured figure with a dark cape. |
| `cmp-slime-16-llm-grid` | llm-grid | 16 | spend | 980 / 1791 | - | 0.019870 | 1.03104 | 1.05 | sidecar-tokens-x-price |  |
| `cmp-slime-16-openrouter` | openrouter | 16 | spend | 535 / 15 | 0.006075 | 0.007811 | 0.27957 | 0.3 | sidecar tokens + openrouter-key-usage | llm-grid shows a green blob with two eyes; openrouter shows a grey-purple blob on a green base. |
| `cmp-slime-32-llm-grid` | llm-grid | 32 | spend | 980 / 5251 | - | 0.054470 | 1.03104 | 1.05 | sidecar-tokens-x-price |  |
| `cmp-slime-32-openrouter` | openrouter | 32 | spend | 535 / 15 | 0.006075 | 0.007811 | 0.27957 | 0.3 | sidecar tokens + openrouter-key-usage | llm-grid shows a green blob with a red mouth; openrouter shows a blue teardrop with a smile. |

### Failed runs

No run failed.

### Totals

Total spent: 0.259956 USD

Cost per image (sum of the arm's six runs / 6): llm-grid 0.035450 USD (0.212698 / 6), openrouter 0.007876 USD (0.047258 / 6).

The total covers the 12 slugs and no failed run. The direct probe below was billed under the same approval and is outside it; all PXG-44 spend including the probe is 0.266553 USD.

### PXG-42 open items

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Credit-purchase fee % | CONFIRMED | The operator's OpenRouter purchase screen: $0.85 service fee on $10.00, read 2026-11-06 (OR-46). The fee is charged when credits are bought, not on usage. Key-usage deltas equal list price exactly (run 2: 0.006075 = 196 × $0.00003 + 39 × $0.000005; probe: 0.00608), which is consistent with D3's `cost = m × (1 + f)`. It replaces research §5's UNVERIFIED and D5's 10% planning ceiling. Stage 2: five more deltas 0.00608–0.0061, each = list price. |
| 2 | Whether a non-streaming 502 is billed | NOT OBSERVED | No 502 in 12 runs (6 openrouter calls) or the probe; recorded NOT OBSERVED by operator ruling OR-48; the worst case still assumes a 502 is billed. |
| 3 | Request fields honoured (`size`, `quality`, `output_format`) | CONFIRMED | All three fields honoured. Probe, the same body as PXG-42: decoded PNG 1024×1024 (`size`). 196 image tokens, $0.00588 against about $0.211 for high, and the run-2 delta $0.006075 on PXG-42's own path (`quality: low`). `media_type` `image/png` and PNG bytes (`output_format`). Nothing rejected; `resolution`/`aspect_ratio` were not sent, as designed. |
| 4 | Response fields beyond the feature-page example | CORRECTED | Top level `created` (int), `data` (list, length 1), `usage`, as the example says. `data[0]` = `b64_json` (str, **raw base64, not a data URL**, 866,882 bytes decoded) + `media_type`. New in `usage`: `is_byok` (false), `prompt_tokens_details.cached_tokens` (0), `completion_tokens_details.{reasoning_tokens 0, image_tokens 196}`, `cost_details.{upstream_inference_cost 0.00608, upstream_inference_prompt_cost 0.0002, upstream_inference_completions_cost 0.00588}`. PXG-42 tolerates unknown keys (its AC5 tolerated rows), so this is informational. |
| 5 | `usage.cost` and token fields: presence and shape | CONFIRMED | `usage.cost` 0.00608 (float, USD, before the fee), equal to the token reproduction and to the key-usage delta. `prompt_tokens` 40, `completion_tokens` 196, `total_tokens` 236 (ints, 40 + 196 = 236). PXG-42 still never reads them (X-PXG-120). Any use of `usage.cost` as a per-call cost source is cycle-5 material, not a ruling here. |
| 6 | `media_type` values | CONFIRMED | `image/png` for `output_format: png`. Probe `data[0].media_type` = `image/png`. Only png was requested, so other formats were not exercised. PXG-42 lets the bytes decide, so no other value matters to it. |

### Probe record

One direct request, run after the stage-1 pair under the CALIBRATION line. Its body equals the request PXG-42 sends: `openai/gpt-image-2`, the `build_prompt` prompt, `1024x1024`, quality `low`, `png`, `n: 1`. Only field names, types and sizes were kept, and the reply file was deleted.

- Key-usage delta 0.00608 on the stage-1 key; cost round(0.00608 x 1.085, 6) = 0.006597 USD, against a probe cap of 0.25 USD.
- `usage.cost` 0.00608: 196 image tokens and 40 prompt tokens, 236 in total.
- Decoded PNG 1024x1024, `media_type` `image/png`, `b64_json` raw base64 (not a data URL).
