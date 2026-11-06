#!/usr/bin/env python3
"""Offline checker for ``examples/`` (PXG-14): palette-only PNGs, exact previews, run records.

Usage: ``python scripts/check_examples.py [ROOT=examples] [--require-runs N]``.
Exit 0 ok/skip, 1 problems, 2 usage error. Never imports anthropic, pixelgen.llm, pixelgen.cli,
pixelgen.providers, pixelgen.backends or any network module, and never echoes matched key material.

Run records come in two versions. Version 1 (PXG-14) prices one ``usage`` dict. Version 2 (PXG-44)
prices each provider's sidecar ``usage`` records on a line of its own and, for a comparison pair,
checks that the two members share their prompt and flags.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pixelgen.export import Sidecar, load_sidecar
from pixelgen.palettes import load_palette

SKIP_LINE = "SKIP examples/ absent: live samples are operator-gated (PXG-14, OR-03)"
KEY_MARKERS = (b"sk-ant-", b"sk-or-")
KEY_TEXTS = tuple(m.decode("ascii") for m in KEY_MARKERS)
KEY_PATTERN = re.compile("(?:" + "|".join(re.escape(t) for t in KEY_TEXTS) + r")\S*")
BANNED_COMMAND_TEXTS = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", *KEY_TEXTS)
REDACTED = "[redacted]"
RECORD_VERSIONS = (1, 2)
VERSION_REASON = "run_record_version must be 1 or 2"
ARMS = ("llm-grid", "openrouter")
EPS = 1e-9
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_TEXT_BYTES = 4 * 1024 * 1024  # decompression bound for PNG text chunks


@dataclass(frozen=True)
class Problem:
    relpath: str
    reason: str


@dataclass(frozen=True)
class PairMember:
    """What the pair rules need from one v2 slug (collected while the slug is checked)."""

    slug: str
    pair: str
    backend: Any
    prompt: Any
    size: tuple[int, int] | None
    palette: str | None
    flags: tuple[str, ...] | None
    out: str | None
    backend_flag: str | None
    sidecar_backend: str | None


@dataclass
class Stats:
    runs: int = 0
    steps: int = 0
    images: int = 0
    key_hits: int = 0
    billed_runs: int = 0
    slugs: list[str] = field(default_factory=list)
    prices: list[tuple[str, float]] = field(default_factory=list)
    members: list[PairMember] = field(default_factory=list)


def _redact(text: str) -> str:
    """Last line of defence: nothing printed may contain ``sk-ant-`` plus whatever follows it."""
    return KEY_PATTERN.sub(REDACTED, text)


def _has_key(text: str) -> bool:
    return any(t in text for t in KEY_TEXTS)


def _has_key_bytes(data: bytes) -> bool:
    return any(m in data for m in KEY_MARKERS)


def _redact_path(rel: str) -> str:
    """Per path segment, replace key text through that segment's end (keeps the path usable)."""
    return "/".join(
        seg[: min(seg.index(t) for t in KEY_TEXTS if t in seg)] + REDACTED if _has_key(seg) else seg
        for seg in rel.split("/")
    )


def _scrub(text: str, generic: str) -> str:
    """Swap any text that carries key material for a generic reason."""
    return generic if _has_key(text) else text


def _is_number(value: Any) -> bool:
    """A real, finite number (bool, NaN and +-Infinity are not)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        try:
            return math.isfinite(float(value))
        except OverflowError:
            return False
    return isinstance(value, float) and math.isfinite(value)


def _hex(rgb: Any) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


# --------------------------------------------------------------------------- sidecar


def _sidecar_problems(sidecar_path: Path, rel: str) -> tuple[Sidecar | None, list[Problem]]:
    try:
        sidecar = load_sidecar(sidecar_path)
    except Exception as exc:  # load_sidecar raises a one-line SpecError for every failure
        text = " ".join(str(exc).replace(str(sidecar_path) + ":", "").split())
        text = _scrub(text, "content carries key-like material (not echoed)")
        return None, [Problem(rel, f"sidecar rejected by load_sidecar: {text}")]
    name = sidecar.palette.name
    try:
        shipped = [_hex(row)[1:] for row in load_palette(name)]
    except Exception as exc:
        if _has_key(f"{name}{exc}"):
            return sidecar, [Problem(rel, "unknown shipped palette (name not echoed)")]
        return sidecar, [Problem(rel, f"unknown shipped palette {name!r}: {exc}")]
    if list(sidecar.palette.colors) != shipped:
        return sidecar, [Problem(rel, f"sidecar palette.colors differ from shipped palette {name}")]
    return sidecar, []


# --------------------------------------------------------------------------- images


def _open_rgba(path: Path, rel: str) -> tuple[Image.Image | None, list[Problem]]:
    if not path.is_file():
        return None, [Problem(rel, "missing file")]
    try:
        with Image.open(path) as img:
            img.load()
            if img.mode != "RGBA":
                return None, [Problem(rel, f"mode {img.mode} != RGBA")]
            return img.copy(), []
    except (OSError, ValueError, SyntaxError) as exc:
        return None, [Problem(rel, f"cannot read image: {type(exc).__name__}")]


def _palette_problems(
    image: Image.Image, rel: str, colors: list[str], width: int, height: int
) -> list[Problem]:
    problems: list[Problem] = []
    if image.size != (width, height):
        w, h = image.size
        problems.append(Problem(rel, f"size {w}x{h} != spec {width}x{height}"))
    arr = np.asarray(image, dtype=np.uint8)
    alpha = arr[..., 3]
    partial = (alpha > 0) & (alpha < 255)
    if partial.any():
        problems.append(Problem(rel, "partial alpha"))
    opaque = arr[alpha == 255][:, :3].astype(np.int64)
    packed = (opaque[:, 0] << 16) | (opaque[:, 1] << 8) | opaque[:, 2]
    allowed = np.array([int(c, 16) for c in colors], dtype=np.int64)
    bad = np.unique(packed[~np.isin(packed, allowed)])
    if bad.size:
        listing = ", ".join(f"#{int(v):06x}" for v in bad)
        problems.append(Problem(rel, f"colour(s) not in palette: {listing}"))
    return problems


def _preview_problems(
    native: Image.Image, preview: Image.Image, rel: str, scale: int
) -> list[Problem]:
    w, h = native.size
    if preview.size != (w * scale, h * scale):
        pw, ph = preview.size
        return [Problem(rel, f"preview size {pw}x{ph} != native x{scale}")]
    expected = np.repeat(np.repeat(np.asarray(native, dtype=np.uint8), scale, 0), scale, 1)
    if not np.array_equal(np.asarray(preview, dtype=np.uint8), expected):
        return [Problem(rel, f"preview pixels differ from nearest-neighbour x{scale}")]
    return []


# --------------------------------------------------------------------------- run record


def _step_basics(step: dict[str, Any], index: int, bad: Any) -> list[Problem]:
    """Rules shared by both record versions: kind, command and prompt."""
    problems: list[Problem] = []
    kind = step.get("kind")
    if kind not in ("generate", "refine"):
        problems.append(bad("kind must be generate or refine"))
    elif index == 0 and kind != "generate":
        problems.append(bad("first step must be generate"))

    command = step.get("command")
    prompt = step.get("prompt")
    if not isinstance(command, str) or not command.startswith("pixelgen "):
        problems.append(bad("command must start with 'pixelgen '"))
    elif any(text in command for text in BANNED_COMMAND_TEXTS):
        problems.append(bad("command contains key material or an API key variable"))
    if not isinstance(prompt, str) or not prompt.strip():
        problems.append(bad("prompt missing or empty"))
    elif kind == "generate" and isinstance(command, str) and prompt not in command:
        problems.append(bad("command does not contain the prompt verbatim"))
    return problems


def _record_problems(step: Any, index: int, sidecar: Sidecar | None, rel: str) -> list[Problem]:
    label = f"step {index + 1}"

    def bad(reason: str) -> Problem:
        return Problem(rel, f"{label}: {reason}")

    if not isinstance(step, dict):
        return [bad("must be an object")]
    problems = _step_basics(step, index, bad)

    if sidecar is not None:
        if step.get("model") != sidecar.model:
            problems.append(bad("model differs from sidecar model"))
        usage = step.get("usage")
        want = {
            "input_tokens": sidecar.token_usage.input_tokens,
            "output_tokens": sidecar.token_usage.output_tokens,
        }
        got = (
            {k: usage.get(k) for k in ("input_tokens", "output_tokens")}
            if isinstance(usage, dict)
            else None
        )
        if got != want:
            problems.append(bad("usage differs from sidecar token_usage"))

    problems.extend(_cost_problems(step, bad))
    return problems


def _cost_problems(step: dict[str, Any], bad: Any) -> list[Problem]:
    usage = step.get("usage")
    prices = step.get("prices_usd_per_mtok")
    cost = step.get("cost_usd")
    cap = step.get("cap_usd")
    if not isinstance(usage, dict) or not all(_is_number(v) for v in usage.values()):
        return [bad("usage must map token kinds to numbers")]
    if not isinstance(prices, dict):
        return [bad("prices_usd_per_mtok must be an object")]
    missing = sorted(k for k in usage if not _is_number(prices.get(k)))
    if missing:
        return [bad(f"no price for usage key(s): {', '.join(missing)}")]
    if not _is_number(cost) or not _is_number(cap):
        return [bad("cost_usd and cap_usd must be numbers")]
    problems: list[Problem] = []
    raw = sum(float(usage[k]) * float(prices[k]) for k in usage)
    if not math.isfinite(raw):
        return [bad("usage x prices overflow the cost formula")]
    expected = round(raw / 1e6, 6)
    if abs(cost - expected) > 1e-9:
        problems.append(bad(f"cost_usd {cost} != formula {expected}"))
    if cost > cap:
        problems.append(bad(f"cost_usd {cost} exceeds cap_usd {cap}"))
    return problems


# --------------------------------------------------------------------------- run record v2


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= EPS


def _tokens_line_problems(line: dict[str, Any], sidecar: Sidecar, recs: list[Any], bad: Any):
    problems: list[Problem] = []
    tokens_in, tokens_out = line.get("input_tokens"), line.get("output_tokens")
    counted = (
        _is_number(tokens_in)
        and _is_number(tokens_out)
        and tokens_in == sidecar.token_usage.input_tokens
        and tokens_out == sidecar.token_usage.output_tokens
        and tokens_in + tokens_out == sum(r.quantity for r in recs)
    )
    if not counted:
        problems.append(bad("tokens line differs from sidecar token counts"))
    prices = line.get("prices_usd_per_mtok")
    price_in = prices.get("input_tokens") if isinstance(prices, dict) else None
    price_out = prices.get("output_tokens") if isinstance(prices, dict) else None
    if not (_is_number(price_in) and _is_number(price_out)):
        problems.append(bad("tokens line needs input_tokens and output_tokens prices"))
    elif _is_number(tokens_in) and _is_number(tokens_out) and _is_number(line.get("cost_usd")):
        raw = (float(tokens_in) * float(price_in) + float(tokens_out) * float(price_out)) / 1e6
        if not math.isfinite(raw) or not _close(line["cost_usd"], round(raw, 6)):
            problems.append(bad("tokens line cost_usd off formula"))
    if line.get("cost_source") != "sidecar-tokens-x-price":
        problems.append(bad("tokens line cost_source must be sidecar-tokens-x-price"))
    return problems


def _images_line_problems(
    line: dict[str, Any], recs: list[Any], bad: Any, slug_prices: list[float]
) -> list[Problem]:
    problems: list[Problem] = []
    n = sum(r.calls for r in recs)
    before = line.get("key_usage_before_usd")
    after = line.get("key_usage_after_usd")
    price = line.get("price_per_image_usd")
    fee = line.get("credit_fee_pct")
    if not all(_is_number(v) for v in (before, after, price, fee)):
        return [bad("images line needs numeric key usage readings, price and fee")]
    slug_prices.append(float(price))
    measured = round(after - before, 6)
    expected = round(n * price, 6)
    if not (_is_number(line.get("calls")) and line["calls"] == n):
        problems.append(bad("images line calls differ from usage records"))
    if after < before:
        problems.append(bad("key_usage_after_usd below key_usage_before_usd"))
    if not (_is_number(line.get("metered_usd")) and _close(line["metered_usd"], measured)):
        problems.append(bad("metered_usd differs from the key usage delta"))
    if not (_is_number(line.get("expected_usd")) and _close(line["expected_usd"], expected)):
        problems.append(bad("expected_usd differs from calls x price_per_image_usd"))
    word = "match" if abs(measured - expected) <= 0.5 * price else "mismatch"
    if line.get("cross_check") != word:
        problems.append(bad("cross_check differs from the recomputation"))
    if _is_number(line.get("cost_usd")):
        want = round(measured * (1 + fee / 100), 6)
        if not _close(line["cost_usd"], want):
            problems.append(bad("images line cost_usd off fee formula"))
    if line.get("cost_source") != "openrouter-key-usage":
        problems.append(bad("images line cost_source must be openrouter-key-usage"))
    return problems


def _line_problems(
    lines: list[dict[str, Any]], sidecar: Sidecar, usage: list[Any], bad: Any, slug_prices: list
) -> list[Problem]:
    """Rules (5)-(7): one cost line per provider, each priced from its own usage records."""
    problems: list[Problem] = []
    providers = list(dict.fromkeys(r.provider for r in usage))
    token_providers = {r.provider for r in usage if r.unit == "tokens"}
    if len(token_providers) > 1:
        problems.append(bad("token records from more than one provider cannot be split"))
    if [line.get("provider") for line in lines] != providers:
        problems.append(bad("one cost line per provider"))
        return problems
    for line, provider in zip(lines, providers, strict=True):
        recs = [r for r in usage if r.provider == provider]
        units = list(dict.fromkeys(r.unit for r in recs))
        if line.get("steps") != [r.step for r in recs] or [line.get("unit")] != units:
            problems.append(bad("cost line steps differ from usage records"))
        if len(units) != 1:
            continue
        if units[0] == "tokens":
            if len(token_providers) == 1:
                problems.extend(_tokens_line_problems(line, sidecar, recs, bad))
        elif units[0] == "images":
            problems.extend(_images_line_problems(line, recs, bad, slug_prices))
        else:
            problems.append(bad(f"no pricing rule for unit {units[0]}"))
    return problems


def _record_problems_v2(
    step: Any, index: int, sidecar: Sidecar | None, rel: str, slug_prices: list[float]
) -> list[Problem]:
    label = f"step {index + 1}"

    def bad(reason: str) -> Problem:
        return Problem(rel, f"{label}: {reason}")

    if not isinstance(step, dict):
        return [bad("must be an object")]
    problems = _step_basics(step, index, bad)
    if "usage" in step:
        problems.append(bad("v2 step must not carry a whole-run usage figure"))
    if step.get("approval") not in ("calibration", "spend"):
        problems.append(bad("approval must be calibration or spend"))

    raw_lines = step.get("costs")
    lines_ok = (
        isinstance(raw_lines, list)
        and bool(raw_lines)
        and all(isinstance(x, dict) for x in raw_lines)
    )
    if not lines_ok:
        problems.append(bad("costs must be a non-empty list of objects"))

    if sidecar is not None:
        if sidecar.schema_version != 2:
            problems.append(bad("run record v2 needs a v2 sidecar"))
            sidecar = None
    if sidecar is not None:
        for key, have in (
            ("backend", sidecar.backend),
            ("model", sidecar.model),
            ("provider", sidecar.provider),
            ("provider_model", sidecar.provider_model),
        ):
            if step.get(key) != have:
                problems.append(bad(f"{key} differs from sidecar"))
        usage = list(sidecar.usage or [])
        if step.get("usage_records") != [r.model_dump(mode="json") for r in usage]:
            problems.append(bad("usage_records differ from sidecar usage"))
        if lines_ok:
            problems.extend(_line_problems(raw_lines, sidecar, usage, bad, slug_prices))

    cost, worst, cap = step.get("cost_usd"), step.get("worst_usd"), step.get("cap_usd")
    if not all(_is_number(v) for v in (cost, worst, cap)):
        problems.append(bad("cost_usd, worst_usd and cap_usd must be numbers"))
        return problems
    if lines_ok:
        if not all(_is_number(x.get("cost_usd")) for x in raw_lines):
            problems.append(bad("cost line cost_usd must be a number"))
        elif not _close(cost, round(math.fsum(x["cost_usd"] for x in raw_lines), 6)):
            problems.append(bad("step cost_usd differs from the sum of its lines"))
    if cost > cap:
        problems.append(bad("cost_usd above cap_usd"))
    if worst > cap:
        problems.append(bad("cap_usd below worst_usd"))
    return problems


def _v2_billed(step: Any) -> bool:
    """A v2 generate step is billed when every tokens line has output tokens and every images
    line was metered."""
    lines = step.get("costs") if isinstance(step, dict) else None
    if not isinstance(lines, list) or not lines or not all(isinstance(x, dict) for x in lines):
        return False
    for line in lines:
        if line.get("unit") == "tokens":
            out = line.get("output_tokens")
            if not (_is_number(out) and out > 0):
                return False
        elif line.get("unit") == "images":
            metered = line.get("metered_usd")
            if not (_is_number(metered) and metered > 0):
                return False
        else:
            return False
    return True


# --------------------------------------------------------------------------- pairs

_OPTION_FLAGS = ("-o", "--backend", "--size", "--palette")


def _command_parts(command: Any, prompt: Any) -> tuple[tuple[str, ...] | None, dict[str, str]]:
    """The command's flags (prompt and the -o/--backend/--size/--palette pairs removed) and the
    values of those four options."""
    if not isinstance(command, str):
        return None, {}
    try:
        argv = shlex.split(command)
    except ValueError:
        return None, {}
    flags: list[str] = []
    values: dict[str, str] = {}
    dropped_prompt = False
    pos = 0
    while pos < len(argv):
        token = argv[pos]
        pos += 1
        if token in _OPTION_FLAGS:
            if pos < len(argv):
                values.setdefault(token, argv[pos])
                pos += 1
            continue
        name, eq, value = token.partition("=")
        if eq and name in _OPTION_FLAGS and name != "-o":
            values.setdefault(name, value)
            continue
        if not dropped_prompt and token == prompt:
            dropped_prompt = True
            continue
        flags.append(token)
    return tuple(flags), values


def _member_of(
    slug: str, record: dict[str, Any], step: dict[str, Any], sidecar: Sidecar | None
) -> PairMember:
    flags, values = _command_parts(step.get("command"), step.get("prompt"))
    return PairMember(
        slug=slug,
        pair=record["pair"],
        backend=step.get("backend"),
        prompt=step.get("prompt"),
        size=(sidecar.spec.width, sidecar.spec.height) if sidecar is not None else None,
        palette=sidecar.palette.name if sidecar is not None else None,
        flags=flags,
        out=values.get("-o"),
        backend_flag=values.get("--backend"),
        sidecar_backend=sidecar.backend if sidecar is not None else None,
    )


def _pair_problems(members: list[PairMember]) -> list[Problem]:
    problems: list[Problem] = []
    groups: dict[str, list[PairMember]] = {}
    for member in members:
        groups.setdefault(member.pair, []).append(member)
    for pair, group in sorted(groups.items()):
        group.sort(key=lambda m: m.slug)
        for member in group:
            rel = f"{member.slug}/run.json"
            if member.out != f"examples/{member.slug}/{member.slug}.png":
                problems.append(
                    Problem(
                        rel, f"{member.slug} -o is not examples/{member.slug}/{member.slug}.png"
                    )
                )
            if member.sidecar_backend is not None and member.backend_flag != member.sidecar_backend:
                problems.append(
                    Problem(rel, f"{member.slug} --backend differs from sidecar backend")
                )
        wanted = {f"{pair}-{arm}": arm for arm in ARMS}
        complete = (
            len(group) == 2
            and {m.slug for m in group} == set(wanted)
            and all(wanted[m.slug] == m.backend for m in group)
        )
        if not complete:
            problems.append(
                Problem(
                    f"{group[-1].slug}/run.json",
                    f"pair {pair} needs exactly one llm-grid and one openrouter slug",
                )
            )
            continue
        first, second = (next(m for m in group if m.slug == name) for name in wanted)
        differing = []
        if first.prompt != second.prompt:
            differing.append("prompt")
        if first.size != second.size and None not in (first.size, second.size):
            differing.append("size")
        if first.palette != second.palette and None not in (first.palette, second.palette):
            differing.append("palette")
        if first.flags != second.flags:
            differing.append("flags")
        if differing:
            problems.append(
                Problem(
                    f"{second.slug}/run.json",
                    f"pair {first.slug}/{second.slug} differs in {','.join(differing)}",
                )
            )
    return problems


# --------------------------------------------------------------------------- key material


def _png_text_chunks(data: bytes) -> tuple[list[bytes], list[str]]:
    """Return decoded tEXt/zTXt/iTXt payloads (keyword + text) and decode-failure labels."""
    texts: list[bytes] = []
    failures: list[str] = []
    if not data.startswith(PNG_SIGNATURE):
        return texts, failures
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype not in (b"tEXt", b"zTXt", b"iTXt"):
            continue
        keyword, _, rest = body.partition(b"\x00")
        compressed: bytes | None = None
        if ctype == b"tEXt":
            texts.append(keyword + rest)
            continue
        if ctype == b"zTXt":
            compressed = rest[1:]
        else:  # iTXt: flag, method, language\0, translated keyword\0, text
            if len(rest) < 2:
                continue
            flag, remainder = rest[0], rest[2:]
            _, _, remainder = remainder.partition(b"\x00")
            _, _, remainder = remainder.partition(b"\x00")
            if flag == 0:
                texts.append(keyword + remainder)
                continue
            compressed = remainder
        try:
            inflater = zlib.decompressobj()
            plain = inflater.decompress(compressed, MAX_TEXT_BYTES + 1)
        except zlib.error:
            failures.append(f"{ctype.decode('ascii')} chunk failed to decompress")
            continue
        if len(plain) > MAX_TEXT_BYTES:
            failures.append(f"{ctype.decode('ascii')} chunk exceeds the decompression bound")
            plain = plain[:MAX_TEXT_BYTES]
        texts.append(keyword + plain)
    return texts, failures


def _json_strings(obj: Any) -> Any:
    """Yield every dict key, dict value and list item that is a string (explicit stack)."""
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for key, value in item.items():
                stack.append(key)
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)


def _decoded_json_hit(data: bytes) -> bool:
    try:
        obj = json.loads(data)
    except (ValueError, RecursionError):
        return False
    return any(_has_key(s) for s in _json_strings(obj))


def _key_material_problems(root: Path, stats: Stats) -> list[Problem]:
    problems: list[Problem] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for dname in dirnames:
            if _has_key(dname):
                drel = (Path(dirpath) / dname).relative_to(root).as_posix() + "/"
                stats.key_hits += 1
                problems.append(Problem(drel, "key-like material in name"))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if _has_key(name):
                stats.key_hits += 1
                problems.append(Problem(rel, "key-like material in name"))
            try:
                data = path.read_bytes()
            except OSError:
                problems.append(Problem(rel, "cannot read file"))
                continue
            hit = _has_key_bytes(data)
            if data.startswith(PNG_SIGNATURE):
                texts, failures = _png_text_chunks(data)
                for reason in sorted(set(failures)):
                    problems.append(Problem(rel, reason))
                hit = hit or any(_has_key_bytes(t) for t in texts)
            else:
                hit = hit or _decoded_json_hit(data)
            if hit:
                stats.key_hits += 1
                problems.append(Problem(rel, "key-like material"))
    return problems


# --------------------------------------------------------------------------- tree


def _load_record(path: Path, rel: str) -> tuple[dict[str, Any] | None, list[Problem]]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except RecursionError:
        return None, [Problem(rel, "run.json nests too deeply to parse")]
    except (OSError, ValueError):
        return None, [Problem(rel, "run.json missing or not valid JSON")]
    if not isinstance(record, dict):
        return None, [Problem(rel, "run.json must be a JSON object")]
    problems: list[Problem] = []
    if record.get("run_record_version") not in RECORD_VERSIONS:
        problems.append(Problem(rel, VERSION_REASON))
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        problems.append(Problem(rel, "steps must be a non-empty list"))
        return None, problems
    return record, problems


def _bare_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value not in ("", ".", "..")
        and "/" not in value
        and "\\" not in value
    )


def _check_slug(slug_dir: Path, root: Path, stats: Stats) -> list[Problem]:
    slug = slug_dir.name
    rel_of = lambda name: f"{slug}/{name}"  # noqa: E731
    record_rel = rel_of("run.json")
    record, problems = _load_record(slug_dir / "run.json", record_rel)
    expected = {"run.json"}
    names_ok = True
    load_failed = False
    if record is not None:
        version = record.get("run_record_version")
        slug_prices: list[float] = []
        if record.get("slug") != slug:
            problems.append(Problem(record_rel, "slug differs from directory name"))
        for index, step in enumerate(record["steps"]):
            stats.steps += 1
            sidecar: Sidecar | None = None
            name = step.get("sidecar") if isinstance(step, dict) else None
            if not _bare_name(name):
                problems.append(
                    Problem(record_rel, f"step {index + 1}: sidecar must be a basename")
                )
                names_ok = False
            else:
                expected.add(name)
                sidecar, found = _sidecar_problems(slug_dir / name, rel_of(name))
                problems.extend(found)
                if sidecar is None:
                    load_failed = True
                else:
                    expected.update((sidecar.outputs.png, sidecar.outputs.preview))
                    problems.extend(_image_problems(slug_dir, slug, sidecar, stats))
            if version == 1:
                problems.extend(_record_problems(step, index, sidecar, record_rel))
            elif version == 2:
                problems.extend(_record_problems_v2(step, index, sidecar, record_rel, slug_prices))
            if (
                version != 1
                and index == 0
                and isinstance(step, dict)
                and isinstance(record.get("pair"), str)
            ):
                # an unknown version already has its own problem; its pair stays complete
                stats.members.append(_member_of(slug, record, step, sidecar))
        first = record["steps"][0]
        if version == 2 and slug_prices:
            stats.prices.append((slug, slug_prices[0]))
        if version == 1:
            usage = first.get("usage") if isinstance(first, dict) else None
            billed = (
                isinstance(first, dict)
                and first.get("kind") == "generate"
                and isinstance(usage, dict)
                and _is_number(usage.get("output_tokens"))
                and usage["output_tokens"] > 0
            )
        else:
            billed = (
                version == 2
                and isinstance(first, dict)
                and first.get("kind") == "generate"
                and _v2_billed(first)
            )
        if billed:
            stats.billed_runs += 1
    if record is not None and names_ok:
        # A sidecar that failed to load hides its outputs; derive the likely names from the slug.
        derived = re.compile(re.escape(slug) + r"(@\d+x)?\.png")
        for entry in sorted(os.listdir(slug_dir)):
            if entry in expected or (load_failed and derived.fullmatch(entry)):
                continue
            problems.append(Problem(rel_of(entry), "unexpected file in slug dir"))
    return problems


def _image_problems(slug_dir: Path, slug: str, sidecar: Sidecar, stats: Stats) -> list[Problem]:
    problems: list[Problem] = []
    png_name, preview_name = sidecar.outputs.png, sidecar.outputs.preview
    native, found = _open_rgba(slug_dir / png_name, f"{slug}/{png_name}")
    problems.extend(found)
    stats.images += 1
    if native is not None:
        problems.extend(
            _palette_problems(
                native,
                f"{slug}/{png_name}",
                list(sidecar.palette.colors),
                sidecar.spec.width,
                sidecar.spec.height,
            )
        )
    preview, found = _open_rgba(slug_dir / preview_name, f"{slug}/{preview_name}")
    problems.extend(found)
    stats.images += 1
    if native is not None and preview is not None:
        problems.extend(
            _preview_problems(native, preview, f"{slug}/{preview_name}", sidecar.outputs.scale)
        )
    return problems


def check_tree(root: Path, require_runs: int = 0, stats: Stats | None = None) -> list[Problem]:
    """Check every ``<root>/<slug>/`` directory; return all problems (empty = ok)."""
    stats = stats if stats is not None else Stats()
    root = Path(root)
    if not root.exists():
        if require_runs > 0:
            return [Problem(".", f"{root.name}/ absent but --require-runs {require_runs} given")]
        return []
    problems: list[Problem] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            stats.runs += 1
            stats.slugs.append(entry.name)
            problems.extend(_check_slug(entry, root, stats))
        elif entry.name != "README.md":
            problems.append(Problem(entry.name, "unexpected file at examples root"))
    problems.extend(_pair_problems(stats.members))
    for slug, price in stats.prices[1:]:
        if not _close(price, stats.prices[0][1]):
            problems.append(
                Problem(f"{slug}/run.json", "price_per_image_usd differs across the tree")
            )
    problems.extend(_key_material_problems(root, stats))
    if require_runs > 0:
        if stats.billed_runs < require_runs:
            problems.append(
                Problem(
                    ".",
                    f"{stats.billed_runs} run(s) with a billed first generate step, "
                    f"--require-runs wants {require_runs}",
                )
            )
        readme = root / "README.md"
        if not readme.is_file():
            problems.append(Problem("README.md", "missing"))
        else:
            text = readme.read_text(encoding="utf-8", errors="replace")
            for slug in stats.slugs:
                # whole-token match: a hyphen/word char on either side means a different name
                if not re.search(rf"(?<![\w-]){re.escape(slug)}(?![\w-])", text):
                    problems.append(Problem("README.md", f"does not name slug {slug}"))
    return problems


def _count(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check examples/ against the PXG-14 contract.")
    parser.add_argument("root", nargs="?", default="examples", help="examples root")
    parser.add_argument("--require-runs", type=_count, default=0, metavar="N")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if root.exists() and not root.is_dir():
        print(_redact(f"usage error: {args.root} is not a directory"), file=sys.stderr)
        return 2
    if args.require_runs < 0:
        print("usage error: --require-runs must be >= 0", file=sys.stderr)
        return 2
    if not root.exists() and args.require_runs == 0:
        print(SKIP_LINE)
        return 0
    stats = Stats()
    problems = check_tree(root, args.require_runs, stats)
    if problems:
        for problem in problems:
            print(_redact(f"FAIL {_redact_path(problem.relpath)}: {problem.reason}"))
        print(_redact(f"FAIL {_redact_path(args.root)}: {len(problems)} problem(s)"))
        return 1
    print(
        _redact(
            f"OK {args.root}: {stats.runs} runs, {stats.steps} steps, {stats.images} images, "
            f"key-material hits {stats.key_hits}"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
