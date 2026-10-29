#!/usr/bin/env python3
"""Offline checker for ``examples/`` (PXG-14): palette-only PNGs, exact previews, run records.

Usage: ``python scripts/check_examples.py [ROOT=examples] [--require-runs N]``.
Exit 0 ok/skip, 1 problems, 2 usage error. Never imports anthropic, pixelgen.llm or pixelgen.cli,
and never echoes matched key material.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
KEY_MARKER = b"sk-ant-"
KEY_TEXT = KEY_MARKER.decode("ascii")
KEY_PATTERN = re.compile(re.escape(KEY_TEXT) + r"\S*")
REDACTED = "[redacted]"
RECORD_VERSION = 1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_TEXT_BYTES = 4 * 1024 * 1024  # decompression bound for PNG text chunks


@dataclass(frozen=True)
class Problem:
    relpath: str
    reason: str


@dataclass
class Stats:
    runs: int = 0
    steps: int = 0
    images: int = 0
    key_hits: int = 0
    billed_runs: int = 0
    slugs: list[str] = field(default_factory=list)


def _redact(text: str) -> str:
    """Last line of defence: nothing printed may contain ``sk-ant-`` plus whatever follows it."""
    return KEY_PATTERN.sub(REDACTED, text)


def _scrub(text: str, generic: str) -> str:
    """Swap any text that carries key material for a generic reason."""
    return generic if KEY_TEXT in text else text


def _is_number(value: Any) -> bool:
    """A real, finite number (bool, NaN and +-Infinity are not)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
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
        if KEY_TEXT in f"{name}{exc}":
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


def _record_problems(step: Any, index: int, sidecar: Sidecar | None, rel: str) -> list[Problem]:
    label = f"step {index + 1}"

    def bad(reason: str) -> Problem:
        return Problem(rel, f"{label}: {reason}")

    if not isinstance(step, dict):
        return [bad("must be an object")]
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
    elif "ANTHROPIC_API_KEY" in command or "sk-ant-" in command:
        problems.append(bad("command contains key material or an API key variable"))
    if not isinstance(prompt, str) or not prompt.strip():
        problems.append(bad("prompt missing or empty"))
    elif kind == "generate" and isinstance(command, str) and prompt not in command:
        problems.append(bad("command does not contain the prompt verbatim"))

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
    expected = round(sum(usage[k] * prices[k] for k in usage) / 1e6, 6)
    if abs(cost - expected) > 1e-9:
        problems.append(bad(f"cost_usd {cost} != formula {expected}"))
    if cost > cap:
        problems.append(bad(f"cost_usd {cost} exceeds cap_usd {cap}"))
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
            plain = inflater.decompress(compressed, MAX_TEXT_BYTES)
        except zlib.error:
            failures.append(ctype.decode("ascii"))
            continue
        texts.append(keyword + plain)
    return texts, failures


def _key_material_problems(root: Path, stats: Stats) -> list[Problem]:
    problems: list[Problem] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            try:
                data = path.read_bytes()
            except OSError:
                problems.append(Problem(rel, "cannot read file"))
                continue
            hit = KEY_MARKER in data
            if data.startswith(PNG_SIGNATURE):
                texts, failures = _png_text_chunks(data)
                for kind in sorted(set(failures)):
                    problems.append(Problem(rel, f"{kind} chunk failed to decompress"))
                hit = hit or any(KEY_MARKER in t for t in texts)
            if hit:
                stats.key_hits += 1
                problems.append(Problem(rel, "key-like material"))
    return problems


# --------------------------------------------------------------------------- tree


def _load_record(path: Path, rel: str) -> tuple[dict[str, Any] | None, list[Problem]]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, [Problem(rel, "run.json missing or not valid JSON")]
    if not isinstance(record, dict):
        return None, [Problem(rel, "run.json must be a JSON object")]
    problems: list[Problem] = []
    if record.get("run_record_version") != RECORD_VERSION:
        problems.append(Problem(rel, f"run_record_version must be {RECORD_VERSION}"))
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
            problems.extend(_record_problems(step, index, sidecar, record_rel))
        first = record["steps"][0]
        usage = first.get("usage") if isinstance(first, dict) else None
        if (
            isinstance(first, dict)
            and first.get("kind") == "generate"
            and isinstance(usage, dict)
            and _is_number(usage.get("output_tokens"))
            and usage["output_tokens"] > 0
        ):
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
            print(_redact(f"FAIL {problem.relpath}: {problem.reason}"))
        print(_redact(f"FAIL {args.root}: {len(problems)} problem(s)"))
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
