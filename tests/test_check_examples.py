"""PXG-14 Part A: scripts/check_examples.py against synthetic fixtures (no network, no API)."""

import ast
import copy
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from pixelgen.export import TokenUsage, export
from pixelgen.palettes import load_palette
from pixelgen.spec import ArtSpec

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_examples.py"
SKIP_LINE = "SKIP examples/ absent: live samples are operator-gated (PXG-14, OR-03)"
SLUG = "red-apple"
PROMPT = "a red apple"
SIZE = 16
SCALE = 8

RECORD = {
    "run_record_version": 1,
    "slug": SLUG,
    "steps": [
        {
            "kind": "generate",
            "command": (
                f'pixelgen generate "{PROMPT}" --size 16 --palette pico8 '
                f"-o examples/{SLUG}/{SLUG}.png"
            ),
            "prompt": PROMPT,
            "model": "claude-sonnet-5-5",
            "pixelgen_version": "0.1.0",
            "ran_at": "2026-11-02T10:00:00Z",
            "sidecar": f"{SLUG}.json",
            "usage": {"input_tokens": 3000, "output_tokens": 900},
            "prices_usd_per_mtok": {"input_tokens": 3.0, "output_tokens": 15.0},
            "prices_confirmed_on": "2026-11-02",
            "estimate_usd": 0.0225,
            "cost_usd": 0.0225,
            "cap_usd": 1.0,
            "notes": "",
        }
    ],
}


def run_checker(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )


def build_grid(colors: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """Checkerboard-ish grid: adjacent cells differ, so a blended preview cannot pass."""
    grid = [[1 + (x + y) % 4 for x in range(SIZE)] for y in range(SIZE)]
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    for y, row in enumerate(grid):
        for x, idx in enumerate(row):
            rgba[y, x, :3] = colors[idx]
            rgba[y, x, 3] = 255
    return grid, rgba


def make_example(
    root: Path,
    *,
    mutate=None,
    palette_colors: np.ndarray | None = None,
    use_index: int | None = None,
    record: dict | None = None,
) -> Path:
    """Write ``root/red-apple/`` through the real export(); return the slug dir."""
    colors = load_palette("pico8") if palette_colors is None else palette_colors
    grid, rgba = build_grid(colors)
    if use_index is not None:
        grid[0][0] = use_index
        rgba[0, 0, :3] = colors[use_index]
    if mutate is not None:
        mutate(rgba)
    slug_dir = root / SLUG
    export(
        slug_dir / f"{SLUG}.png",
        rgba,
        spec=ArtSpec(subject=PROMPT, width=SIZE, height=SIZE, palette="pico8"),
        grid=grid,
        palette_colors=colors,
        model="claude-sonnet-5-5",
        backend="llm-grid",
        token_usage=TokenUsage(input_tokens=3000, output_tokens=900, calls=2),
        scale=SCALE,
    )
    (slug_dir / "run.json").write_text(json.dumps(record or RECORD, indent=2), encoding="utf-8")
    return slug_dir


def preview_of(native: np.ndarray, scale: int = SCALE) -> np.ndarray:
    return np.repeat(np.repeat(native, scale, 0), scale, 1)


def save_png(path: Path, array: np.ndarray, **kw) -> None:
    Image.fromarray(np.ascontiguousarray(array)).save(path, format="PNG", **kw)


def assert_single_fail(proc, root: Path, relpath: str, reason: str) -> None:
    assert proc.returncode == 1, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    details = [ln for ln in lines if ln.startswith("FAIL ")][:-1]
    assert len(details) == 1, proc.stdout
    assert details[0].startswith(f"FAIL {relpath}: "), details[0]
    assert reason in details[0], details[0]
    assert lines[-1] == f"FAIL {root}: 1 problem(s)", proc.stdout


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "examples"


def test_valid_fixture_passes(root):
    make_example(root)
    proc = run_checker(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith(f"OK {root}: 1 runs, 1 steps, 2 images, key-material hits 0")


def test_absent_root_skips_unless_required(root):
    proc = run_checker(root)
    assert proc.returncode == 0
    assert proc.stdout.strip() == SKIP_LINE
    required = run_checker(root, "--require-runs", 3)
    assert required.returncode == 1, required.stdout


def test_palette_only_pairs(root, tmp_path):
    make_example(root)
    ok = run_checker(root)
    assert ok.returncode == 0 and ok.stdout.startswith("OK "), ok.stdout

    gameboy = load_palette("gameboy")[0]
    assert not (load_palette("pico8") == gameboy).all(axis=1).any()

    def off_palette(rgba):
        rgba[3, 3, :3] = gameboy

    def half_alpha(rgba):
        rgba[3, 3, 3] = 128

    def clear_magenta(rgba):
        rgba[3, 3] = [255, 0, 255, 0]

    base = tmp_path / "off"
    make_example(base, mutate=off_palette)
    hexcode = "#{:02x}{:02x}{:02x}".format(*gameboy)
    assert_single_fail(
        run_checker(base), base, f"{SLUG}/{SLUG}.png", f"colour(s) not in palette: {hexcode}"
    )

    base = tmp_path / "half"
    make_example(base, mutate=half_alpha)
    assert_single_fail(run_checker(base), base, f"{SLUG}/{SLUG}.png", "partial alpha")

    base = tmp_path / "clear"
    make_example(base, mutate=clear_magenta)
    proc = run_checker(base)
    assert proc.returncode == 0, proc.stdout


def test_sidecar_palette_must_match_shipped(root):
    extra = np.vstack([load_palette("pico8"), np.array([[0x12, 0x34, 0x56]], dtype=np.uint8)])
    make_example(root, palette_colors=extra, use_index=16)
    assert_single_fail(
        run_checker(root), root, f"{SLUG}/{SLUG}.json", "sidecar palette.colors differ"
    )


def test_size_and_preview_exact(tmp_path):
    # 16x15 native against a 16x16 spec; the preview is rewritten from the cropped native.
    crop = tmp_path / "crop"
    slug_dir = make_example(crop)
    native = np.asarray(Image.open(slug_dir / f"{SLUG}.png"))[:15]
    save_png(slug_dir / f"{SLUG}.png", native)
    save_png(slug_dir / f"{SLUG}@{SCALE}x.png", preview_of(native))
    assert_single_fail(run_checker(crop), crop, f"{SLUG}/{SLUG}.png", "size 16x15 != spec 16x16")

    blur = tmp_path / "blur"
    slug_dir = make_example(blur)
    size = SIZE * SCALE
    with Image.open(slug_dir / f"{SLUG}.png") as img:
        img.resize((size, size), Image.BILINEAR).save(slug_dir / f"{SLUG}@{SCALE}x.png")
    assert_single_fail(
        run_checker(blur), blur, f"{SLUG}/{SLUG}@{SCALE}x.png", "preview pixels differ"
    )

    seven = tmp_path / "seven"
    slug_dir = make_example(seven)
    native = np.asarray(Image.open(slug_dir / f"{SLUG}.png"))
    save_png(slug_dir / f"{SLUG}@{SCALE}x.png", preview_of(native, 7))
    assert_single_fail(run_checker(seven), seven, f"{SLUG}/{SLUG}@{SCALE}x.png", "preview size")


def _cost_over_cap(rec):
    rec["steps"][0]["cap_usd"] = 0.02


def _usage_mismatch(rec):
    rec["steps"][0]["usage"] = {"input_tokens": 3100, "output_tokens": 900}
    rec["steps"][0]["cost_usd"] = 0.0228


RECORD_CASES = {
    "missing_prompt": (lambda r: r["steps"][0].pop("prompt"), "prompt missing or empty"),
    "model_mismatch": (
        lambda r: r["steps"][0].update(model="other-model"),
        "model differs from sidecar model",
    ),
    "usage_mismatch": (_usage_mismatch, "usage differs from sidecar token_usage"),
    "cost_off_formula": (lambda r: r["steps"][0].update(cost_usd=0.022502), "!= formula"),
    "cost_over_cap": (_cost_over_cap, "exceeds cap_usd"),
    "nan_cost": (lambda r: r["steps"][0].update(cost_usd=float("nan")), "must be numbers"),
    "nan_cap": (lambda r: r["steps"][0].update(cap_usd=float("nan")), "must be numbers"),
    "inf_cap": (lambda r: r["steps"][0].update(cap_usd=float("inf")), "must be numbers"),
    "nan_price": (
        lambda r: r["steps"][0]["prices_usd_per_mtok"].update(input_tokens=float("nan")),
        "no price for usage key(s)",
    ),
    "key_in_command": (
        lambda r: r["steps"][0].update(
            command=f'pixelgen ANTHROPIC_API_KEY= generate "{PROMPT}" --size 16'
        ),
        "key material or an API key variable",
    ),
}


@pytest.mark.parametrize("case", [*RECORD_CASES, "stray_file"])
def test_run_record_rules(case, root):
    if case == "stray_file":
        slug_dir = make_example(root)
        (slug_dir / "debug.log").write_text("noise\n", encoding="utf-8")
        assert_single_fail(
            run_checker(root), root, f"{SLUG}/debug.log", "unexpected file in slug dir"
        )
        return
    edit, reason = RECORD_CASES[case]
    record = copy.deepcopy(RECORD)
    edit(record)
    make_example(root, record=record)
    assert_single_fail(run_checker(root), root, f"{SLUG}/run.json", reason)


def _inject_chunk(path: Path, ctype: bytes, body: bytes) -> None:
    data = path.read_bytes()
    end_ihdr = 8 + 12 + 13
    crc = struct.pack(">I", zlib.crc32(ctype + body))
    chunk = struct.pack(">I", len(body)) + ctype + body + crc
    path.write_bytes(data[:end_ihdr] + chunk + data[end_ihdr:])


@pytest.mark.parametrize("where", ["notes", "text", "ztxt", "itxt_zip"])
def test_key_material_detected_not_echoed(where, root):
    secret = "sk-ant-FAKEFAKE"
    record = copy.deepcopy(RECORD)
    if where == "notes":
        record["steps"][0]["notes"] = secret
    slug_dir = make_example(root, record=record)
    target = f"{SLUG}/run.json" if where == "notes" else f"{SLUG}/{SLUG}.png"
    png = slug_dir / f"{SLUG}.png"
    if where != "notes":
        info = PngInfo()
        if where == "text":
            info.add_text("Comment", secret)
        elif where == "ztxt":
            info.add_text("Comment", secret, zip=True)
        else:
            info.add_itxt("Comment", secret, zip=True)
        native = np.asarray(Image.open(png))
        save_png(png, native, pnginfo=info)
        assert b"sk-ant-" in png.read_bytes() or where != "text"
        if where != "text":
            assert b"sk-ant-" not in png.read_bytes()
    proc = run_checker(root)
    assert_single_fail(proc, root, target, "key-like material")
    assert "FAKEFAKE" not in proc.stdout + proc.stderr


def test_corrupt_ztxt_chunk_is_a_problem(root):
    slug_dir = make_example(root)
    _inject_chunk(slug_dir / f"{SLUG}.png", b"zTXt", b"Comment\x00\x00not-zlib-data")
    assert_single_fail(
        run_checker(root), root, f"{SLUG}/{SLUG}.png", "zTXt chunk failed to decompress"
    )


def test_sidecar_loads_with_refine_reader(root):
    slug_dir = make_example(root)
    path = slug_dir / f"{SLUG}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    assert_single_fail(run_checker(root), root, f"{SLUG}/{SLUG}.json", "sidecar rejected")


def test_require_runs_counts_billed_runs_and_readme(root):
    make_example(root)
    short = run_checker(root, "--require-runs", 2)
    assert short.returncode == 1
    assert "--require-runs wants 2" in short.stdout
    assert "FAIL README.md: missing" in short.stdout
    (root / "README.md").write_text(f"# Examples\n\n- {SLUG}\n", encoding="utf-8")
    ok = run_checker(root, "--require-runs", 1)
    assert ok.returncode == 0, ok.stdout
    (root / "README.md").write_text("# Examples\n", encoding="utf-8")
    nameless = run_checker(root, "--require-runs", 1)
    assert nameless.returncode == 1
    assert f"FAIL README.md: does not name slug {SLUG}" in nameless.stdout


def test_root_that_is_a_file_exits_2(tmp_path):
    scratch = tmp_path / "not-a-dir"
    scratch.write_text("x", encoding="utf-8")
    proc = run_checker(scratch)
    assert proc.returncode == 2
    assert "usage error" in proc.stderr


def test_committed_examples_tree():
    examples = REPO / "examples"
    if examples.exists():
        proc = run_checker(examples, "--require-runs", 3)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return
    proc = run_checker(examples)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = proc.stdout.strip()
    assert line == SKIP_LINE
    pytest.skip(line)


def test_checker_imports_no_llm():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    banned = ("anthropic", "pixelgen.llm", "pixelgen.cli")
    assert not [m for m in imported if m.split(".")[0] == "anthropic" or m in banned], imported


def test_nan_usage_value_is_rejected(root):
    record = copy.deepcopy(RECORD)
    record["steps"][0]["usage"]["input_tokens"] = float("nan")
    make_example(root, record=record)
    proc = run_checker(root)
    assert proc.returncode == 1, proc.stdout
    assert f"FAIL {SLUG}/run.json: step 1: usage must map token kinds to numbers" in proc.stdout


def test_require_runs_readme_match_is_whole_token(root):
    make_example(root)
    readme = root / "README.md"
    for text in (f"- {SLUG}-ish\n", f"- my-{SLUG}\n", f"- {SLUG}_2\n"):
        readme.write_text(text, encoding="utf-8")
        proc = run_checker(root, "--require-runs", 1)
        assert proc.returncode == 1, (text, proc.stdout)
        assert f"FAIL README.md: does not name slug {SLUG}" in proc.stdout
    readme.write_text(f"See `examples/{SLUG}/{SLUG}.png`, and {SLUG}.\n", encoding="utf-8")
    assert run_checker(root, "--require-runs", 1).returncode == 0


def test_unexpected_root_level_file_is_flagged(root):
    make_example(root)
    (root / "notes.txt").write_text("stray\n", encoding="utf-8")
    assert_single_fail(run_checker(root), root, "notes.txt", "unexpected file at examples root")
    (root / "notes.txt").unlink()
    (root / "README.md").write_text(f"{SLUG}\n", encoding="utf-8")
    assert run_checker(root).returncode == 0


def test_stray_file_still_found_when_sidecar_fails_to_load(root):
    slug_dir = make_example(root)
    path = slug_dir / f"{SLUG}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    (slug_dir / "debug.log").write_text("noise\n", encoding="utf-8")
    proc = run_checker(root)
    assert proc.returncode == 1
    assert f"FAIL {SLUG}/debug.log: unexpected file in slug dir" in proc.stdout
    assert f"FAIL {SLUG}/{SLUG}.png" not in proc.stdout
    assert f"FAIL {SLUG}/{SLUG}@{SCALE}x.png" not in proc.stdout


SECRET = "sk-ant-FAKEFAKE"


def _sidecar_edit(edit):
    def apply(slug_dir: Path) -> None:
        path = slug_dir / f"{SLUG}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    return apply


def _record_edit(edit):
    def apply(slug_dir: Path) -> None:
        path = slug_dir / "run.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    return apply


def _color_entry(d):
    d["palette"]["colors"][2] = SECRET


def _palette_name(d):
    d["palette"]["name"] = SECRET


def _spec_palette(d):
    d["spec"]["palette"] = SECRET


def _extra_key(d):
    d[SECRET] = 1


def _usage_key(d):
    d["steps"][0]["usage"][SECRET] = 5


def _command(d):
    d["steps"][0]["command"] = f"pixelgen generate x --token {SECRET}"


def _stray_name(slug_dir: Path) -> None:
    (slug_dir / f"{SECRET}.txt").write_text("x\n", encoding="utf-8")


KEY_PLACES = {
    "schema_version": _sidecar_edit(lambda d: d.update(schema_version=SECRET)),
    "palette_colors_entry": _sidecar_edit(_color_entry),
    "palette_name": _sidecar_edit(_palette_name),
    "spec_palette": _sidecar_edit(_spec_palette),
    "sidecar_key_name": _sidecar_edit(_extra_key),
    "run_usage_key_name": _record_edit(_usage_key),
    "run_command": _record_edit(_command),
    "run_notes": _record_edit(lambda d: d["steps"][0].update(notes=SECRET)),
    "stray_filename": _stray_name,
}


@pytest.mark.parametrize("place", list(KEY_PLACES))
def test_key_material_never_printed_anywhere(place, root):
    slug_dir = make_example(root)
    KEY_PLACES[place](slug_dir)
    proc = run_checker(root)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert any(ln.startswith("FAIL ") for ln in out.splitlines()), out
    assert "FAKEFAKE" not in out, out
    assert "sk-ant-" not in out, out


def test_key_in_root_argument_is_not_echoed(tmp_path):
    scratch = tmp_path / f"{SECRET}-file"
    scratch.write_text("x", encoding="utf-8")
    proc = run_checker(scratch)
    assert proc.returncode == 2
    assert "FAKEFAKE" not in proc.stdout + proc.stderr
    bad_count = run_checker(tmp_path, "--require-runs", SECRET)
    assert bad_count.returncode == 2
    assert "FAKEFAKE" not in bad_count.stdout + bad_count.stderr
