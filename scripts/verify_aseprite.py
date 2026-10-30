#!/usr/bin/env python3
"""Operator-run check that a local Aseprite reads pixelgen's own `.aseprite` files (PXG-30).

Usage: ``python scripts/verify_aseprite.py --operator NAME [--out DIR] [--case NAME ...]
[--q4-profile-prompt yes|no|OWED] [--note TEXT]``.

Aseprite is never run in CI (EULA, research O5, OR-20). With ``PIXELGEN_ASEPRITE`` unset the script
prints a fixed SKIP line, exits 0 and writes nothing. Otherwise it writes the research section 12
five-case set through ``export(..., aseprite=True)`` and runs the named Aseprite in batch mode
only: a ``--save-as`` PNG, ``--script dump.lua``, and (three cases) a batch ``--save-as`` re-save.
Aseprite runs with HOME and XDG_*_HOME in a scratch directory and no display. ``PIXELGEN_HOME``
moves to a temp dir before pixelgen is imported, and the real ``last.json`` is stat-ed before and
after. The result is ``<out>/RUN.md`` plus the per-case files that
``tests/test_aseprite_golden.py`` replays offline.

Exit codes: 0 PASS (or SKIP), 1 FAIL, 2 usage error.

Golden session (PXG-30 AC15, run by the orchestrator on the operator's go-ahead, OR-20)
---------------------------------------------------------------------------------------
From a fresh git worktree of apps/pixelgen at the landed SHA::

    stat -c '%n %Y' /home/void/.pixelgen/last.json /home/void/.config/aseprite   # BEFORE
    SCR=$(mktemp -d)
    export HOME="$SCR/home" PIXELGEN_HOME="$SCR/pxhome" XDG_CONFIG_HOME="$SCR/xdg/config" \\
      XDG_CACHE_HOME="$SCR/xdg/cache" XDG_DATA_HOME="$SCR/xdg/data" XDG_STATE_HOME="$SCR/xdg/state"
    mkdir -p "$HOME" "$PIXELGEN_HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" \\
      "$XDG_STATE_HOME"
    unset DISPLAY WAYLAND_DISPLAY
    export PIXELGEN_ASEPRITE=/home/void/.local/bin/aseprite
    uv sync --frozen
    uv run python scripts/verify_aseprite.py --operator "orchestrator (OR-20)"   # RESULT PASS
    uv run pytest tests/test_aseprite_golden.py -q -rs                           # 0 SKIPPED
    (unset PIXELGEN_ASEPRITE; uv run pytest -q -W error)
    git status --short       # only tests/fixtures/aseprite/operator/ (19 files)
    stat -c '%n %Y' /home/void/.pixelgen/last.json /home/void/.config/aseprite   # AFTER == BEFORE

Commit the 19 files in a new shell without the scratch exports (git needs the real gitconfig). A
FAIL result is never committed. Section 11 Q4 (a colour-profile prompt in the GUI) stays
operator-owed: after opening the three re-save ``ours.aseprite`` files in the GUI, re-run with
``--q4-profile-prompt yes|no`` and commit the regenerated RUN.md.
"""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ASEPRITE_ENV_VAR = "PIXELGEN_ASEPRITE"
SKIP_LINE = (
    "SKIP verify_aseprite: PIXELGEN_ASEPRITE is unset; set it to a local Aseprite build to run "
    "the golden session (PXG-30 AC15, OR-20, research §12)"
)
ASEPRITE_SCRATCH_KEYS = (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)
ASEPRITE_DROPPED_KEYS = ("DISPLAY", "WAYLAND_DISPLAY")
DUMP_TOKENS = ("PXGDUMP 1", "size ", "mode ", "transparent ", "ncolors ", "palette ", "row ", "end")
ASEPRITE_TIMEOUT_S = 120
RESAVE_CASES = ("pico8-16x16-index0", "gameboy-8x8-border", "endesga32-17x33")
Q4_VALUES = ("yes", "no", "OWED")
REPO = Path(__file__).resolve().parent.parent
DUMP_LUA = Path(__file__).resolve().parent / "dump.lua"
DEFAULT_OUT = REPO / "tests" / "fixtures" / "aseprite" / "operator"
_FIXED_NOW = (2026, 1, 1)


@dataclass(frozen=True)
class Case:
    name: str
    palette: str
    width: int
    height: int


CASES = (
    Case("pico8-16x16-index0", "pico8", 16, 16),
    Case("gameboy-8x8-border", "gameboy", 8, 8),
    Case("nes-128x128-hashed", "nes", 128, 128),
    Case("endesga32-17x33", "endesga32", 17, 33),
    Case("grayscale4-16x16-solid", "grayscale4", 16, 16),
)
CASE_NAMES = tuple(c.name for c in CASES)
CASE_FILES = ("ours.aseprite", "aseprite-export.png", "dump.txt")
RESAVE_FILE = "aseprite-resave.aseprite"


def expected_files() -> list[str]:
    """The D5 file set below the out dir, excluding RUN.md, in a stable order."""
    files = []
    for name in CASE_NAMES:
        files.extend(f"{name}/{f}" for f in CASE_FILES)
        if name in RESAVE_CASES:
            files.append(f"{name}/{RESAVE_FILE}")
    return files


def _cell(name: str, x: int, y: int, lo: int, hi: int) -> int:
    digest = hashlib.sha256(f"{name}:{x}:{y}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (hi - lo + 1) + lo


def build_case(name: str):
    """Return (grid as H rows of W ints, palette_colors (N, 3) uint8) for one D1 case."""
    from pixelgen.palettes import load_palette

    case = next((c for c in CASES if c.name == name), None)
    if case is None:
        raise KeyError(name)
    palette = load_palette(case.palette)
    n, w, h = len(palette), case.width, case.height
    if name == "grayscale4-16x16-solid":
        return [[n - 1] * w for _ in range(h)], palette
    if name == "gameboy-8x8-border":
        grid = [
            [
                -1 if x in (0, w - 1) or y in (0, h - 1) else _cell(name, x, y, 0, n - 1)
                for x in range(w)
            ]
            for y in range(h)
        ]
        forced = {(1, 1): 0, (6, 1): 3, (1, 6): 1, (6, 6): 2}
    else:
        grid = [[_cell(name, x, y, -1, n - 1) for x in range(w)] for y in range(h)]
        if name == "pico8-16x16-index0":
            forced = {(0, 0): 0, (15, 0): 15, (0, 15): -1, (15, 15): 7}
            forced.update({(x, 8): 0 for x in range(8)})
        elif name == "nes-128x128-hashed":
            forced = {(0, 0): 0, (127, 0): n - 1, (0, 127): -1}
        else:
            forced = {(0, 0): 0, (16, 0): n - 1, (0, 32): -1, (16, 32): 5}
    for (x, y), v in forced.items():
        grid[y][x] = v
    return grid, palette


# ---------------------------------------------------------------- dump format v1


class DumpFormatError(ValueError):
    """The dump text is not dump format v1."""


@dataclass(frozen=True)
class Dump:
    width: int
    height: int
    mode: str
    transparent: int
    palette: list[tuple[int, int, int, int]]
    rows: list[list[int]]


def _ints(parts: list[str], count: int, lineno: int, line: str) -> list[int]:
    if len(parts) != count:
        raise DumpFormatError(f"line {lineno}: expected {count} values: {line!r}")
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise DumpFormatError(f"line {lineno}: non-integer value: {line!r}") from None


def parse_dump(text: str) -> Dump:
    """Parse dump format v1 (scripts/dump.lua); raise DumpFormatError naming the first bad line."""
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0] != "PXGDUMP 1":
        raise DumpFormatError(f"line 1: header is not 'PXGDUMP 1': {lines[:1]!r}")

    def need(i: int, prefix: str) -> list[str]:
        if i >= len(lines):
            raise DumpFormatError(f"line {i + 1}: missing, expected '{prefix.strip()}'")
        if not lines[i].startswith(prefix):
            raise DumpFormatError(f"line {i + 1}: expected '{prefix.strip()}': {lines[i]!r}")
        return lines[i][len(prefix) :].split()

    width, height = _ints(need(1, "size "), 2, 2, lines[1] if len(lines) > 1 else "")
    mode_parts = need(2, "mode ")
    if len(mode_parts) != 1:
        raise DumpFormatError(f"line 3: bad mode: {lines[2]!r}")
    (transparent,) = _ints(need(3, "transparent "), 1, 4, lines[3])
    (ncolors,) = _ints(need(4, "ncolors "), 1, 5, lines[4])
    palette = []
    for i in range(ncolors):
        at = 5 + i
        vals = _ints(need(at, "palette "), 5, at + 1, lines[at])
        if vals[0] != i:
            raise DumpFormatError(f"line {at + 1}: palette index {vals[0]} != {i}")
        palette.append((vals[1], vals[2], vals[3], vals[4]))
    rows = []
    at = 5 + ncolors
    while at < len(lines) and lines[at].startswith("row "):
        rows.append(_ints(lines[at][4:].split(), width, at + 1, lines[at]))
        at += 1
    if len(rows) != height:
        raise DumpFormatError(f"line {at + 1}: {len(rows)} rows, size says {height}")
    if at >= len(lines) or lines[at] != "end":
        raise DumpFormatError(f"line {at + 1}: missing 'end'")
    if at + 1 != len(lines):
        raise DumpFormatError(f"line {at + 2}: text after 'end'")
    return Dump(width, height, mode_parts[0], transparent, palette, rows)


def dump_mismatches(dump: Dump, grid, palette) -> list[str]:
    """Compare a parsed dump with the D4 expectation for a case; empty means it matches."""
    n = len(palette)
    out: list[str] = []
    if (dump.width, dump.height) != (len(grid[0]), len(grid)):
        out.append(f"size {dump.width}x{dump.height} != {len(grid[0])}x{len(grid)}")
    if dump.mode != "indexed":
        out.append(f"mode {dump.mode} != indexed")
    if dump.transparent != n:
        out.append(f"transparent {dump.transparent} != {n}")
    if len(dump.palette) != n + 1:
        out.append(f"ncolors {len(dump.palette)} != {n + 1}")
    else:
        for i in range(n):
            want = (int(palette[i][0]), int(palette[i][1]), int(palette[i][2]), 255)
            if dump.palette[i] != want:
                out.append(f"palette {i} {dump.palette[i]} != {want}")
        if dump.palette[n][3] != 0:
            out.append(f"palette {n} alpha {dump.palette[n][3]} != 0")
    want_rows = [[n if v < 0 else v for v in row] for row in grid]
    if dump.rows != want_rows:
        bad = sum(
            a != b
            for r1, r2 in zip(dump.rows, want_rows, strict=False)
            for a, b in zip(r1, r2, strict=False)
        )
        out.append(f"rows differ in {bad} cells")
    return out[:6]


def pixel_diffs(ours, theirs) -> list[tuple[int, int, tuple, tuple]]:
    """Research pixel rule: where our alpha is 0 theirs must be alpha 0 (RGB ignored); elsewhere
    RGBA must be equal. Both are (H, W, 4) uint8 arrays; returns (x, y, ours, theirs) diffs."""
    import numpy

    ours = numpy.asarray(ours)
    theirs = numpy.asarray(theirs)
    clear = ours[..., 3] == 0
    bad = numpy.where(clear, theirs[..., 3] != 0, (ours != theirs).any(axis=-1))
    ys, xs = numpy.nonzero(bad)
    return [
        (int(x), int(y), tuple(int(c) for c in ours[y, x]), tuple(int(c) for c in theirs[y, x]))
        for y, x in zip(ys, xs, strict=True)
    ]


# ---------------------------------------------------------------- Aseprite driver


def _aseprite_env(root: Path) -> dict[str, str]:
    """os.environ copy with HOME and XDG_* under `root` and no display (OR-20 rule 2)."""
    env = dict(os.environ)
    dirs = {
        "HOME": root / "home",
        "XDG_CONFIG_HOME": root / "xdg" / "config",
        "XDG_CACHE_HOME": root / "xdg" / "cache",
        "XDG_DATA_HOME": root / "xdg" / "data",
        "XDG_STATE_HOME": root / "xdg" / "state",
    }
    for key in ASEPRITE_SCRATCH_KEYS:
        dirs[key].mkdir(parents=True, exist_ok=True)
        env[key] = str(dirs[key])
    for key in ASEPRITE_DROPPED_KEYS:
        env.pop(key, None)
    return env


@dataclass
class AseResult:
    rc: int | None  # None on timeout
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def why(self) -> str:
        if self.rc is None:
            return f"timed out after {ASEPRITE_TIMEOUT_S}s"
        return f"aseprite exited {self.rc}"


@dataclass
class Session:
    exe: Path
    env: dict[str, str]
    out: Path
    commands: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def run(self, args: list[str]) -> AseResult:
        shown = ["scripts/dump.lua" if a == str(DUMP_LUA) else a for a in args]
        self.commands.append("$ $PIXELGEN_ASEPRITE " + " ".join(shown))
        try:
            cp = subprocess.run(
                [str(self.exe), *args],
                env=self.env,
                cwd=self.out,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=ASEPRITE_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return AseResult(None, _text(exc.stdout), _text(exc.stderr))
        return AseResult(cp.returncode, cp.stdout, cp.stderr)


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _read_png_rgba(path: Path):
    import numpy
    from PIL import Image

    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        return rgba.size, numpy.asarray(rgba)


def run_case(case: Case, sess: Session, work: Path, report: dict) -> None:
    """Run one case; fill report with check results and failure detail."""
    import numpy

    from pixelgen.export import TokenUsage, export
    from pixelgen.pixelate import grid_to_rgba
    from pixelgen.spec import ArtSpec

    name = case.name
    grid, palette = build_case(name)
    expected_rgba = grid_to_rgba(numpy.asarray(grid, dtype=numpy.int64), palette)
    result = export(
        work / name,
        expected_rgba,
        spec=ArtSpec(
            subject=f"verify-aseprite {name}",
            width=case.width,
            height=case.height,
            palette=case.palette,
        ),
        grid=grid,
        palette_colors=palette,
        model=None,
        backend="verify-aseprite",
        token_usage=TokenUsage(),
        scale=1,
        now=datetime(*_FIXED_NOW, tzinfo=UTC),
        aseprite=True,
    )
    case_dir = sess.out / name
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result.aseprite_path, case_dir / "ours.aseprite")

    checks = {
        "export_png": "FAIL",
        "dump": "FAIL",
        "resave": "FAIL" if name in RESAVE_CASES else "n/a",
    }
    lines: list[str] = []

    def fail(check: str, detail: str, res: AseResult | None = None) -> None:
        lines.append(f"FAIL {name} {check}: {detail}")
        if res is not None:
            sess.failures.append(
                f"### {name} {check}\n\nstdout:\n\n```\n{res.stdout}\n```\n\n"
                f"stderr:\n\n```\n{res.stderr}\n```\n"
            )

    # export_png
    png = f"{name}/aseprite-export.png"
    res = sess.run(["-b", f"{name}/ours.aseprite", "--save-as", png])
    if not res.ok:
        fail("export_png", res.why(), res)
    else:
        try:
            size, theirs = _read_png_rgba(sess.out / png)
        except (OSError, ValueError) as exc:
            fail("export_png", f"unreadable PNG: {exc}")
        else:
            if size != (case.width, case.height):
                fail("export_png", f"size {size} != {(case.width, case.height)}")
            else:
                diffs = pixel_diffs(expected_rgba, theirs)
                if diffs:
                    shown = ", ".join(str(d) for d in diffs[:5])
                    fail("export_png", f"{len(diffs)} pixels differ (x, y, ours, theirs): {shown}")
                else:
                    checks["export_png"] = "PASS"

    # dump
    res = sess.run(["-b", f"{name}/ours.aseprite", "--script", str(DUMP_LUA)])
    if not res.ok:
        fail("dump", res.why(), res)
    else:
        (case_dir / "dump.txt").write_bytes(res.stdout.replace("\r\n", "\n").encode("utf-8"))
        try:
            problems = dump_mismatches(parse_dump(res.stdout), grid, palette)
        except DumpFormatError as exc:
            fail("dump", f"malformed dump: {exc}", res)
        else:
            if problems:
                fail("dump", "; ".join(problems))
            else:
                checks["dump"] = "PASS"

    # resave
    if name in RESAVE_CASES:
        resave = f"{name}/{RESAVE_FILE}"
        res = sess.run(["-b", f"{name}/ours.aseprite", "--save-as", resave])
        if not res.ok:
            fail("resave", res.why(), res)
        else:
            res = sess.run(["-b", resave, "--script", str(DUMP_LUA)])
            if not res.ok:
                fail("resave", res.why(), res)
            else:
                try:
                    problems = dump_mismatches(parse_dump(res.stdout), grid, palette)
                except DumpFormatError as exc:
                    fail("resave", f"malformed dump: {exc}", res)
                else:
                    if problems:
                        fail("resave", "; ".join(problems))
                    else:
                        checks["resave"] = "PASS"

    print(
        f"CASE {name} export_png={checks['export_png']} dump={checks['dump']} "
        f"resave={checks['resave']}"
    )
    for line in lines:
        print(line)
    report[name] = checks


# ---------------------------------------------------------------- RUN.md


def _sha256_file(path: Path) -> str:
    data = path.read_bytes()
    if path.name == "dump.txt":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str | None:
    try:
        cp = subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.stdout if cp.returncode == 0 else None


def _pixelgen_commit() -> str:
    head = _git("rev-parse", "HEAD")
    if head is None:
        return "unknown"
    status = _git("status", "--porcelain")
    return head.strip() + ("+dirty" if status is None or status.strip() else "")


def _scrub(text: str, mapping: dict[str, str]) -> str:
    for real, label in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        if real:
            text = text.replace(real, label)
    return text


def write_run_md(
    out: Path,
    *,
    operator: str,
    version: str,
    commit: str,
    ase_env: dict[str, str],
    q4: str,
    cases: list[str],
    result: str,
    note: str,
    sess: Session,
    report: dict,
    scrub: dict[str, str],
) -> None:
    display = " ".join(f"{k}={'set' if k in ase_env else 'unset'}" for k in ASEPRITE_DROPPED_KEYS)
    head = [
        "# Aseprite verification run (PXG-30, research §12)",
        "run_format: 1",
        f"date: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"operator: {operator}",
        f"aseprite_version: {' '.join(version.split())}",
        f"os: {platform.platform()}",
        f"python: {platform.python_version()}",
        f"pixelgen_version: {importlib.metadata.version('pixelgen')}",
        f"pixelgen_commit: {commit}",
        f"display: {display}",
        f"q4_profile_prompt: {q4}",
        f"cases: {', '.join(cases)}",
        f"result: {result}",
        f"note: {note}",
        "## Commands",
        *sess.commands,
        "## Cases",
        "| case | export_png | dump | resave |",
        "|---|---|---|---|",
    ]
    for name in cases:
        c = report.get(name)
        if c is None:
            head.append(f"| {name} | FAIL | FAIL | FAIL |")
        else:
            head.append(f"| {name} | {c['export_png']} | {c['dump']} | {c['resave']} |")
    if sess.failures:
        head.append("")
        head.extend(_scrub(f, scrub) for f in sess.failures)
    head.append("## Files")
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "RUN.md"):
        head.append(f"sha256 {_sha256_file(path)}  {path.relative_to(out).as_posix()}")
    (out / "RUN.md").write_text("\n".join(head) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main


def _stat(path: Path) -> tuple[bool, int, str]:
    try:
        st = path.stat()
        return True, st.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return False, 0, ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verify_aseprite", description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--operator", default=None)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--q4-profile-prompt", choices=Q4_VALUES, default="OWED")
    parser.add_argument("--note", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in args.case or []:
        if name not in CASE_NAMES:
            parser.error(f"unknown case {name!r}; choose from {', '.join(CASE_NAMES)}")
    value = os.environ.get(ASEPRITE_ENV_VAR, "")
    if not value:
        print(SKIP_LINE)
        return 0
    if not args.operator:
        parser.error("--operator is required when PIXELGEN_ASEPRITE is set")
    exe = Path(value)
    if not (exe.is_file() and os.access(exe, os.X_OK)):
        print(
            f"verify_aseprite: error: PIXELGEN_ASEPRITE is not an executable file: {value}",
            file=sys.stderr,
        )
        return 1

    guarded: dict[Path, tuple[bool, int, str]] = {}
    sources = [Path.home() / ".pixelgen" / "last.json"]
    inherited = os.environ.get("PIXELGEN_HOME", "")
    if inherited:
        sources.append(Path(inherited) / "last.json")
    for path in sources:
        guarded.setdefault(path, _stat(path))
    commit = _pixelgen_commit()

    tmp_home = tempfile.mkdtemp(prefix="pixelgen-verify-home-")
    os.environ["PIXELGEN_HOME"] = tmp_home
    ase_root = tempfile.mkdtemp(prefix="pixelgen-verify-aseprite-")
    work = Path(tempfile.mkdtemp(prefix="pixelgen-verify-work-"))
    out = Path(args.out).absolute()
    try:
        ase_env = _aseprite_env(Path(ase_root))
        out.mkdir(parents=True, exist_ok=True)
        sess = Session(exe, ase_env, out)
        names = [c for c in CASE_NAMES if c in (args.case or CASE_NAMES)]
        report: dict = {}
        ver = sess.run(["--version"])
        failed = False
        if not ver.ok:
            print(f"verify_aseprite: error: --version {ver.why()}", file=sys.stderr)
            failed = True
        else:
            for case in CASES:
                if case.name in names:
                    run_case(case, sess, work, report)
            failed = any("FAIL" in c.values() for c in report.values()) or len(report) != len(names)
        incidents = []
        for path, before in guarded.items():
            if _stat(path) != before:
                print(f"verify_aseprite: error: real last.json changed: {path}", file=sys.stderr)
                incidents.append(path)
            else:
                print(f"HOME unchanged: {path}")
        result = "FAIL" if failed or incidents else "PASS"
        scrub = {
            str(exe): "$PIXELGEN_ASEPRITE",
            str(out): "<out>",
            ase_root: "<aseprite-scratch>",
            tmp_home: "<pixelgen-home>",
            str(work): "<work>",
            str(REPO): "<repo>",
            str(Path.home()): "<home>",
        }
        write_run_md(
            out,
            operator=args.operator,
            version=ver.stdout,
            commit=commit,
            ase_env=ase_env,
            q4=args.q4_profile_prompt,
            cases=names,
            result=result,
            note=args.note,
            sess=sess,
            report=report,
            scrub=scrub,
        )
        print(f"RESULT {result} {out / 'RUN.md'}")
        return 0 if result == "PASS" else 1
    finally:
        for path in (tmp_home, ase_root, str(work)):
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
