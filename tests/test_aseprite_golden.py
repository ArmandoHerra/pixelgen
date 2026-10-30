"""PXG-30: scripts/verify_aseprite.py against a stub Aseprite, and the offline golden replay.

No test here runs the real Aseprite except `test_dump_lua_on_real_aseprite`, which skips unless
PIXELGEN_ASEPRITE is set (OR-20). Every script run goes through `_run_script`, whose environment is
built by `_script_env`: HOME is always a tmp_path directory, never the shell's.
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import aseprite_oracle as ao
import numpy as np
import pytest
from aseprite_local import require_aseprite, run_aseprite
from PIL import Image

from pixelgen.aseprite import encode_aseprite
from pixelgen.pixelate import grid_to_rgba

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
SCRIPT = REPO / "scripts" / "verify_aseprite.py"
DUMP_LUA = REPO / "scripts" / "dump.lua"
THIRD_PARTY = TESTS / "fixtures" / "aseprite" / "third_party"
INDEXED = THIRD_PARTY / "indexed.aseprite"
INDEXED_PNG = THIRD_PARTY / "indexed_01.png"
OPERATOR_DIR = TESTS / "fixtures" / "aseprite" / "operator"
OWNER_SKIP_LINE = (
    "SKIP tests/fixtures/aseprite/operator/ absent: the Aseprite golden session is owed "
    "(owner: orchestrator on operator go-ahead; PXG-30 AC15, OR-20, research §12)"
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="stub Aseprite needs /bin/sh")


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_aseprite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_aseprite"] = module  # dataclasses resolve their module through here
    spec.loader.exec_module(module)
    return module


va = _load_script()

RESAVE = ("pico8-16x16-index0", "gameboy-8x8-border", "endesga32-17x33")
D5_FILES = [
    f"{c}/{f}" for c in va.CASE_NAMES for f in ("ours.aseprite", "aseprite-export.png", "dump.txt")
] + [f"{c}/aseprite-resave.aseprite" for c in RESAVE]
EXPECTED_SCRATCH_KEYS = (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)
STUB_ENV_KEYS = (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "DISPLAY",
    "WAYLAND_DISPLAY",
)
TOUCH_BYTES = b'{"pxg30_stub": "touch-home"}\n'
DRIFT = (
    "{case}: ours.aseprite no longer matches the writer's output for this case; "
    "re-run the §12 procedure (research §12, PXG-30)"
)

# ------------------------------------------------------------------ stub Aseprite

_STUB = r"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, @@TESTS@@)
import aseprite_oracle as ao
from PIL import Image

argv = sys.argv[1:]
mode = os.environ.get("PXG30_STUB_MODE", "ok")


def spawner_home():
    # The script's own HOME (its parent process), before the script swapped in the scratch one.
    try:
        raw = Path(f"/proc/{os.getppid()}/environ").read_bytes()
    except OSError:
        return None
    for item in raw.split(b"\0"):
        if item.startswith(b"HOME="):
            return Path(item[5:].decode())
    return None


log_path = os.environ.get("PXG30_STUB_ENV_LOG")
if log_path:
    keys = ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
            "DISPLAY", "WAYLAND_DISPLAY")
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"argv": argv, "env": {k: os.environ.get(k) for k in keys}}) + "\n")
    cfg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.environ.get("HOME", ""), ".config")
    Path(cfg, "aseprite").mkdir(parents=True, exist_ok=True)
    Path(cfg, "aseprite", "Aseprite.log").write_text("stub\n")

if argv == ["--version"]:
    print("Aseprite 1.3-stub")
    sys.exit(0)

_b, src, kind, target = argv
ase = ao.parse_aseprite(Path(src).read_bytes())
if kind == "--save-as":
    if mode == "touch-home" and target.endswith(".png"):
        home = spawner_home()
        root = os.environ.get("PXG30_STUB_FAKE_ROOT")
        if not root or home is None or not home.resolve().is_relative_to(Path(root).resolve()):
            print(f"stub: refusing to write outside PXG30_STUB_FAKE_ROOT: {home}", file=sys.stderr)
            sys.exit(97)
        (home / ".pixelgen").mkdir(parents=True, exist_ok=True)
        (home / ".pixelgen" / "last.json").write_bytes(b'{"pxg30_stub": "touch-home"}\n')
    if target.endswith(".png"):
        grid = ao.to_sidecar_grid(ase)
        rows = []
        for row in grid:
            for v in row:
                clear = v < 0 or (mode == "index0-transparent" and v == 0)
                rows.append((0, 0, 0, 0) if clear else (*ase.palette[v][:3], 255))
        img = Image.new("RGBA", (ase.width, ase.height))
        img.putdata(rows)
        img.save(target)
    else:
        Path(target).write_bytes(Path(src).read_bytes())
else:
    assert kind == "--script" and target.endswith("dump.lua") and Path(target).exists(), argv
    transparent = 0 if mode == "index0-transparent" else ase.transparent_index
    print("PXGDUMP 1")
    print(f"size {ase.width} {ase.height}")
    print("mode indexed" if ase.color_depth == 8 else "mode other")
    print(f"transparent {transparent}")
    print(f"ncolors {len(ase.palette)}")
    for i, (r, g, b, a) in enumerate(ase.palette):
        print(f"palette {i} {r} {g} {b} {a}")
    for row in ao.composite_indices(ase, 0):
        print("row " + " ".join(str(v) for v in row))
    print("end")
"""


def _write_stub(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    stub = tmp / "stub_aseprite.py"
    stub.write_text(_STUB.replace("@@TESTS@@", repr(str(TESTS))), encoding="utf-8")
    wrapper = tmp / "fake-aseprite"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stub}" "$@"\n', encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


_DEFAULT = object()
_REMOVED_ENV = (
    "HOME",
    "PIXELGEN_HOME",
    "PIXELGEN_ASEPRITE",
    "PXG30_STUB_MODE",
    "PXG30_STUB_FAKE_ROOT",
    "PXG30_STUB_ENV_LOG",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "DISPLAY",
    "WAYLAND_DISPLAY",
)


def _script_env(home: Path, *, pixelgen_home=_DEFAULT, **extra: str) -> dict[str, str]:
    """Environment for a script run: HOME is the given tmp path, never the shell's."""
    env = {k: v for k, v in os.environ.items() if k not in _REMOVED_ENV}
    env["HOME"] = str(home)
    if pixelgen_home is _DEFAULT:
        env["PIXELGEN_HOME"] = str(home.parent / "pxhome")
    elif pixelgen_home is not None:
        env["PIXELGEN_HOME"] = str(pixelgen_home)
    env.update(extra)
    return env


def _run_script(env: dict[str, str], out: Path, *args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), "--out", str(out)]
    if "PIXELGEN_ASEPRITE" in env:
        cmd += ["--operator", "pxg30-test"]
    return subprocess.run(
        [*cmd, *args],
        env=env,
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _stub_env(
    tmp: Path, home: Path, mode: str = "ok", *, pixelgen_home=_DEFAULT, **extra: str
) -> dict[str, str]:
    stub = _write_stub(tmp / "stub")
    return _script_env(
        home,
        pixelgen_home=pixelgen_home,
        PIXELGEN_ASEPRITE=str(stub),
        PXG30_STUB_MODE=mode,
        **extra,
    )


# ------------------------------------------------------------------ RUN.md helpers


def _sha(path: Path) -> str:
    import hashlib

    data = path.read_bytes()
    if path.name == "dump.txt":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _run_keys(text: str) -> dict[str, str]:
    keys = {}
    for line in text.split("## Commands")[0].splitlines()[1:]:
        key, _, value = line.partition(":")
        keys[key.strip()] = value.strip()
    return keys


def _manifest(text: str) -> dict[str, str]:
    out = {}
    for line in text.split("## Files\n", 1)[1].splitlines() if "## Files\n" in text else []:
        m = re.fullmatch(r"sha256 ([0-9a-f]{64})  (.+)", line)
        if m:
            out[m.group(2)] = m.group(1)
    return out


def _pixel_problem(case: str, name: str, got_rgba, want_rgba) -> str | None:
    diffs = va.pixel_diffs(want_rgba, got_rgba)
    if not diffs:
        return None
    shown = ", ".join(str(d) for d in diffs[:5])
    return f"{case}: {name} differs from the expected pixels in {len(diffs)} cells: {shown}"


def _replay_problems(root: Path) -> list[str]:
    """Offline, oracle-only check of an operator directory; empty means it replays clean."""
    problems: list[str] = []
    run_md = root / "RUN.md"
    manifest: dict[str, str] | None = None
    if not run_md.is_file():
        problems.append("RUN.md: missing file")
    else:
        text = run_md.read_text(encoding="utf-8")
        keys = _run_keys(text)
        manifest = _manifest(text)
        if keys.get("run_format") != "1":
            problems.append(f"RUN.md: run_format {keys.get('run_format')!r} is not '1'")
        for key in (
            "date",
            "operator",
            "aseprite_version",
            "os",
            "python",
            "pixelgen_version",
            "pixelgen_commit",
            "display",
            "q4_profile_prompt",
            "cases",
            "result",
        ):
            if not keys.get(key):
                problems.append(f"RUN.md: {key} is missing or empty")
        if keys.get("q4_profile_prompt") and keys["q4_profile_prompt"] not in va.Q4_VALUES:
            problems.append(
                f"RUN.md: q4_profile_prompt {keys['q4_profile_prompt']!r} not in yes/no/OWED"
            )
        if keys.get("result") and keys["result"] != "PASS":
            problems.append(f"RUN.md: result {keys['result']!r} is not PASS")
        listed = [c.strip() for c in keys.get("cases", "").split(",") if c.strip()]
        if keys.get("cases") and sorted(listed) != sorted(va.CASE_NAMES):
            problems.append(f"RUN.md: cases {listed} is not all five case names")

    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "RUN.md"
    }
    for rel in sorted(present - set(D5_FILES)):
        problems.append(f"unexpected file: {rel}")
    for rel in D5_FILES:
        if rel not in present:
            problems.append(f"missing file: {rel}")

    if manifest is not None:
        for rel in sorted(present | set(manifest)):
            path = root / rel
            if rel not in manifest or not path.is_file() or manifest[rel] != _sha(path):
                problems.append(f"{rel}: sha256 differs from RUN.md")

    for case in va.CASE_NAMES:
        grid, palette = va.build_case(case)
        n = len(palette)
        want_rgba = grid_to_rgba(np.asarray(grid, dtype=np.int64), palette)
        ours = root / case / "ours.aseprite"
        if ours.is_file():
            try:
                writer = ao.parse_aseprite(encode_aseprite(grid, palette))
                have = ao.parse_aseprite(ours.read_bytes())
                key = lambda a: (  # noqa: E731
                    a.width,
                    a.height,
                    a.color_depth,
                    a.transparent_index,
                    a.palette,
                    [f.chunk_types for f in a.frames],
                    ao.composite_indices(a),
                )
                if key(have) != key(writer):
                    problems.append(DRIFT.format(case=case))
            except ao.AseFormatError as exc:
                problems.append(f"{case}: ours.aseprite is not a readable .aseprite file: {exc}")
        png = root / case / "aseprite-export.png"
        if png.is_file():
            try:
                with Image.open(png) as img:
                    got = np.asarray(img.convert("RGBA"))
            except OSError as exc:
                problems.append(f"{case}: aseprite-export.png is unreadable: {exc}")
            else:
                if got.shape != want_rgba.shape:
                    problems.append(f"{case}: aseprite-export.png has shape {got.shape}")
                elif msg := _pixel_problem(case, "aseprite-export.png", got, want_rgba):
                    problems.append(msg)
        dump = root / case / "dump.txt"
        if dump.is_file():
            try:
                mism = va.dump_mismatches(
                    va.parse_dump(dump.read_text(encoding="utf-8")), grid, palette
                )
            except va.DumpFormatError as exc:
                problems.append(f"{case}: dump.txt is malformed: {exc}")
            else:
                if mism:
                    problems.append(f"{case}: dump.txt differs from the case: {'; '.join(mism)}")
        resave = root / case / "aseprite-resave.aseprite"
        if resave.is_file() and case in RESAVE:
            try:
                ase = ao.parse_aseprite(resave.read_bytes())
                bad = []
                if ao.to_sidecar_grid(ase) != grid:
                    bad.append("pixels differ")
                if ase.transparent_index != n:
                    bad.append(f"transparent index {ase.transparent_index} != {n}")
                if [c[:3] for c in ase.palette[:n]] != [tuple(int(v) for v in c) for c in palette]:
                    bad.append("palette RGB differs")
            except ao.AseFormatError as exc:
                bad = [f"unreadable: {exc}"]
            if bad:
                problems.append(f"{case}: aseprite-resave.aseprite {'; '.join(bad)}")
    return problems


def _operator_gate(path: Path) -> list[str]:
    if not path.exists():
        pytest.skip(OWNER_SKIP_LINE)
    return _replay_problems(path)


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("stubrun")
    out = base / "op"
    env = _stub_env(base, base / "home")
    proc = _run_script(env, out)
    return SimpleNamespace(base=base, out=out, proc=proc, stub=Path(env["PIXELGEN_ASEPRITE"]))


@pytest.fixture
def run_copy(stub_run, tmp_path):
    dest = tmp_path / "op"
    shutil.copytree(stub_run.out, dest)
    return dest


def _rewrite_line(root: Path, rel: str) -> None:
    """Refresh (or add) the RUN.md manifest line of `rel` from the file on disk."""
    run_md = root / "RUN.md"
    text = run_md.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.endswith(f"  {rel}")]
    lines.append(f"sha256 {_sha(root / rel)}  {rel}")
    run_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _drop_line(root: Path, rel: str) -> None:
    run_md = root / "RUN.md"
    lines = [ln for ln in run_md.read_text().splitlines() if not ln.endswith(f"  {rel}")]
    run_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_key(root: Path, key: str, value: str) -> None:
    run_md = root / "RUN.md"
    text = re.sub(
        rf"(?m)^{key}: .*$", f"{key}: {value}", run_md.read_text(encoding="utf-8"), count=1
    )
    run_md.write_text(text, encoding="utf-8")


def _recompress(data: bytes) -> bytes:
    """Re-deflate the cel stream at a level that changes its bytes; patch every size field."""
    pos = 128 + 16
    (frame_size,) = struct.unpack_from("<I", data, 128)
    end = 128 + frame_size
    while pos < end:
        csize, ctype = struct.unpack_from("<IH", data, pos)
        if ctype == 0x2005 and struct.unpack_from("<H", data, pos + 6 + 7)[0] == 2:
            start = pos + 6 + 16 + 4
            old = data[start : pos + csize]
            raw = zlib.decompress(old)
            for level in (9, 1, 0):
                new = zlib.compress(raw, level)
                if new != old:
                    break
            assert new != old, "recompression must change the bytes"
            delta = len(new) - len(old)
            out = bytearray(data[:start] + new + data[pos + csize :])
            struct.pack_into("<I", out, pos, csize + delta)
            struct.pack_into("<I", out, 128, frame_size + delta)
            struct.pack_into("<I", out, 0, len(out))
            assert bytes(out) != data
            return bytes(out)
        pos += csize
    raise AssertionError("no zlib cel chunk found")


# ------------------------------------------------------------------ AC1, AC2, AC3


def test_script_skips_without_aseprite(tmp_path):
    out = tmp_path / "op"
    cp = _run_script(_script_env(tmp_path / "home"), out)
    assert cp.returncode == 0
    assert cp.stdout == va.SKIP_LINE + "\n"
    assert not out.exists()


def test_script_rejects_unusable_aseprite(tmp_path):
    out = tmp_path / "op"
    env = _script_env(tmp_path / "home", PIXELGEN_ASEPRITE=str(tmp_path / "missing"))
    cp = _run_script(env, out)
    assert cp.returncode == 1
    err = cp.stderr.strip().splitlines()
    assert len(err) == 1
    assert err[0].startswith("verify_aseprite: error: PIXELGEN_ASEPRITE is not an executable file:")
    assert not out.exists()


def _seed(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b'{"seed": "fake last.json"}\n'
    path.write_bytes(data)
    os.utime(path, (1_000_000_000, 1_000_000_000))
    return data


@pytest.mark.parametrize("arm", ["inherited", "default"])
def test_script_never_writes_real_home(tmp_path, arm):
    if arm == "default":
        home = tmp_path / "home"
        env = _stub_env(tmp_path, home, pixelgen_home=None)
        guarded = [home / ".pixelgen" / "last.json"]
    else:
        home = tmp_path / "home2"
        env = _stub_env(tmp_path, home, pixelgen_home=tmp_path / "home" / ".pixelgen")
        guarded = [home / ".pixelgen" / "last.json", tmp_path / "home" / ".pixelgen" / "last.json"]
    seeds = {p: _seed(p) for p in guarded}
    cp = _run_script(env, tmp_path / "op", "--case", "pico8-16x16-index0")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    for path, data in seeds.items():
        assert path.read_bytes() == data
        assert path.stat().st_mtime_ns == 1_000_000_000 * 10**9
    lines = [ln for ln in cp.stdout.splitlines() if ln.startswith("HOME unchanged: ")]
    assert lines == [f"HOME unchanged: {p}" for p in guarded]


def test_script_flags_real_home_change(tmp_path):
    home = tmp_path / "home"
    env = _stub_env(tmp_path, home, "touch-home", PXG30_STUB_FAKE_ROOT=str(tmp_path))
    fake = home / ".pixelgen" / "last.json"
    _seed(fake)
    cp = _run_script(env, tmp_path / "op", "--case", "pico8-16x16-index0")
    assert cp.returncode == 1, cp.stdout + cp.stderr
    assert f"verify_aseprite: error: real last.json changed: {fake}" in cp.stderr
    assert cp.stdout.strip().splitlines()[-1].startswith("RESULT FAIL")
    assert fake.read_bytes() == TOUCH_BYTES


# ------------------------------------------------------------------ AC4


def test_script_import_surface_and_home_order():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    heavy = {"pixelgen", "numpy", "PIL"}
    banned_mods = {"pixelgen.llm", "pixelgen.cli", "pixelgen.pipeline", "pixelgen.backends"}
    banned_names = {"llm", "cli", "pipeline", "backends"}

    def imported(node):
        if isinstance(node, ast.Import):
            return [a.name for a in node.names]
        if isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
            if node.module == "pixelgen":
                mods += [f"pixelgen.{a.name}" for a in node.names]
            return mods
        return []

    for node in tree.body:
        for mod in imported(node):
            assert mod.split(".")[0] not in heavy, f"module-level import of {mod}"
    for node in ast.walk(tree):
        for mod in imported(node):
            assert mod.split(".")[0] != "anthropic", f"forbidden import {mod} (line {node.lineno})"
            assert mod not in banned_mods, f"forbidden import {mod} (line {node.lineno})"
        if isinstance(node, ast.ImportFrom) and node.module == "pixelgen":
            hit = banned_names & {a.name for a in node.names}
            assert not hit, f"forbidden 'from pixelgen import {sorted(hit)}' (line {node.lineno})"

    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    main = funcs["main"]

    def is_home_assign(n):
        return isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Subscript)
            and ast.unparse(t.value) == "os.environ"
            and ast.unparse(t.slice) == "'PIXELGEN_HOME'"
            for t in n.targets
        )

    assigns = [n.lineno for n in ast.walk(main) if is_home_assign(n)]
    assert assigns, "main never assigns os.environ['PIXELGEN_HOME']"
    first = min(assigns)
    for node in ast.walk(main):
        if any(m.split(".")[0] == "pixelgen" for m in imported(node)):
            assert node.lineno > first, f"pixelgen import in main (line {node.lineno}) before home"
    importers = {
        name
        for name, fn in funcs.items()
        if name != "main"
        and any(m.split(".")[0] == "pixelgen" for n in ast.walk(fn) for m in imported(n))
    }
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in importers:
                assert node.lineno > first, f"{node.func.id}() imports pixelgen, called before home"


# ------------------------------------------------------------------ AC5


def test_case_set_matches_research_s12():
    spec = [
        ("pico8-16x16-index0", "pico8", 16, 16, 16),
        ("gameboy-8x8-border", "gameboy", 8, 8, 4),
        ("nes-128x128-hashed", "nes", 128, 128, 54),
        ("endesga32-17x33", "endesga32", 17, 33, 32),
        ("grayscale4-16x16-solid", "grayscale4", 16, 16, 4),
    ]
    assert [(c.name, c.palette, c.width, c.height) for c in va.CASES] == [s[:4] for s in spec]
    grids = {}
    for name, _pal, w, h, n in spec:
        grid, palette = va.build_case(name)
        assert palette.shape == (n, 3)
        assert (len(grid), {len(r) for r in grid}) == (h, {w})
        assert va.build_case(name)[0] == grid
        grids[name] = grid
    flat = lambda g: [v for r in g for v in r]  # noqa: E731
    p = grids["pico8-16x16-index0"]
    assert p[0][0] == 0 and 15 in flat(p) and -1 in flat(p)
    g = grids["gameboy-8x8-border"]
    border = [g[y][x] for y in range(8) for x in range(8) if x in (0, 7) or y in (0, 7)]
    interior = [g[y][x] for y in range(1, 7) for x in range(1, 7)]
    assert set(border) == {-1}
    assert set(interior) == {0, 1, 2, 3}
    assert {0, 53, -1} <= set(flat(grids["nes-128x128-hashed"]))
    e = grids["endesga32-17x33"]
    assert len(e) == 33 and {len(r) for r in e} == {17} and e[0][0] == 0 and e[0][16] == 31
    assert set(flat(grids["grayscale4-16x16-solid"])) == {3}


# ------------------------------------------------------------------ AC6, AC7


def test_script_end_to_end_with_stub_aseprite(stub_run):
    cp, out = stub_run.proc, stub_run.out
    assert cp.returncode == 0, cp.stdout + cp.stderr
    lines = cp.stdout.splitlines()
    cases = [ln for ln in lines if ln.startswith("CASE ")]
    assert len(cases) == 5
    for ln in cases:
        name = ln.split()[1]
        assert ln.endswith("resave=PASS" if name in RESAVE else "resave=n/a"), ln
    assert lines[-1].startswith("RESULT PASS ")
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == sorted([*D5_FILES, "RUN.md"]) and len(files) == 19
    text = (out / "RUN.md").read_text(encoding="utf-8")
    keys = _run_keys(text)
    for key, value in keys.items():
        if key != "note":
            assert value, key
    assert keys["q4_profile_prompt"] == "OWED" and keys["result"] == "PASS"
    assert [c.strip() for c in keys["cases"].split(",")] == list(va.CASE_NAMES)
    commands = text.split("## Commands")[1].split("## Cases")[0].splitlines()
    assert len([c for c in commands if c.startswith("$ $PIXELGEN_ASEPRITE -b")]) == 16
    assert str(stub_run.stub) not in text and str(stub_run.base) not in text
    assert _replay_problems(out) == []


def test_script_flags_index0_transparency(tmp_path):
    env = _stub_env(tmp_path, tmp_path / "home", "index0-transparent")
    cp = _run_script(env, tmp_path / "op", "--case", "pico8-16x16-index0")
    assert cp.returncode == 1, cp.stdout + cp.stderr
    assert "FAIL pico8-16x16-index0 export_png:" in cp.stdout
    assert "FAIL pico8-16x16-index0 dump:" in cp.stdout
    assert cp.stdout.strip().splitlines()[-1].startswith("RESULT FAIL")


# ------------------------------------------------------------------ AC8

SAMPLE = """PXGDUMP 1
size 2 2
mode indexed
transparent 2
ncolors 3
palette 0 255 0 0 255
palette 1 0 255 0 255
palette 2 0 0 0 0
row 0 1
row 2 0
end
"""


def test_dump_lua_emits_parsed_format():
    lua = DUMP_LUA.read_text(encoding="utf-8")
    for token in va.DUMP_TOKENS:
        assert re.search(r'print\(\s*"' + re.escape(token), lua), f"dump.lua never prints {token!r}"
    d = va.parse_dump(SAMPLE)
    assert d == va.Dump(
        2, 2, "indexed", 2, [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 0, 0)], [[0, 1], [2, 0]]
    )


@pytest.mark.parametrize(
    "variant,text",
    [
        ("header_v2", SAMPLE.replace("PXGDUMP 1", "PXGDUMP 2")),
        ("row_narrow", SAMPLE.replace("row 0 1", "row 0")),
        ("no_end", SAMPLE.replace("end\n", "")),
        ("ncolors_mismatch", SAMPLE.replace("ncolors 3", "ncolors 4")),
    ],
)
def test_parse_dump_rejects_malformed(variant, text):
    with pytest.raises(va.DumpFormatError):
        va.parse_dump(text)


# ------------------------------------------------------------------ AC9, AC10, AC11


def test_replay_flags_writer_drift(run_copy):
    grid, palette = va.build_case("pico8-16x16-index0")
    grid[3][3] = (grid[3][3] + 1) % 16
    rel = "pico8-16x16-index0/ours.aseprite"
    (run_copy / rel).write_bytes(encode_aseprite(grid, palette))
    _rewrite_line(run_copy, rel)
    assert _replay_problems(run_copy) == [DRIFT.format(case="pico8-16x16-index0")]


def test_replay_ignores_recompression(run_copy):
    rel = "pico8-16x16-index0/ours.aseprite"
    before = (run_copy / rel).read_bytes()
    after = _recompress(before)
    assert after != before
    (run_copy / rel).write_bytes(after)
    _rewrite_line(run_copy, rel)
    assert _replay_problems(run_copy) == []


def _edit_png(path: Path, xy, rgba):
    with Image.open(path) as img:
        im = img.convert("RGBA")
    im.putpixel(xy, rgba)
    im.save(path)


def _arm_export_index0(root):
    rel = "pico8-16x16-index0/aseprite-export.png"
    with Image.open(root / rel) as img:
        r, g, b, _a = img.convert("RGBA").getpixel((0, 0))
    _edit_png(root / rel, (0, 0), (r, g, b, 0))
    return rel


def _arm_export_rgb_under_alpha0(root):
    rel = "pico8-16x16-index0/aseprite-export.png"
    _edit_png(root / rel, (0, 15), (255, 0, 255, 0))
    return rel


def _arm_dump_transparent(root):
    rel = "pico8-16x16-index0/dump.txt"
    p = root / rel
    p.write_text(p.read_text().replace("transparent 16", "transparent 0"))
    return rel


def _arm_resave_grid(root):
    rel = "endesga32-17x33/aseprite-resave.aseprite"
    grid, palette = va.build_case("endesga32-17x33")
    grid[0][0] = 1
    (root / rel).write_bytes(encode_aseprite(grid, palette))
    return rel


@pytest.mark.parametrize(
    "arm,prefix",
    [
        (_arm_export_index0, "pico8-16x16-index0: aseprite-export.png"),
        (_arm_export_rgb_under_alpha0, None),
        (_arm_dump_transparent, "pico8-16x16-index0: dump.txt"),
        (_arm_resave_grid, "endesga32-17x33: aseprite-resave.aseprite"),
    ],
    ids=["export_index0", "export_rgb_under_alpha0", "dump_transparent", "resave_grid"],
)
def test_replay_checks_aseprite_outputs(run_copy, arm, prefix):
    rel = arm(run_copy)
    _rewrite_line(run_copy, rel)
    problems = _replay_problems(run_copy)
    if prefix is None:
        assert problems == []
    else:
        assert len(problems) == 1 and problems[0].startswith(prefix), problems


def _arm_sha_mismatch(root):
    run_md = root / "RUN.md"
    rel = "pico8-16x16-index0/ours.aseprite"
    digest = _sha(root / rel)
    line = f"sha256 {digest}  {rel}"  # the resave copy shares the digest, so match the whole line
    assert line in run_md.read_text()
    flipped = ("0" if digest[0] != "0" else "1") + digest[1:]
    run_md.write_text(run_md.read_text().replace(line, f"sha256 {flipped}  {rel}"))
    return [f"{rel}: sha256 differs from RUN.md"]


def _arm_unexpected(root):
    rel = "nes-128x128-hashed/notes.txt"
    (root / rel).write_text("hello\n")
    _rewrite_line(root, rel)
    return [f"unexpected file: {rel}"]


def _arm_resave_missing(root):
    rel = "gameboy-8x8-border/aseprite-resave.aseprite"
    (root / rel).unlink()
    _drop_line(root, rel)
    return [f"missing file: {rel}"]


def _arm_q4_unanswered(root):
    _set_key(root, "q4_profile_prompt", "UNANSWERED")
    return "RUN.md: q4_profile_prompt"


def _arm_q4_answered(root):
    _set_key(root, "q4_profile_prompt", "no")
    return []


def _arm_result_fail(root):
    _set_key(root, "result", "FAIL")
    return "RUN.md: result"


@pytest.mark.parametrize(
    "arm",
    [
        _arm_sha_mismatch,
        _arm_unexpected,
        _arm_resave_missing,
        _arm_q4_unanswered,
        _arm_q4_answered,
        _arm_result_fail,
    ],
    ids=[
        "sha_mismatch",
        "unexpected",
        "resave_missing",
        "q4_unanswered",
        "q4_answered",
        "result_fail",
    ],
)
def test_replay_checks_record(run_copy, arm):
    want = arm(run_copy)
    problems = _replay_problems(run_copy)
    if isinstance(want, str):
        assert len(problems) == 1 and problems[0].startswith(want), problems
    else:
        assert problems == want


# ------------------------------------------------------------------ AC12


@pytest.mark.parametrize("arm", ["absent", "empty", "no_run_md"])
def test_operator_gate_arms(stub_run, tmp_path, arm):
    root = tmp_path / "op"
    if arm == "absent":
        with pytest.raises(pytest.skip.Exception) as info:
            _operator_gate(root)
        assert info.value.msg == (
            "SKIP tests/fixtures/aseprite/operator/ absent: the Aseprite golden session is owed "
            "(owner: orchestrator on operator go-ahead; PXG-30 AC15, OR-20, research §12)"
        )
        return
    if arm == "empty":
        root.mkdir()
    else:
        shutil.copytree(stub_run.out, root)
        (root / "RUN.md").unlink()
    try:
        problems = _operator_gate(root)
    except pytest.skip.Exception as e:
        pytest.fail(f"gate skipped a present directory: {e.msg}")
    if arm == "empty":
        assert len(problems) == 19
        assert problems[0] == "RUN.md: missing file"
        assert problems[1:] == [f"missing file: {rel}" for rel in D5_FILES]
    else:
        assert problems == ["RUN.md: missing file"]


def test_operator_golden_replay():
    problems = _operator_gate(OPERATOR_DIR)
    assert problems == [], "\n".join(problems)


# ------------------------------------------------------------------ AC17


def test_script_runs_aseprite_in_scratch_home(tmp_path):
    xdg_in = tmp_path / "xdg-in"
    log = tmp_path / "env.jsonl"
    env = _stub_env(
        tmp_path,
        tmp_path / "home",
        XDG_CONFIG_HOME=str(xdg_in / "config"),
        XDG_CACHE_HOME=str(xdg_in / "cache"),
        XDG_DATA_HOME=str(xdg_in / "data"),
        XDG_STATE_HOME=str(xdg_in / "state"),
        DISPLAY=":99",
        WAYLAND_DISPLAY="wayland-pxg30",
        PXG30_STUB_ENV_LOG=str(log),
    )
    out = tmp_path / "op"
    cp = _run_script(env, out, "--case", "pico8-16x16-index0")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    records = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert len(records) == 5
    common = None
    for rec in records:
        assert set(rec["env"]) == set(STUB_ENV_KEYS)
        paths = []
        for key in EXPECTED_SCRATCH_KEYS:
            value = rec["env"][key]
            assert value and os.path.isabs(value), f"{key} is not a scratch path: {value!r}"
            for forbidden in (tmp_path / "home", xdg_in):
                p = Path(value)
                assert p != forbidden and forbidden not in p.parents, f"{key} leaks {value}"
            paths.append(value)
        common = os.path.commonpath(paths)
        prefix = "pixelgen-verify-aseprite-"
        assert os.path.basename(common).startswith(prefix), f"scratch keys share {common}"
        assert rec["env"]["DISPLAY"] is None and rec["env"]["WAYLAND_DISPLAY"] is None
    assert not Path(common).exists()
    assert not (xdg_in / "config" / "aseprite").exists()
    assert not (tmp_path / "home" / ".config").exists()
    assert "display: DISPLAY=unset WAYLAND_DISPLAY=unset" in (out / "RUN.md").read_text()
    assert va.ASEPRITE_SCRATCH_KEYS == EXPECTED_SCRATCH_KEYS


# ------------------------------------------------------------------ AC18 (local only)


def test_dump_lua_on_real_aseprite(tmp_path):
    exe = require_aseprite()
    cp = run_aseprite(exe, ["-b", str(INDEXED), "--script", str(DUMP_LUA)], root=tmp_path)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    d = va.parse_dump(cp.stdout)
    ase = ao.parse_aseprite(INDEXED.read_bytes())
    assert (d.width, d.height, d.mode) == (64, 64, "indexed")
    assert d.transparent == ase.transparent_index
    assert d.palette == ase.palette
    assert d.rows == ao.composite_indices(ase, 0)
    rendered = np.zeros((d.height, d.width, 4), dtype=np.uint8)
    for y, row in enumerate(d.rows):
        for x, v in enumerate(row):
            rendered[y, x] = (0, 0, 0, 0) if v == d.transparent else d.palette[v]
    with Image.open(INDEXED_PNG) as img:
        theirs = np.asarray(img.convert("RGBA"))
    assert va.pixel_diffs(rendered, theirs) == []
