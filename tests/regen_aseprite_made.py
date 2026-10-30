"""Regenerate `tests/fixtures/aseprite/aseprite_made/` with the operator's local Aseprite.

Local only (OR-20): needs PIXELGEN_ASEPRITE; with it unset the driver prints LOCAL_SKIP_LINE and
exits 0. It writes the four Aseprite-made cases, Aseprite's own PNG export of each, and
PROVENANCE.md. It never writes generate.lua. Stdlib plus `aseprite_local` only.

    PIXELGEN_ASEPRITE=/path/to/aseprite python -I tests/regen_aseprite_made.py --out DIR
"""

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aseprite_local import (  # noqa: E402
    LOCAL_SKIP_LINE,
    aseprite_from_env,
    run_aseprite,
)

CASES = ("black0", "nonsquare", "grid128", "offset_cel")
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "aseprite" / "aseprite_made"
GENERATOR = FIXTURE_DIR / "generate.lua"
REGEN_COMMAND = (
    "PIXELGEN_ASEPRITE=/path/to/aseprite uv run python -I tests/regen_aseprite_made.py "
    "--out tests/fixtures/aseprite/aseprite_made"
)


class RegenError(RuntimeError):
    """A regeneration step failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def aseprite_version(exe: Path, *, root: Path) -> str:
    result = run_aseprite(exe, ["--version"], root=root)
    if result.returncode != 0 or not result.stdout.strip():
        raise RegenError(f"aseprite --version failed with exit code {result.returncode}")
    return result.stdout.strip()


def _step(exe: Path, args: list[str], *, root: Path, case: str, step: str, produces: Path) -> None:
    result = run_aseprite(exe, args, root=root)
    if result.returncode != 0:
        raise RegenError(f"case {case}: {step} exited {result.returncode}")
    if not produces.is_file():
        raise RegenError(
            f"case {case}: {step} exited {result.returncode} but wrote no {produces.name}"
        )


def regenerate(exe: Path, out_dir: Path, *, root: Path) -> dict[str, str]:
    """Write the 8 case files and PROVENANCE.md into out_dir; return {file: sha256} for 9 files."""
    version = aseprite_version(exe, root=root)
    out_dir = out_dir.resolve()  # the child runs with cwd=root, so it needs an absolute path
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        aseprite_file = out_dir / f"{case}.aseprite"
        _step(
            exe,
            [
                "-b",
                "--script-param",
                f"case={case}",
                "--script-param",
                f"out={aseprite_file}",
                "--script",
                str(GENERATOR),
            ],
            root=root,
            case=case,
            step="generate",
            produces=aseprite_file,
        )
        png_file = out_dir / f"{case}.png"
        _step(
            exe,
            ["-b", str(aseprite_file), "--save-as", str(png_file)],
            root=root,
            case=case,
            step="export-png",
            produces=png_file,
        )
    hashed = {}
    for case in CASES:
        for suffix in (".aseprite", ".png"):
            hashed[case + suffix] = _sha256(out_dir / (case + suffix))
    hashed["generate.lua"] = _sha256(GENERATOR)
    rows = "\n".join(f"| `{name}` | {digest} |" for name, digest in hashed.items())
    text = (
        "# Aseprite-made fixtures: provenance (PXG-24, OR-20)\n\n"
        "These files are Aseprite's own output, made from `generate.lua` by a personal source\n"
        "build on the operator's machine. Aseprite itself is never committed and never runs in\n"
        "CI.\n"
        "This file is written by `tests/regen_aseprite_made.py`, never by hand.\n\n"
        f"- aseprite --version: {version}\n"
        f"- aseprite binary sha256: {_sha256(exe)}\n\n"
        f"Regenerate (local only): `{REGEN_COMMAND}`\n\n"
        "| file | sha256 |\n|---|---|\n"
        f"{rows}\n"
    )
    (out_dir / "PROVENANCE.md").write_text(text, encoding="utf-8")
    return hashed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regen_aseprite_made", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        exe = aseprite_from_env()
    except ValueError as exc:
        print(f"regen_aseprite_made: error: {exc}", file=sys.stderr)
        return 1
    if exe is None:
        print(LOCAL_SKIP_LINE)
        return 0
    try:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            regenerate(exe, args.out, root=root)
            version = aseprite_version(exe, root=root)
    except (RegenError, OSError) as exc:
        message = " ".join(str(exc).split())
        print(f"regen_aseprite_made: error: {message}", file=sys.stderr)
        return 1
    print(f"OK {args.out}: {len(CASES)} cases, {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
