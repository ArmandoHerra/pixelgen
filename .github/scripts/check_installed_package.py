"""Check that the installed pixelgen is non-editable and ships every source file.

Usage: check_installed_package.py SRC_PKG_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_installed_package.py SRC_PKG_DIR", file=sys.stderr)
        return 2
    src = Path(argv[1]).resolve()
    if not src.is_dir():
        print(f"check_installed_package: not a directory: {src}", file=sys.stderr)
        return 2

    import pixelgen

    installed = Path(pixelgen.__file__).resolve().parent
    if installed == src or src in installed.parents:
        print(
            "check_installed_package: pixelgen imported from the source tree (editable install?)",
            file=sys.stderr,
        )
        return 1

    count = 0
    missing = []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if not path.is_file() or "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        count += 1
        if not (installed / rel).is_file():
            missing.append(rel.as_posix())
    for rel in missing:
        print(f"check_installed_package: missing from installed package: {rel}", file=sys.stderr)
    if missing:
        return 1
    print(f"check_installed_package: OK {count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
