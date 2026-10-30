"""Shared local-Aseprite seam (PXG-24, X-PXG-77, OR-20).

Tests and tools that need the operator's own Aseprite build go through this module. Aseprite never
runs in CI: the PIXELGEN_ASEPRITE variable is unset there, so `require_aseprite` skips. The child
process gets a scrubbed environment (scratch HOME and XDG roots, no display) so Aseprite can never
write the operator's real configuration. Stdlib plus pytest only; no product code.
"""

import os
import subprocess
from pathlib import Path

import pytest

ASEPRITE_ENV_VAR = "PIXELGEN_ASEPRITE"
LOCAL_SKIP_LINE = (
    "SKIP local Aseprite: PIXELGEN_ASEPRITE is unset; Aseprite never runs in CI (PXG-24, OR-20)"
)


def aseprite_from_env() -> Path | None:
    """The configured Aseprite path, None when unset or empty, ValueError when unusable."""
    value = os.environ.get(ASEPRITE_ENV_VAR, "")
    if not value:
        return None
    path = Path(value)
    if not (path.is_file() and os.access(path, os.X_OK)):
        raise ValueError(f"{ASEPRITE_ENV_VAR}={value} is not an executable file")
    return path


def require_aseprite() -> Path:
    """Skip when unset; fail (never skip) when set to something unusable."""
    try:
        exe = aseprite_from_env()
    except ValueError as exc:
        pytest.fail(str(exc), pytrace=False)
    if exe is None:
        pytest.skip(LOCAL_SKIP_LINE)
    return exe


def aseprite_env(root: Path) -> dict[str, str]:
    """A copy of os.environ with HOME, PIXELGEN_HOME and XDG_* under `root` and no display."""
    env = dict(os.environ)
    dirs = {
        "HOME": root / "home",
        "PIXELGEN_HOME": root / "pixelgen-home",
        "XDG_CONFIG_HOME": root / "xdg" / "config",
        "XDG_CACHE_HOME": root / "xdg" / "cache",
        "XDG_DATA_HOME": root / "xdg" / "data",
        "XDG_STATE_HOME": root / "xdg" / "state",
    }
    for key, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    return env


def run_aseprite(
    exe: Path, args: list[str], *, root: Path, timeout: float = 120
) -> subprocess.CompletedProcess:
    """Run `exe args` with the scrubbed environment, cwd `root`; callers check the outputs."""
    return subprocess.run(
        [str(exe), *args],
        env=aseprite_env(root),
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
