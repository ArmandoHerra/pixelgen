"""Runtime configuration helpers."""

import os
from pathlib import Path


def pixelgen_home() -> Path:
    """Return the pixelgen home directory without creating it.

    ``PIXELGEN_HOME`` is read at call time; the default is ``~/.pixelgen``.
    """
    value = os.environ.get("PIXELGEN_HOME")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".pixelgen"
