"""Reply builders for the OpenRouter tests (PXG-42). Built at test time; no binary blob.

Canonical import (``tests/`` is not a package)::

    from openrouter_fixture import images_reply, png_b64, sprite_on_backdrop
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

COST_SENTINEL = 0.0123456
TOKEN_SENTINEL = 98765


def sprite_on_backdrop(seed: int, side: int = 1024) -> np.ndarray:
    """Seeded RGB sprite (a noisy red disc) on a flat backdrop."""
    rng = np.random.default_rng(seed)
    img = np.empty((side, side, 3), dtype=np.uint8)
    img[:] = (200, 220, 240)
    yy, xx = np.mgrid[0:side, 0:side]
    inside = (yy - side // 2) ** 2 + (xx - side // 2) ** 2 < (side // 3) ** 2
    colour = np.array([220, 30, 30], dtype=np.int64)
    noise = rng.integers(-20, 21, size=(side, side, 3))
    img[inside] = np.clip(colour + noise[inside], 0, 255).astype(np.uint8)
    return img


def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def png_b64(img: np.ndarray) -> str:
    return base64.b64encode(png_bytes(img)).decode("ascii")


def jpeg_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def images_reply(b64: str, **over) -> dict:
    reply = {
        "created": 1790000000,
        "data": [{"b64_json": b64, "media_type": "image/png"}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": TOKEN_SENTINEL,
            "total_tokens": TOKEN_SENTINEL + 11,
            "cost": COST_SENTINEL,
        },
    }
    reply.update(over)
    return reply
