"""OpenRouter image backend (PXG-42): one Images API call per ``generate``.

The request goes through the injected ``Transport`` (never a socket of its own). The reply's
``data[0].b64_json`` is decoded as a PNG, centre-cropped to the target aspect and returned as a
raster (PXG-35); keying, block-downscale and quantizing happen downstream. ``usage`` in the
reply (cost and token fields) is never read. The key value is only ever placed in the
Authorization header.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from pixelgen import providers
from pixelgen.backends.base import GenerationResult, GenerationUsage, Generator, RefineResult
from pixelgen.errors import BackendUnavailableError, PixelgenError
from pixelgen.providers import TransportError

log = logging.getLogger(__name__)

REQUEST_SIZE = "1024x1024"
REQUEST_QUALITY = "low"
MAX_IMAGE_SIDE = 4096
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MODEL_ID_RE = re.compile(
    r"[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,127}(:[a-z0-9._-]{1,32})?", re.ASCII
)


class ModelIdError(PixelgenError):
    default_message = "PIXELGEN_OPENROUTER_MODEL is not an OpenRouter model id."


class OpenRouterResponseError(TransportError):
    default_message = "the response from openrouter.ai could not be used"


def _host() -> str:
    return urlsplit(providers.OPENROUTER.base_url or "").hostname or "openrouter.ai"


def build_prompt(spec: Any) -> str:
    text = (
        f"Pixel art sprite of {spec.subject}, {spec.view} view, centred and filling most of "
        "the frame, on a plain flat single-colour background, no text, no border."
    )
    if spec.notes:
        text += f" Style notes: {spec.notes}."
    return text


def centre_crop(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Crop (never pad) to the ``w:h`` aspect around the centre."""
    big_h, big_w = img.shape[0], img.shape[1]
    if big_w * h >= big_h * w:
        ch = big_h
        cw = max(1, round(big_h * w / h))
    else:
        cw = big_w
        ch = max(1, round(big_w * h / w))
    x0 = (big_w - cw) // 2
    y0 = (big_h - ch) // 2
    return np.ascontiguousarray(img[y0 : y0 + ch, x0 : x0 + cw])


def decode_image(reply: Mapping[str, Any]) -> np.ndarray:
    host = _host()
    failure = _decode_failure(reply, host)
    if isinstance(failure, str):
        raise OpenRouterResponseError(failure) from None
    return failure


def _decode_failure(reply: Mapping[str, Any], host: str) -> str | np.ndarray:
    """The refusal message, or the decoded array. Raising is left to the caller so no
    ``__context__`` carries library text."""
    data = reply.get("data")
    if not isinstance(data, list) or not data:
        return f"the response from {host} had no image data"
    first = data[0]
    b64 = first.get("b64_json") if isinstance(first, dict) else None
    if not isinstance(b64, str):
        return f"the response from {host} had no b64_json image"
    raw = None
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        pass
    if raw is None:
        return f"the image from {host} was not valid base64"
    if not raw.startswith(PNG_SIGNATURE):
        return f"the image from {host} was not a PNG"
    undecodable = f"the image from {host} could not be decoded"
    arr: np.ndarray | None = None
    too_big: tuple[int, int] | None = None
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format != "PNG":
                return undecodable
            width, height = img.size
            if max(width, height) > MAX_IMAGE_SIDE:
                too_big = (width, height)
            else:
                img.load()
                has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
                arr = np.asarray(img.convert("RGBA" if has_alpha else "RGB"), dtype=np.uint8)
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return undecodable
    if too_big is not None:
        return (
            f"the image from {host} is {too_big[0]}x{too_big[1]}; "
            f"pixelgen accepts at most {MAX_IMAGE_SIDE} pixels per side"
        )
    if arr is None:
        return undecodable
    return arr


class OpenRouterGenerator(Generator):
    def __init__(self, llm: Any = None, transport: Any = None, provider: Any = None) -> None:
        self._llm = llm
        self._transport = transport
        self._provider = provider

    @classmethod
    def check_env(cls, environ: Mapping[str, str] | None = None) -> None:
        model = providers.resolve_model(providers.OPENROUTER, environ)
        if model is None or not MODEL_ID_RE.fullmatch(model):
            raise ModelIdError(
                "PIXELGEN_OPENROUTER_MODEL is not an OpenRouter model id "
                "(expected <author>/<model>, for example openai/gpt-image-2)"
            )

    def generate(self, spec: Any, palette: np.ndarray) -> GenerationResult:
        provider = self._provider or providers.OPENROUTER
        if self._transport is None:
            raise BackendUnavailableError("the 'openrouter' backend was built without a transport")
        self.check_env()
        model = providers.resolve_model(provider)
        key = providers.resolve_key(provider)
        reply = self._transport.post_json(
            f"{provider.base_url}/images",
            headers={"Authorization": f"Bearer {key}"},
            body={
                "model": model,
                "prompt": build_prompt(spec),
                "size": REQUEST_SIZE,
                "quality": REQUEST_QUALITY,
                "output_format": "png",
                "n": 1,
            },
            timeout=providers.DEFAULT_TIMEOUT_SECONDS,
        )
        img = centre_crop(decode_image(reply), spec.width, spec.height)
        return GenerationResult(
            image=img, generated_size=(img.shape[1], img.shape[0]), usage=GenerationUsage(calls=1)
        )

    def refine(
        self,
        previous_spec: Any,
        previous_grid: list[list[int]],
        palette: np.ndarray,
        instruction: str,
    ) -> RefineResult:
        raise BackendUnavailableError(
            "the 'openrouter' backend does not refine; refine edits the saved grid "
            "through the spec LLM"
        )
