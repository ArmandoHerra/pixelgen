"""The cycle-1 error hierarchy. Every error renders as one user-facing line."""

import re


class PixelgenError(Exception):
    """Base class; ``str(err)`` is always a single line."""

    default_message = "pixelgen failed."

    def __init__(self, message: str | None = None) -> None:
        text = message if message is not None else self.default_message
        self.message = re.sub(r"\s+", " ", text).strip()
        super().__init__(self.message)


class PaletteError(PixelgenError):
    default_message = "The palette file is missing, malformed or the palette name is unknown."


class PresetError(PixelgenError):
    default_message = "The preset file is missing, malformed or the preset name is unknown."


class SpecError(PixelgenError):
    default_message = "The art spec could not be resolved or validated."


class MissingAPIKeyError(PixelgenError):
    default_message = "ANTHROPIC_API_KEY is not set. Export your Anthropic API key and try again."


class LLMError(PixelgenError):
    default_message = "The language model request failed."


class LLMOutputError(LLMError):
    default_message = "The language model answered, but not with usable JSON."


class GridValidationError(PixelgenError):
    default_message = "The pixel grid has invalid dimensions or palette indices out of range."


class BackendUnavailableError(PixelgenError):
    default_message = "The requested generator backend is not implemented or not registered."


class OutputExistsError(PixelgenError):
    default_message = "The output file already exists; choose another path or enable overwrite."


class NoPreviousResultError(PixelgenError):
    default_message = "There is no previous result to refine; pass --from or generate first."
