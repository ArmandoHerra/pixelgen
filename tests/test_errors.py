from pixelgen import errors
from pixelgen.errors import PixelgenError, SpecError

NAMES = [
    "PaletteError", "PresetError", "SpecError", "MissingAPIKeyError", "LLMError",
    "LLMOutputError", "GridValidationError", "BackendUnavailableError",
    "OutputExistsError", "NoPreviousResultError",
]


def test_hierarchy():
    assert issubclass(PixelgenError, Exception)
    classes = [getattr(errors, n) for n in NAMES]
    for cls in classes:
        assert issubclass(cls, PixelgenError)
    assert issubclass(errors.LLMOutputError, errors.LLMError)


def test_default_messages_are_distinct_single_lines():
    msgs = [str(getattr(errors, n)()) for n in NAMES] + [str(PixelgenError())]
    for m in msgs:
        assert m and "\n" not in m
    assert len(set(msgs)) == len(msgs)


def test_message_is_collapsed_and_custom_preserved():
    assert str(SpecError("a\nb   c")) == "a b c"
    assert str(SpecError("custom text")) == "custom text"
    assert SpecError("custom text").message == "custom text"
