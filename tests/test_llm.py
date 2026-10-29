"""PXG-06: the JSON-only Anthropic wrapper. No network, no real client, no httpx."""

import logging
from types import SimpleNamespace

import anthropic
import pytest
from fakes import FakeCall, FakeLLM, FakeReply

from pixelgen import llm
from pixelgen.errors import LLMError, LLMOutputError, MissingAPIKeyError, PixelgenError
from pixelgen.llm import (
    DEFAULT_MODEL,
    JSON_ONLY_INSTRUCTION,
    AnthropicLLM,
    LLMClient,
    LLMResult,
    Usage,
    extract_json,
    resolve_model,
)

SENTINEL = "sk-ant-PXG06-SENTINEL-never-print"


def fake_sdk_error(cls, msg="boom", status=None):
    def __init__(self, msg="boom", status=None):
        Exception.__init__(self, msg)
        if status is not None:
            self.status_code = status

    return type("Fake" + cls.__name__, (cls,), {"__init__": __init__})(msg, status)


def response(text, stop_reason="end_turn", usage=(123, 456)):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
        stop_reason=stop_reason,
    )


class FakeClient:
    """Stands in for anthropic.Anthropic; outcomes are responses or exceptions."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(item, BaseException):
            raise item
        return item


class Recorder:
    constructions = []

    def __init__(self, **kwargs):
        type(self).constructions.append(kwargs)
        self.messages = SimpleNamespace(create=lambda **kw: response('{"ok": 1}'))


@pytest.fixture
def recorder(monkeypatch):
    Recorder.constructions = []
    monkeypatch.setattr(anthropic, "Anthropic", Recorder)
    return Recorder


def make(*outcomes, model="m"):
    delays = []
    client = FakeClient(*outcomes)
    return AnthropicLLM(client, model, sleep=delays.append), client, delays


# AC1
def test_clients_satisfy_protocol():
    assert isinstance(AnthropicLLM(FakeClient(), "m"), LLMClient)
    assert isinstance(FakeLLM(), LLMClient)
    result = LLMResult(data={}, usage=Usage(1, 2), model="m", attempts=1)
    assert (result.data, result.usage.input_tokens, result.usage.output_tokens) == ({}, 1, 2)
    assert (result.model, result.attempts) == ("m", 1)


# AC2
def test_model_defaults_to_sonnet_5_5():
    assert resolve_model({}) == "claude-sonnet-5-5"
    assert resolve_model({"PIXELGEN_MODEL": "  "}) == "claude-sonnet-5-5"


def test_pixelgen_model_overrides_default(recorder):
    env = {"ANTHROPIC_API_KEY": SENTINEL, "PIXELGEN_MODEL": "claude-override-test-model"}
    assert resolve_model(env) == "claude-override-test-model"
    adapter = AnthropicLLM.from_env(env)
    client = FakeClient(response('{"a": 1}'))
    adapter._client = client
    result = adapter.complete_json("s", "u")
    assert client.calls[0]["model"] == "claude-override-test-model"
    assert result.model == "claude-override-test-model"


# AC3
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key_raises_before_construction(recorder, monkeypatch, value):
    env = {} if value is None else {"ANTHROPIC_API_KEY": value}
    with pytest.raises(MissingAPIKeyError) as info:
        AnthropicLLM.from_env(env)
    assert recorder.constructions == []
    message = str(info.value)
    assert "\n" not in message
    assert "ANTHROPIC_API_KEY is not set" in message
    assert "export ANTHROPIC_API_KEY=" in message
    # the real os.environ path
    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    with pytest.raises(MissingAPIKeyError):
        AnthropicLLM.from_env()
    assert recorder.constructions == []


# AC4
def test_from_env_builds_client_with_sdk_retries_off(recorder):
    AnthropicLLM.from_env({"ANTHROPIC_API_KEY": SENTINEL})
    assert recorder.constructions == [{"api_key": SENTINEL, "max_retries": 0, "timeout": 120.0}]


def test_from_env_is_blocked_by_conftest_guard():
    with pytest.raises(BaseException) as info:  # noqa: B017 - the guard raises pytest's Failed
        AnthropicLLM.from_env({"ANTHROPIC_API_KEY": SENTINEL})
    assert not isinstance(info.value, PixelgenError)
    assert isinstance(info.value, pytest.fail.Exception)


# AC5
def test_request_shape():
    adapter, client, _ = make(response('{"a": 1}'), model="the-model")
    adapter.complete_json("SYS", "USER", max_tokens=77)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert set(call) == {"model", "max_tokens", "system", "messages"}
    assert call["model"] == "the-model"
    assert call["max_tokens"] == 77
    assert call["system"].startswith("SYS")
    assert call["system"].endswith(JSON_ONLY_INSTRUCTION)
    assert call["messages"] == [{"role": "user", "content": "USER"}]


# AC6
ACCEPTED = [
    ('{"a": 1}', {"a": 1}),
    ('  \n{"a": 1}\n  ', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('Here you go:\n```json\n{"a": 1}\n```\nHope it helps.', {"a": 1}),
]


@pytest.mark.parametrize(("text", "expected"), ACCEPTED)
def test_extract_json_accepts(text, expected):
    assert extract_json(text) == expected
    adapter, _, _ = make(response(text))
    assert adapter.complete_json("s", "u").data == expected


# AC7
REJECTED = [
    "",
    "[1, 2]",
    'Sure! {"a": 1}',
    '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```',
    "```json\n{not json}\n```",
]


@pytest.mark.parametrize("text", REJECTED)
def test_extract_json_rejects(text):
    with pytest.raises(LLMOutputError, match="was not a single JSON object"):
        extract_json(text)


@pytest.mark.parametrize(
    ("stop", "needle"), [("max_tokens", "was truncated"), ("refusal", "refused")]
)
def test_truncated_and_refused_responses(stop, needle):
    adapter, _, _ = make(response('{"a": 1}', stop_reason=stop))
    with pytest.raises(LLMOutputError) as info:
        adapter.complete_json("s", "u")
    assert needle in str(info.value)
    assert info.value.usage == Usage(123, 456)


def test_output_error_is_not_transport_retried():
    adapter, client, delays = make(response("no json here"))
    with pytest.raises(LLMOutputError) as info:
        adapter.complete_json("s", "u")
    assert len(client.calls) == 1
    assert delays == []
    assert info.value.usage == Usage(123, 456)
    assert info.value.raw_text == "no json here"


# AC8
def test_usage_is_returned():
    adapter, _, _ = make(response('{"a": 1}'))
    result = adapter.complete_json("s", "u")
    assert result.usage == Usage(123, 456)
    assert result.attempts == 1
    assert FakeLLM(FakeReply({"a": 1}, Usage(7, 9))).complete_json("s", "u").usage == Usage(7, 9)


# AC9
TRANSIENT = [
    pytest.param(anthropic.RateLimitError, None, id="429"),
    pytest.param(anthropic.APIStatusError, 529, id="529"),
    pytest.param(anthropic.InternalServerError, 500, id="500"),
    pytest.param(anthropic.APIConnectionError, None, id="connection"),
    pytest.param(anthropic.APITimeoutError, None, id="timeout"),
]


@pytest.mark.parametrize(("cls", "status"), TRANSIENT)
def test_transient_errors_retry_then_succeed(cls, status):
    err = fake_sdk_error(cls, status=status)
    adapter, client, delays = make(err, err, response('{"a": 1}'))
    result = adapter.complete_json("s", "u")
    assert len(client.calls) == 3
    assert result.attempts == 3
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize(("cls", "status"), TRANSIENT)
def test_transient_errors_exhaust_budget(cls, status):
    adapter, client, delays = make(fake_sdk_error(cls, status=status))
    with pytest.raises(LLMError, match="gave up after 3 attempts"):
        adapter.complete_json("s", "u")
    assert len(client.calls) == 3
    assert delays == [0.5, 1.0]


# AC10
@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (anthropic.BadRequestError, None),
        (anthropic.AuthenticationError, None),
        (anthropic.PermissionDeniedError, None),
        (anthropic.NotFoundError, None),
        (anthropic.APIStatusError, 422),
    ],
)
def test_non_retryable_errors_fail_fast(cls, status):
    adapter, client, delays = make(fake_sdk_error(cls, status=status))
    with pytest.raises(LLMError):
        adapter.complete_json("s", "u")
    assert len(client.calls) == 1
    assert delays == []


# AC11
MAPPING = [
    (anthropic.AuthenticationError, None, "Anthropic rejected the key in ANTHROPIC_API_KEY"),
    (anthropic.PermissionDeniedError, None, "Your Anthropic key may not use model 'm'"),
    (anthropic.NotFoundError, None, "Model 'm' was not found; check PIXELGEN_MODEL."),
    (anthropic.RateLimitError, None, "Anthropic rate limit hit; gave up after 3 attempts."),
    (anthropic.APITimeoutError, None, "The Anthropic API timed out; gave up after 3 attempts."),
    (anthropic.APIConnectionError, None, "Could not reach the Anthropic API; gave up after 3"),
    (anthropic.APIStatusError, 529, "The Anthropic API is unavailable (HTTP 529); gave up after 3"),
    (anthropic.APIStatusError, 503, "The Anthropic API is unavailable (HTTP 503)"),
    (anthropic.APIStatusError, 504, "The Anthropic API is unavailable (HTTP 504)"),
    (anthropic.APIStatusError, 408, "The Anthropic API is unavailable (HTTP 408)"),
    (anthropic.APIStatusError, 409, "The Anthropic API is unavailable (HTTP 409)"),
    (anthropic.InternalServerError, 500, "The Anthropic API is unavailable (HTTP 500)"),
    (anthropic.BadRequestError, None, "Anthropic rejected the request (HTTP 400)."),
    (anthropic.APIStatusError, 413, "Anthropic rejected the request (HTTP 413)."),
    (anthropic.APIStatusError, 422, "Anthropic rejected the request (HTTP 422)."),
    (anthropic.APIError, None, "Unexpected Anthropic SDK error (FakeAPIError)."),
]


@pytest.mark.parametrize(("cls", "status", "needle"), MAPPING)
def test_sdk_errors_are_mapped(cls, status, needle):
    adapter, _, _ = make(fake_sdk_error(cls, status=status))
    with pytest.raises(LLMError) as info:
        adapter.complete_json("s", "u")
    err = info.value
    assert type(err).__module__.startswith("pixelgen")
    assert needle in str(err)
    assert "\n" not in str(err)
    assert err.__cause__ is None
    assert err.__suppress_context__ is True


def test_unknown_anthropic_error_is_mapped():
    class Weird(anthropic.AnthropicError):
        pass

    adapter, _, _ = make(Weird("x"))
    with pytest.raises(LLMError, match="Unexpected Anthropic SDK error"):
        adapter.complete_json("s", "u")


def test_default_model_404_names_the_model(recorder):
    adapter = AnthropicLLM.from_env({"ANTHROPIC_API_KEY": SENTINEL})
    client = FakeClient(fake_sdk_error(anthropic.NotFoundError))
    adapter._client = client
    with pytest.raises(LLMError) as info:
        adapter.complete_json("s", "u")
    assert client.calls[0]["model"] == DEFAULT_MODEL
    assert str(info.value) == "Model 'claude-sonnet-5-5' was not found; check PIXELGEN_MODEL."


# AC12
def test_key_never_appears_in_errors_or_logs(recorder, caplog):
    caplog.set_level(logging.DEBUG, logger="pixelgen")
    adapter = AnthropicLLM.from_env({"ANTHROPIC_API_KEY": SENTINEL}, sleep=lambda _s: None)
    leaky = f"key {SENTINEL} rejected"
    scenarios = [
        [fake_sdk_error(anthropic.AuthenticationError, msg=leaky)],
        [fake_sdk_error(anthropic.RateLimitError, msg=leaky)],
        [response("not json")],
        [response('{"ok": 1}')],
    ]
    seen = []
    for outcomes in scenarios:
        adapter._client = FakeClient(*outcomes)
        try:
            adapter.complete_json("s", "u")
        except PixelgenError as err:
            seen.append(f"{err!s} {err!r}")
    assert len(seen) == 3
    for text in (*seen, caplog.text, repr(adapter)):
        assert SENTINEL not in text
    assert caplog.text  # logging was actually exercised


# AC13
def test_retry_budgets_are_documented():
    assert llm.TRANSPORT_MAX_RETRIES == 2
    for needle in ("TRANSPORT_MAX_RETRIES", "validation", "PXG-08", "9 requests"):
        assert needle in llm.__doc__


# AC14
def test_fake_llm_returns_script_in_order_as_copies():
    payload = {"a": [1]}
    fake = FakeLLM(payload, payload, {"b": 2})
    first = fake.complete_json("s1", "u1", max_tokens=5)
    first.data["a"].append(99)
    second = fake.complete_json("s2", "u2")
    assert second.data == {"a": [1]}
    assert fake.complete_json("s3", "u3").data == {"b": 2}
    assert payload == {"a": [1]}
    assert (second.model, second.attempts, second.usage) == ("fake-model", 1, Usage(10, 20))
    assert fake.call_count == 3
    assert fake.calls[0] == FakeCall("s1", "u1", 5)
    assert fake.last_call.user == "u3"


def test_fake_llm_raises_scripted_exception_and_records_call():
    fake = FakeLLM(LLMError("x"))
    with pytest.raises(LLMError, match="x"):
        fake.complete_json("s", "u")
    assert fake.call_count == 1
    assert fake.last_call == FakeCall("s", "u", llm.DEFAULT_MAX_TOKENS)


def test_fake_llm_malformed_string_raises_output_error():
    fake = FakeLLM("not json")
    with pytest.raises(LLMOutputError) as info:
        fake.complete_json("s", "u")
    assert info.value.usage == Usage(10, 20)
    assert FakeLLM('```json\n{"a": 1}\n```').complete_json("s", "u").data == {"a": 1}


def test_fake_llm_callable_item_receives_call():
    fake = FakeLLM(lambda call: {"echo": call.user})
    assert fake.complete_json("s", "hello").data == {"echo": "hello"}


def test_fake_llm_queue_and_exhaustion():
    fake = FakeLLM().queue({"a": 1})
    fake.complete_json("s", "u")
    with pytest.raises(AssertionError, match="no scripted response for call #2"):
        fake.complete_json("s", "u")
    with pytest.raises(AssertionError):
        FakeLLM().last_call  # noqa: B018
