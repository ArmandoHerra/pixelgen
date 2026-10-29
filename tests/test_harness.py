import asyncio
import os

import anthropic
import pytest

MESSAGE = dict(model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}])


def test_env_is_scrubbed():
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "PIXELGEN_MODEL",
    ):
        assert key not in os.environ


def test_pixelgen_home_is_per_test_tmp(tmp_path):
    home = os.environ["PIXELGEN_HOME"]
    assert home.startswith(str(tmp_path))
    assert not os.path.exists(home)


def test_constructing_anthropic_client_fails_loudly():
    with pytest.raises(pytest.fail.Exception, match="pixelgen test guard"):
        anthropic.Anthropic(api_key="sk-ant-test")
    with pytest.raises(pytest.fail.Exception, match="pixelgen test guard"):
        anthropic.AsyncAnthropic(api_key="sk-ant-test")
    with pytest.raises(pytest.fail.Exception, match="pixelgen test guard"):
        try:
            anthropic.Anthropic(api_key="sk-ant-test")
        except Exception:
            pass


@pytest.mark.allow_anthropic_client
def test_marked_client_constructs_but_requests_are_blocked():
    client = anthropic.Anthropic(
        api_key="sk-ant-test", base_url="http://127.0.0.1:9", max_retries=0
    )
    with pytest.raises(pytest.fail.Exception, match="SDK request blocked"):
        client.messages.create(**MESSAGE)

    aclient = anthropic.AsyncAnthropic(
        api_key="sk-ant-test", base_url="http://127.0.0.1:9", max_retries=0
    )

    async def go():
        await aclient.messages.create(**MESSAGE)

    with pytest.raises(pytest.fail.Exception, match="SDK request blocked"):
        asyncio.run(go())
