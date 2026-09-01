from types import SimpleNamespace

import pytest

from mini_agent.config import Settings
from mini_agent.model_client import ModelClientError, OpenAIChatClient


class FakeCompletions:
    def __init__(self, response) -> None:
        self.response = response
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def _fake_client(response):
    completions = FakeCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _settings() -> Settings:
    return Settings(api_key="test", model="test-model", request_retries=0)


def test_client_parses_function_tool_calls() -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path":"app.py"}'),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[tool_call],
                    reasoning_content="provider reasoning",
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    fake, completions = _fake_client(response)
    settings = Settings(
        api_key="test",
        model="test-model",
        request_retries=0,
        thinking_mode="enabled",
    )

    reply = OpenAIChatClient(settings, client=fake).complete(
        [{"role": "user", "content": "inspect"}], [{"type": "function"}]
    )

    assert reply.tool_calls[0].name == "read_file"
    assert reply.assistant_message["tool_calls"][0]["id"] == "call-1"
    assert reply.assistant_message["reasoning_content"] == "provider reasoning"
    assert reply.usage["total_tokens"] == 15
    assert completions.requests[0]["tool_choice"] == "auto"
    assert completions.requests[0]["extra_body"] == {"thinking": {"type": "enabled"}}


def test_client_parses_final_text() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="完成。", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    fake, _ = _fake_client(response)

    reply = OpenAIChatClient(_settings(), client=fake).complete([], [])

    assert reply.content == "完成。"
    assert reply.tool_calls == ()


def test_client_rejects_empty_response() -> None:
    fake, _ = _fake_client(SimpleNamespace(choices=[]))

    with pytest.raises(ModelClientError, match="choices"):
        OpenAIChatClient(_settings(), client=fake).complete([], [])


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_client_rejects_abnormally_finished_text(finish_reason: str) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="未完成的回答", tool_calls=[]),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )
    fake, _ = _fake_client(response)

    with pytest.raises(ModelClientError, match=f"未正常结束：{finish_reason}"):
        OpenAIChatClient(_settings(), client=fake).complete([], [])
