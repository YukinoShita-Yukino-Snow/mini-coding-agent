from pathlib import Path

from mini_agent.agent import CodingAgent
from mini_agent.model_client import ModelReply, ToolCallRequest
from mini_agent.tools import ToolRegistry


class ScriptedClient:
    def __init__(self, replies: list[ModelReply]) -> None:
        self.replies = replies
        self.requests: list[list[dict]] = []

    def complete(self, messages, tools) -> ModelReply:
        self.requests.append(list(messages))
        if len(self.replies) == 1:
            return self.replies[0]
        return self.replies.pop(0)


def _tool_reply(call_id: str, name: str, arguments: str) -> ModelReply:
    call = ToolCallRequest(id=call_id, name=name, arguments=arguments)
    return ModelReply(
        content="",
        tool_calls=(call,),
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        finish_reason="tool_calls",
        usage={"prompt_tokens": 2, "completion_tokens": 1},
    )


def _final_reply(text: str = "完成") -> ModelReply:
    return ModelReply(
        content=text,
        tool_calls=(),
        assistant_message={"role": "assistant", "content": text},
        finish_reason="stop",
        usage={"prompt_tokens": 3, "completion_tokens": 2},
    )


def test_agent_executes_tool_then_returns_final_text(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            _tool_reply("call-1", "write_file", '{"path":"answer.txt","content":"42"}'),
            _final_reply("已创建并验证 answer.txt。"),
        ]
    )
    events: list[tuple[str, dict]] = []
    agent = CodingAgent(
        client,
        ToolRegistry(str(tmp_path)),
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    result = agent.run("创建 answer.txt")

    assert result.stop_reason == "completed"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert result.successful_tool_calls == 1
    assert result.prompt_tokens == 5
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "42"
    assert client.requests[1][-1]["role"] == "tool"
    assert any(event == "tool_result" for event, _ in events)


def test_agent_stops_after_repeated_tool_errors_with_progress_summary(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            _tool_reply("call-1", "list_files", "{}"),
            _tool_reply("call-2", "missing", "{}"),
            _tool_reply("call-3", "missing", "{}"),
            _tool_reply("call-4", "missing", "{}"),
        ]
    )
    result = CodingAgent(client, ToolRegistry(str(tmp_path))).run("使用不存在的工具")

    assert result.stop_reason == "tool_error_limit"
    assert result.tool_calls == 4
    assert result.successful_tool_calls == 1
    assert "1 次工具调用成功" in result.final_text
    assert "未知工具" in result.final_text


def test_agent_stops_at_max_steps(tmp_path: Path) -> None:
    client = ScriptedClient([_tool_reply("call-1", "list_files", "{}")])
    result = CodingAgent(client, ToolRegistry(str(tmp_path)), max_steps=2).run("持续检查")

    assert result.stop_reason == "max_steps"
    assert result.steps == 2

