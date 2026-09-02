from pathlib import Path

from mini_agent.agent import CodingAgent
from mini_agent.checkpoint import CheckpointStore
from mini_agent.context import ContextManager
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


def test_agent_executes_multiple_tool_calls_from_one_reply(tmp_path: Path) -> None:
    calls = (
        ToolCallRequest(
            id="call-1",
            name="write_file",
            arguments='{"path":"first.txt","content":"one"}',
        ),
        ToolCallRequest(
            id="call-2",
            name="write_file",
            arguments='{"path":"second.txt","content":"two"}',
        ),
    )
    tool_reply = ModelReply(
        content="",
        tool_calls=calls,
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in calls
            ],
        },
        finish_reason="tool_calls",
        usage={"prompt_tokens": 2, "completion_tokens": 1},
    )
    client = ScriptedClient([tool_reply, _final_reply("两个文件均已创建。")])

    result = CodingAgent(client, ToolRegistry(str(tmp_path))).run("创建两个文件")

    assert result.stop_reason == "completed"
    assert result.steps == 2
    assert result.tool_calls == 2
    assert result.successful_tool_calls == 2
    assert (tmp_path / "first.txt").read_text(encoding="utf-8") == "one"
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "two"
    assert [message["role"] for message in client.requests[1][-3:]] == [
        "assistant",
        "tool",
        "tool",
    ]
    assert [message["tool_call_id"] for message in client.requests[1][-2:]] == [
        "call-1",
        "call-2",
    ]


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


def test_agent_resumes_from_max_steps_checkpoint(tmp_path: Path) -> None:
    first_store = CheckpointStore(tmp_path, "创建 answer.txt")
    first_client = ScriptedClient(
        [_tool_reply("call-1", "write_file", '{"path":"answer.txt","content":"42"}')]
    )
    first_result = CodingAgent(
        first_client,
        ToolRegistry(str(tmp_path)),
        max_steps=1,
        checkpoint_sink=first_store.save,
    ).run("创建 answer.txt")

    assert first_result.stop_reason == "max_steps"
    first_record = CheckpointStore.load_latest(tmp_path)
    assert first_record.status == "max_steps"
    assert [message["role"] for message in first_record.messages[-2:]] == [
        "assistant",
        "tool",
    ]

    context = ContextManager.from_messages(first_record.messages)
    context.append_user("检查现有文件并继续")
    second_store = CheckpointStore(
        tmp_path,
        first_record.task,
        parent_checkpoint=first_record.checkpoint_id,
    )
    second_client = ScriptedClient([_final_reply("已有文件正确，任务完成。")])
    second_result = CodingAgent(
        second_client,
        ToolRegistry(str(tmp_path)),
        checkpoint_sink=second_store.save,
    ).run(first_record.task, context=context)

    assert second_result.stop_reason == "completed"
    assert second_client.requests[0][-1]["role"] == "user"
    assert second_client.requests[0][-1]["content"] == "检查现有文件并继续"
    second_record = CheckpointStore.load_latest(tmp_path)
    assert second_record.status == "completed"
    assert second_record.parent_checkpoint == first_record.checkpoint_id


def test_agent_saves_tool_error_limit_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "调用错误工具")
    client = ScriptedClient(
        [
            _tool_reply("call-1", "missing", "{}"),
            _tool_reply("call-2", "missing", "{}"),
            _tool_reply("call-3", "missing", "{}"),
        ]
    )

    result = CodingAgent(
        client,
        ToolRegistry(str(tmp_path)),
        checkpoint_sink=store.save,
    ).run("调用错误工具")

    assert result.stop_reason == "tool_error_limit"
    record = CheckpointStore.load_latest(tmp_path)
    assert record.status == "tool_error_limit"
    assert record.state["tool_calls"] == 3


def test_agent_does_not_checkpoint_partial_multi_tool_round(tmp_path: Path) -> None:
    calls = (
        ToolCallRequest(id="call-1", name="missing", arguments="{}"),
        ToolCallRequest(id="call-2", name="missing", arguments="{}"),
    )
    reply = ModelReply(
        content="",
        tool_calls=calls,
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in calls
            ],
        },
        finish_reason="tool_calls",
        usage={"prompt_tokens": 2, "completion_tokens": 1},
    )
    store = CheckpointStore(tmp_path, "调用两个错误工具")

    result = CodingAgent(
        ScriptedClient([reply]),
        ToolRegistry(str(tmp_path)),
        max_consecutive_tool_errors=1,
        checkpoint_sink=store.save,
    ).run("调用两个错误工具")

    assert result.stop_reason == "tool_error_limit"
    record = CheckpointStore.load_latest(tmp_path)
    assert record.status == "tool_error_limit"
    assert [message["role"] for message in record.messages] == ["system", "user"]
