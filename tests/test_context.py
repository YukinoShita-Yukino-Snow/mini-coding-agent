import json

import pytest

from mini_agent.context import ContextManager


def _assistant_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"app.py"}'},
            }
        ],
    }


def test_context_keeps_system_task_and_latest_tool_round() -> None:
    context = ContextManager("system", "user task", max_chars=1_200)
    for index in range(4):
        call_id = f"call-{index}"
        context.append_assistant(_assistant_call(call_id))
        context.append_tool(call_id, "x" * 900)

    messages = context.messages_for_model()

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "user task"}
    assert messages[-2]["tool_calls"][0]["id"] == "call-3"
    assert messages[-1]["tool_call_id"] == "call-3"
    assert len(json.dumps(messages, ensure_ascii=False)) <= 1_200


def test_raw_messages_are_not_modified_by_compaction() -> None:
    context = ContextManager("system", "task", max_chars=1_200)
    context.append_assistant(_assistant_call("call-1"))
    context.append_tool("call-1", "x" * 2_000)

    context.messages_for_model()

    assert len(context.raw_messages[-1]["content"]) == 2_000


def test_context_restores_messages_and_appends_recovery_note() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
        _assistant_call("call-1"),
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]

    context = ContextManager.from_messages(messages)
    context.append_user("重新检查文件后继续")

    assert context.raw_messages[:-1] == messages
    assert context.raw_messages[-1] == {
        "role": "user",
        "content": "重新检查文件后继续",
    }


def test_context_rejects_unpaired_tool_call() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        _assistant_call("call-1"),
    ]

    with pytest.raises(ValueError, match="没有结果"):
        ContextManager.from_messages(messages)
