import json
import shutil
from pathlib import Path

import pytest

from mini_agent.checkpoint import CheckpointError, CheckpointStore


def _messages(task: str = "task") -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": task},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"app.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]


def _state(status: str = "max_steps") -> dict:
    return {
        "status": status,
        "steps": 2,
        "tool_calls": 1,
        "successful_tool_calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "fix tests")

    store.save(_messages("fix tests"), _state())
    record = CheckpointStore.load_latest(tmp_path)

    assert record.checkpoint_id == store.checkpoint_id
    assert record.task == "fix tests"
    assert record.status == "max_steps"
    assert record.messages == _messages("fix tests")
    assert record.state["successful_tool_calls"] == 1
    assert not store.path.with_suffix(".tmp").exists()


def test_checkpoint_rejects_incomplete_tool_round(tmp_path: Path) -> None:
    messages = _messages()[:-1]

    with pytest.raises(ValueError, match="没有结果"):
        CheckpointStore(tmp_path, "task").save(messages, _state())


def test_checkpoint_rejects_corrupted_json(tmp_path: Path) -> None:
    directory = tmp_path / ".mini-agent" / "checkpoints"
    directory.mkdir(parents=True)
    (directory / "checkpoint-broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(CheckpointError, match="无法读取"):
        CheckpointStore.load_latest(tmp_path)


def test_checkpoint_rejects_different_workspace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    store = CheckpointStore(source, "task")
    store.save(_messages(), _state())
    target_directory = target / ".mini-agent" / "checkpoints"
    target_directory.mkdir(parents=True)
    shutil.copy2(store.path, target_directory / store.path.name)

    with pytest.raises(CheckpointError, match="工作区不匹配"):
        CheckpointStore.load_latest(target)


def test_checkpoint_rejects_inconsistent_status(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "task")
    store.save(_messages(), _state())
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    raw["status"] = "completed"
    store.path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(CheckpointError, match="前后不一致"):
        CheckpointStore.load_latest(tmp_path)


def test_checkpoint_rejects_task_message_mismatch(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "task")
    store.save(_messages(), _state())
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    raw["task"] = "different task"
    store.path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(CheckpointError, match="初始用户消息不一致"):
        CheckpointStore.load_latest(tmp_path)
