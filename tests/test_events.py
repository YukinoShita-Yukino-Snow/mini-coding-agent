import json
import os
from pathlib import Path

import pytest

from mini_agent.events import ConsoleReporter, JsonlRunLogger


def test_jsonl_logger_writes_structured_events(tmp_path: Path) -> None:
    logger = JsonlRunLogger(tmp_path)

    logger("run_started", {"task": "test"})
    logger("run_finished", {"stop_reason": "completed"})

    records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["run_started", "run_finished"]
    assert logger.path.parent == tmp_path / ".mini-agent" / "runs"


def test_console_reporter_does_not_print_assistant_reasoning(capsys) -> None:
    ConsoleReporter()("assistant_text", {"content": "internal"})

    assert capsys.readouterr().out == ""


@pytest.mark.skipif(os.name == "nt", reason="Windows 默认环境通常禁止创建符号链接")
def test_logger_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / ".mini-agent").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="超出工作区"):
        JsonlRunLogger(workspace)
