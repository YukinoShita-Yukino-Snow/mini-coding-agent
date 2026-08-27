import json
from pathlib import Path

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

