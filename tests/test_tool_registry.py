import json
from pathlib import Path

from mini_agent.tools import ToolRegistry


def test_registry_returns_structured_success(tmp_path: Path) -> None:
    registry = ToolRegistry(str(tmp_path))

    result = registry.execute("write_file", {"path": "hello.txt", "content": "hello"})

    assert result["ok"] is True
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello"


def test_registry_returns_structured_errors(tmp_path: Path) -> None:
    registry = ToolRegistry(str(tmp_path))

    unknown = registry.execute("missing_tool", {})
    invalid_json = json.loads(registry.execute_json("list_files", "not-json"))
    outside = registry.execute("read_file", {"path": "../secret.txt"})

    assert unknown["ok"] is False
    assert invalid_json["ok"] is False
    assert outside["ok"] is False

