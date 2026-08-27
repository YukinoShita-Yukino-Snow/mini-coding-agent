"""工具派发和结构化错误处理。"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from mini_agent.safety import SafetyError, Workspace

from . import filesystem, shell
from .definitions import TOOL_DEFINITIONS

ToolFunction = Callable[..., dict]


class ToolRegistry:
    """持有工作区，并将模型请求派发给已知本地工具。"""

    def __init__(self, workspace: str) -> None:
        self.workspace = Workspace(workspace)
        self._tools: dict[str, ToolFunction] = {
            "list_files": filesystem.list_files,
            "read_file": filesystem.read_file,
            "search_text": filesystem.search_text,
            "write_file": filesystem.write_file,
            "replace_in_file": filesystem.replace_in_file,
            "run_command": shell.run_command,
        }

    @property
    def definitions(self) -> list[dict]:
        return TOOL_DEFINITIONS

    def execute(self, name: str, arguments: dict[str, Any]) -> dict:
        function = self._tools.get(name)
        if function is None:
            return {"ok": False, "tool": name, "error": "未知工具"}
        if not isinstance(arguments, dict):
            return {"ok": False, "tool": name, "error": "arguments 必须是对象"}

        try:
            data = function(self.workspace, **arguments)
        except (SafetyError, TypeError, ValueError, OSError) as exc:
            return {"ok": False, "tool": name, "error": str(exc)}
        return {"ok": True, "tool": name, "data": data}

    def execute_json(self, name: str, raw_arguments: str) -> str:
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else "参数不是字符串"
            result = {"ok": False, "tool": name, "error": f"JSON 参数无效：{detail}"}
        else:
            result = self.execute(name, arguments)
        return json.dumps(result, ensure_ascii=False)
