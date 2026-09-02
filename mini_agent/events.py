"""面向用户的终端事件和机器可读的本地运行日志。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

EventSink = Callable[[str, dict], None]


class JsonlRunLogger:
    """把每个 Agent 事件追加到一个本地 JSONL 记录。"""

    def __init__(self, workspace: str | Path) -> None:
        root = Path(workspace).resolve()
        run_directory = (root / ".mini-agent" / "runs").resolve()
        if root not in run_directory.parents:
            raise ValueError("运行日志目录超出工作区")
        run_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = run_directory / f"run-{timestamp}.jsonl"

    def __call__(self, event: str, payload: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class ConsoleReporter:
    """输出简洁进度，不展示模型隐藏推理。"""

    def __call__(self, event: str, payload: dict) -> None:
        if event == "run_started":
            print(f"[开始] 工作区：{payload['workspace']}")
        elif event == "run_resumed":
            print(
                f"[恢复] 检查点={payload['checkpoint']} "
                f"上次状态={payload['previous_status']}"
            )
        elif event == "model_request":
            print(f"[步骤 {payload['step']}] 正在请求模型决策...")
        elif event == "tool_call":
            arguments = _shorten(str(payload["arguments"]), 240)
            print(f"[工具] {payload['name']} {arguments}")
        elif event == "tool_result":
            result = payload["result"]
            status = "成功" if result.get("ok") else "错误"
            detail = result.get("error") or result.get("data")
            print(f"[结果:{status}] {_shorten(json.dumps(detail, ensure_ascii=False), 300)}")
        elif event == "run_finished":
            print(
                f"[停止:{payload['stop_reason']}] 步骤={payload['steps']} "
                f"工具调用={payload['tool_calls']} 成功={payload['successful_tool_calls']}"
            )


def combine_sinks(*sinks: EventSink) -> EventSink:
    def combined(event: str, payload: dict) -> None:
        for sink in sinks:
            sink(event, payload)

    return combined


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
