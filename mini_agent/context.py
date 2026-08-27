"""Agent 对话历史和结构安全的上下文压缩。"""

from __future__ import annotations

import copy
import json
from typing import Any


class ContextLimitError(RuntimeError):
    """必要消息无法放入配置的上下文预算。"""


class ContextManager:
    """保存完整历史，并为模型生成结构合法的压缩副本。"""

    def __init__(self, system_prompt: str, task: str, max_chars: int = 120_000) -> None:
        if max_chars < 1_000:
            raise ValueError("max_chars 至少为 1000")
        self.max_chars = max_chars
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

    def append_assistant(self, message: dict[str, Any]) -> None:
        if message.get("role") != "assistant":
            raise ValueError("assistant 消息的 role 必须为 assistant")
        self._messages.append(copy.deepcopy(message))

    def append_tool(self, tool_call_id: str, content: str) -> None:
        self._messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    @property
    def raw_messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._messages)

    def messages_for_model(self) -> list[dict[str, Any]]:
        messages = copy.deepcopy(self._messages)
        tool_limit = min(8_000, max(1_000, self.max_chars // 4))
        _truncate_tool_messages(messages, tool_limit)
        removed_rounds = _remove_old_rounds_until_fit(messages, self.max_chars)

        if _message_chars(messages) > self.max_chars:
            emergency_limit = min(1_000, max(200, self.max_chars // 3))
            _truncate_tool_messages(messages, emergency_limit)
        if _message_chars(messages) > self.max_chars:
            raise ContextLimitError("系统提示、用户任务和最新观察超过上下文预算")

        if removed_rounds:
            note = (
                f"\n\n上下文说明：为满足本地预算，已移除 {removed_rounds} 个较早的完整工具轮次；"
                "如需细节，请重新读取相关文件。"
            )
            messages[0]["content"] += note
            if _message_chars(messages) > self.max_chars:
                messages[0]["content"] = messages[0]["content"][: -len(note)]

        return messages


def _truncate_tool_messages(messages: list[dict[str, Any]], limit: int) -> None:
    marker = "\n...[较早工具输出已由上下文管理器截断]"
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content", ""))
        if len(content) > limit:
            message["content"] = content[: limit - len(marker)] + marker


def _remove_old_rounds_until_fit(messages: list[dict[str, Any]], max_chars: int) -> int:
    removed = 0
    while _message_chars(messages) > max_chars:
        rounds = _tool_round_ranges(messages)
        if len(rounds) <= 1:
            break
        start, end = rounds[0]
        del messages[start:end]
        removed += 1
    return removed


def _tool_round_ranges(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    rounds: list[tuple[int, int]] = []
    index = 2
    while index < len(messages):
        if messages[index].get("role") != "assistant":
            index += 1
            continue
        start = index
        index += 1
        while index < len(messages) and messages[index].get("role") == "tool":
            index += 1
        if index > start + 1:
            rounds.append((start, index))
    return rounds


def _message_chars(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False))

