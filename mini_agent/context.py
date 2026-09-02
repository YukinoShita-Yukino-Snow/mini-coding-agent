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

    @classmethod
    def from_messages(
        cls, messages: list[dict[str, Any]], max_chars: int = 120_000
    ) -> "ContextManager":
        safe_messages = validate_message_history(messages)
        context = cls(
            str(safe_messages[0]["content"]),
            str(safe_messages[1]["content"]),
            max_chars,
        )
        context._messages = safe_messages
        return context

    def append_assistant(self, message: dict[str, Any]) -> None:
        if message.get("role") != "assistant":
            raise ValueError("assistant 消息的 role 必须为 assistant")
        self._messages.append(copy.deepcopy(message))

    def append_tool(self, tool_call_id: str, content: str) -> None:
        self._messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    def append_user(self, content: str) -> None:
        if not content.strip():
            raise ValueError("user 消息不能为空")
        self._messages.append({"role": "user", "content": content.strip()})

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


def validate_message_history(value: Any) -> list[dict[str, Any]]:
    """验证可恢复消息，确保每个工具调用都有且仅有一个结果。"""
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("检查点消息至少需要 system 和 user 两项")
    messages = copy.deepcopy(value)
    if (
        not isinstance(messages[0], dict)
        or messages[0].get("role") != "system"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise ValueError("检查点第一条消息必须是 system")
    if (
        not isinstance(messages[1], dict)
        or messages[1].get("role") != "user"
        or not isinstance(messages[1].get("content"), str)
    ):
        raise ValueError("检查点第二条消息必须是 user")

    pending: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"检查点消息 {index} 必须是对象")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"检查点消息 {index} 的 role 无效")
        if index > 0 and role == "system":
            raise ValueError("检查点只能包含一条开头的 system 消息")

        if role == "assistant":
            if pending:
                raise ValueError("新的 assistant 消息前仍有工具结果缺失")
            if message.get("content") is not None and not isinstance(
                message.get("content"), str
            ):
                raise ValueError("assistant 消息内容必须是字符串或 null")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("assistant.tool_calls 必须是数组")
            if not calls and not isinstance(message.get("content"), str):
                raise ValueError("没有工具调用的 assistant 消息必须包含文本")
            ids: list[str] = []
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("工具调用类型无效")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("工具调用缺少 function")
                if not isinstance(function.get("name"), str) or not isinstance(
                    function.get("arguments"), str
                ):
                    raise ValueError("工具调用函数名称或参数无效")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("工具调用缺少有效 id")
                if call_id in ids:
                    raise ValueError("同一 assistant 消息包含重复工具调用 id")
                ids.append(call_id)
            pending = ids
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("tool 消息没有对应的工具调用")
            if not isinstance(message.get("content"), str):
                raise ValueError("tool 消息内容必须是字符串")
            pending.remove(call_id)
        elif not isinstance(message.get("content"), str):
            raise ValueError(f"{role} 消息内容必须是字符串")
        elif pending:
            raise ValueError("工具结果完整返回前不能出现其他消息")

    if pending:
        raise ValueError("检查点末尾存在没有结果的工具调用")
    return messages
