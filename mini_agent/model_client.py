"""OpenAI Chat Completions 兼容模型客户端。

本模块只负责模型通信和响应标准化，不执行任何本地工具。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from .config import Settings

TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class ModelClientError(RuntimeError):
    """模型请求失败或响应无法供 Agent 使用。"""


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelReply:
    content: str
    tool_calls: tuple[ToolCallRequest, ...]
    assistant_message: dict[str, Any]
    finish_reason: str | None
    usage: dict[str, int]


class ModelClient(Protocol):
    def complete(self, messages: Sequence[dict], tools: Sequence[dict]) -> ModelReply:
        """根据对话历史和工具定义返回一次模型决策。"""


class OpenAIChatClient:
    """通过 OpenAI 兼容的 Chat Completions 接口调用模型。"""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._sleep = sleep
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout_sec,
            max_retries=0,
        )

    def complete(self, messages: Sequence[dict], tools: Sequence[dict]) -> ModelReply:
        response = self._create_with_retry(messages, tools)
        if not getattr(response, "choices", None):
            raise ModelClientError("模型响应中没有 choices")

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        parsed_calls: list[ToolCallRequest] = []
        serialized_calls: list[dict[str, Any]] = []

        for call in message.tool_calls or []:
            if call.type != "function":
                raise ModelClientError(f"不支持的工具调用类型：{call.type}")
            parsed_calls.append(
                ToolCallRequest(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
            )
            serialized_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )

        if not content and not parsed_calls:
            raise ModelClientError("模型既没有返回文本，也没有返回工具调用")
        if finish_reason not in {None, "stop", "tool_calls"}:
            raise ModelClientError(f"模型响应未正常结束：{finish_reason}")

        assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is not None:
            assistant_message["reasoning_content"] = reasoning_content
        if serialized_calls:
            assistant_message["tool_calls"] = serialized_calls

        usage = getattr(response, "usage", None)
        usage_data = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        return ModelReply(
            content=content,
            tool_calls=tuple(parsed_calls),
            assistant_message=assistant_message,
            finish_reason=finish_reason,
            usage=usage_data,
        )

    def _create_with_retry(self, messages: Sequence[dict], tools: Sequence[dict]):
        attempts = self.settings.request_retries + 1
        for attempt in range(attempts):
            try:
                request: dict[str, Any] = {
                    "model": self.settings.model,
                    "messages": list(messages),
                    "tools": list(tools),
                    "tool_choice": "auto",
                }
                if self.settings.thinking_mode is not None:
                    request["extra_body"] = {
                        "thinking": {"type": self.settings.thinking_mode}
                    }
                return self._client.chat.completions.create(**request)
            except TRANSIENT_ERRORS as exc:
                if attempt == attempts - 1:
                    raise ModelClientError(
                        f"模型请求在 {attempts} 次尝试后仍然失败：{exc}"
                    ) from exc
                self._sleep(min(2**attempt, 4))
            except OpenAIError as exc:
                raise ModelClientError(f"模型请求失败：{exc}") from exc

        raise AssertionError("unreachable retry state")
