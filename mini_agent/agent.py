"""项目自行实现的编程 Agent 主循环。"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .context import ContextManager
from .model_client import ModelClient, ModelReply
from .tools import ToolRegistry

EventSink = Callable[[str, dict], None]
CheckpointSink = Callable[[list[dict[str, Any]], dict[str, Any]], None]


def _system_prompt() -> str:
    system = platform.system() or "unknown"
    return f"""你是一个在单一工作区中运行的本地编程 Agent。
当前操作系统是 {system}。修改前先使用工具检查项目，所有文件路径使用工作区相对路径。
新文件使用 write_file，精确修改优先使用 replace_in_file；修改后运行相关测试或构建。
不得访问凭据、.git、.idea、.mini-agent 或工作区外路径。
命令不经过 Shell，每个参数必须作为 command 数组中的独立元素。可通过 stdin 给交互程序输入。
在 Windows 运行工作区程序时使用 ./程序名.exe，不要使用 /workspace 等 Linux 虚构路径。
任务完成后，返回简洁的修改摘要和验证结果，不再调用工具。
"""


@dataclass(frozen=True)
class AgentResult:
    final_text: str
    stop_reason: str
    steps: int
    tool_calls: int
    successful_tool_calls: int
    prompt_tokens: int
    completion_tokens: int


class CodingAgent:
    """协调模型决策、本地工具执行和循环终止。"""

    def __init__(
        self,
        client: ModelClient,
        registry: ToolRegistry,
        *,
        max_steps: int = 25,
        max_context_chars: int = 120_000,
        max_consecutive_tool_errors: int = 3,
        event_sink: EventSink | None = None,
        checkpoint_sink: CheckpointSink | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须为正整数")
        if max_consecutive_tool_errors < 1:
            raise ValueError("max_consecutive_tool_errors 必须为正整数")
        self.client = client
        self.registry = registry
        self.max_steps = max_steps
        self.max_context_chars = max_context_chars
        self.max_consecutive_tool_errors = max_consecutive_tool_errors
        self.event_sink = event_sink or (lambda _event, _payload: None)
        self.checkpoint_sink = checkpoint_sink

    def run(self, task: str, *, context: ContextManager | None = None) -> AgentResult:
        if not task.strip():
            raise ValueError("任务不能为空")

        if context is None:
            context = ContextManager(_system_prompt(), task.strip(), self.max_context_chars)
        tool_call_count = 0
        successful_tool_calls = 0
        consecutive_errors = 0
        last_error = ""
        prompt_tokens = 0
        completion_tokens = 0

        self._emit("run_started", {"task": task.strip(), "workspace": str(self.registry.workspace.root)})
        self._save_checkpoint(
            context,
            status="running",
            steps=0,
            tool_calls=0,
            successful_tool_calls=0,
            prompt_tokens=0,
            completion_tokens=0,
        )

        for step in range(1, self.max_steps + 1):
            model_messages = context.messages_for_model()
            safe_messages_before_step = context.raw_messages
            self._emit("model_request", {"step": step, "message_count": len(model_messages)})
            reply = self.client.complete(model_messages, self.registry.definitions)
            context.append_assistant(reply.assistant_message)
            prompt_tokens += reply.usage.get("prompt_tokens", 0)
            completion_tokens += reply.usage.get("completion_tokens", 0)

            if reply.content:
                self._emit("assistant_text", {"step": step, "content": reply.content})

            if not reply.tool_calls:
                self._save_checkpoint(
                    context,
                    status="completed",
                    steps=step,
                    tool_calls=tool_call_count,
                    successful_tool_calls=successful_tool_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                result = AgentResult(
                    final_text=reply.content,
                    stop_reason="completed",
                    steps=step,
                    tool_calls=tool_call_count,
                    successful_tool_calls=successful_tool_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                self._emit("run_finished", _result_payload(result))
                return result

            for call_index, tool_call in enumerate(reply.tool_calls):
                tool_call_count += 1
                self._emit(
                    "tool_call",
                    {
                        "step": step,
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                )
                raw_result = self.registry.execute_json(tool_call.name, tool_call.arguments)
                context.append_tool(tool_call.id, raw_result)
                parsed_result = json.loads(raw_result)
                self._emit(
                    "tool_result",
                    {
                        "step": step,
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "result": parsed_result,
                    },
                )

                if parsed_result.get("ok"):
                    successful_tool_calls += 1
                    consecutive_errors = 0
                else:
                    last_error = str(parsed_result.get("error", "未知错误"))
                    consecutive_errors += 1
                    if consecutive_errors >= self.max_consecutive_tool_errors:
                        if call_index == len(reply.tool_calls) - 1:
                            self._save_checkpoint(
                                context,
                                status="tool_error_limit",
                                steps=step,
                                tool_calls=tool_call_count,
                                successful_tool_calls=successful_tool_calls,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                            )
                        else:
                            self._save_checkpoint_messages(
                                safe_messages_before_step,
                                status="tool_error_limit",
                                steps=step,
                                tool_calls=tool_call_count,
                                successful_tool_calls=successful_tool_calls,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                            )
                        result = AgentResult(
                            final_text=(
                                f"连续 {consecutive_errors} 次工具执行失败，Agent 已停止。"
                                f"本轮已有 {successful_tool_calls} 次工具调用成功，已产生的文件不会回滚。"
                                f"最近错误：{last_error}"
                            ),
                            stop_reason="tool_error_limit",
                            steps=step,
                            tool_calls=tool_call_count,
                            successful_tool_calls=successful_tool_calls,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                        self._emit("run_finished", _result_payload(result))
                        return result

            self._save_checkpoint(
                context,
                status="running",
                steps=step,
                tool_calls=tool_call_count,
                successful_tool_calls=successful_tool_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        self._save_checkpoint(
            context,
            status="max_steps",
            steps=self.max_steps,
            tool_calls=tool_call_count,
            successful_tool_calls=successful_tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        result = AgentResult(
            final_text=(
                f"达到最大模型步骤数 {self.max_steps}，Agent 已停止。"
                f"本轮已有 {successful_tool_calls} 次工具调用成功。"
            ),
            stop_reason="max_steps",
            steps=self.max_steps,
            tool_calls=tool_call_count,
            successful_tool_calls=successful_tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._emit("run_finished", _result_payload(result))
        return result

    def _emit(self, event: str, payload: dict) -> None:
        self.event_sink(event, payload)

    def _save_checkpoint(
        self,
        context: ContextManager,
        *,
        status: str,
        steps: int,
        tool_calls: int,
        successful_tool_calls: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self._save_checkpoint_messages(
            context.raw_messages,
            status=status,
            steps=steps,
            tool_calls=tool_calls,
            successful_tool_calls=successful_tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _save_checkpoint_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        status: str,
        steps: int,
        tool_calls: int,
        successful_tool_calls: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        if self.checkpoint_sink is None:
            return
        self.checkpoint_sink(
            messages,
            {
                "status": status,
                "steps": steps,
                "tool_calls": tool_calls,
                "successful_tool_calls": successful_tool_calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )


def _result_payload(result: AgentResult) -> dict:
    return {
        "final_text": result.final_text,
        "stop_reason": result.stop_reason,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "successful_tool_calls": result.successful_tool_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
