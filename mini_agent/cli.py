"""Mini Coding Agent 命令行界面。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .agent import CodingAgent
from .checkpoint import CheckpointError, CheckpointStore
from .config import ConfigError, Settings
from .context import ContextLimitError, ContextManager
from .events import ConsoleReporter, JsonlRunLogger, combine_sinks
from .model_client import ModelClientError, OpenAIChatClient
from .safety import SafetyError
from .tools import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="在指定工作区中运行轻量级编程智能体。",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="交给 Agent 的编程任务；恢复时可作为补充说明省略。",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Agent 可以访问的项目目录（默认：当前目录）。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="仅为本次运行覆盖 AGENT_MAX_STEPS。",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="不生成本地运行日志和恢复检查点。",
    )
    parser.add_argument(
        "--resume",
        choices=("latest",),
        help="从当前工作区的最新未完成检查点显式恢复。",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.resume is None and not args.task:
            raise ValueError("未恢复运行时必须提供任务")
        if args.resume is not None and args.no_log:
            raise ValueError("--resume 不能与 --no-log 同时使用")

        settings = Settings.from_env()
        workspace = Path(args.workspace).expanduser().resolve()
        registry = ToolRegistry(str(workspace))
        client = OpenAIChatClient(settings)

        sinks = [ConsoleReporter()]
        logger = None
        if not args.no_log:
            logger = JsonlRunLogger(workspace)
            sinks.append(logger)
        event_sink = combine_sinks(*sinks)

        context = None
        resumed = None
        run_task = args.task
        if args.resume == "latest":
            resumed = CheckpointStore.load_latest(workspace)
            if resumed.status == "completed":
                raise CheckpointError("最新检查点中的任务已经完成，无需恢复")
            context = ContextManager.from_messages(
                resumed.messages, settings.max_context_chars
            )
            recovery_note = args.task or (
                "继续上一次未完成的任务。请先检查当前工作区文件和最近验证状态，"
                "不要假设已有修改完整或正确，然后继续实现并重新运行相关测试。"
            )
            context.append_user(recovery_note)
            run_task = resumed.task
            event_sink(
                "run_resumed",
                {
                    "checkpoint": resumed.checkpoint_id,
                    "previous_status": resumed.status,
                },
            )

        if run_task is None:
            raise ValueError("任务不能为空")

        checkpoint = None
        if not args.no_log:
            checkpoint = CheckpointStore(
                workspace,
                run_task,
                parent_checkpoint=(resumed.checkpoint_id if resumed else None),
            )

        max_steps = settings.max_steps if args.max_steps is None else args.max_steps
        agent = CodingAgent(
            client,
            registry,
            max_steps=max_steps,
            max_context_chars=settings.max_context_chars,
            event_sink=event_sink,
            checkpoint_sink=(checkpoint.save if checkpoint else None),
        )
        result = agent.run(run_task, context=context)
    except (ConfigError, SafetyError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except (ModelClientError, ContextLimitError) as exc:
        print(f"Agent 错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n用户已中断运行", file=sys.stderr)
        return 130

    print("\n最终回答：")
    print(result.final_text)
    if logger is not None:
        print(f"运行日志：{logger.path}")
    if checkpoint is not None:
        print(f"恢复检查点：{checkpoint.path}")
    return 0 if result.stop_reason == "completed" else 1
