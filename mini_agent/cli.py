"""Mini Coding Agent 命令行界面。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .agent import CodingAgent
from .config import ConfigError, Settings
from .context import ContextLimitError
from .events import ConsoleReporter, JsonlRunLogger, combine_sinks
from .model_client import ModelClientError, OpenAIChatClient
from .safety import SafetyError
from .tools import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="在指定工作区中运行轻量级编程智能体。",
    )
    parser.add_argument("task", help="交给 Agent 的编程任务。")
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
        help="不生成本地 .mini-agent/runs/*.jsonl 运行记录。",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        workspace = Path(args.workspace).expanduser().resolve()
        registry = ToolRegistry(str(workspace))
        client = OpenAIChatClient(settings)

        sinks = [ConsoleReporter()]
        logger = None
        if not args.no_log:
            logger = JsonlRunLogger(workspace)
            sinks.append(logger)

        max_steps = settings.max_steps if args.max_steps is None else args.max_steps
        agent = CodingAgent(
            client,
            registry,
            max_steps=max_steps,
            max_context_chars=settings.max_context_chars,
            event_sink=combine_sinks(*sinks),
        )
        result = agent.run(args.task)
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
    return 0 if result.stop_reason == "completed" else 1

