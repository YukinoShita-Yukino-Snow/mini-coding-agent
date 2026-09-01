"""不经过 Shell 的本地命令执行工具。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from mini_agent.safety import SafetyError, Workspace, validate_command

MAX_OUTPUT_CHARS = 12_000
MAX_STDIN_CHARS = 100_000
_SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


def run_command(
    workspace: Workspace,
    command: Sequence[str],
    timeout_sec: int = 30,
    stdin: str | None = None,
) -> dict:
    if not 1 <= timeout_sec <= 120:
        raise ValueError("timeout_sec 必须在 1 到 120 之间")
    if stdin is not None and not isinstance(stdin, str):
        raise ValueError("stdin 必须是字符串")
    if stdin is not None and len(stdin) > MAX_STDIN_CHARS:
        raise ValueError(f"stdin 不能超过 {MAX_STDIN_CHARS} 个字符")

    arguments = validate_command(command)
    arguments[0] = _resolve_executable(workspace, arguments[0])

    try:
        completed = subprocess.run(
            arguments,
            cwd=workspace.root,
            input=stdin,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_sec,
            shell=False,
            check=False,
            env=_sanitized_environment(os.environ),
        )
    except FileNotFoundError as exc:
        raise ValueError(f"找不到可执行程序：{command[0]}") from exc
    except PermissionError as exc:
        raise ValueError(f"没有权限执行程序：{command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_output(exc.stdout)
        stderr = _coerce_output(exc.stderr)
        return {
            "command": list(command),
            "timed_out": True,
            "timeout_sec": timeout_sec,
            "stdout": _truncate(stdout)[0],
            "stderr": _truncate(stderr)[0],
        }

    stdout, stdout_truncated = _truncate(completed.stdout)
    stderr, stderr_truncated = _truncate(completed.stderr)
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "timed_out": False,
        "stdout": stdout,
        "stderr": stderr,
        "output_truncated": stdout_truncated or stderr_truncated,
    }


def _resolve_executable(workspace: Workspace, raw: str) -> str:
    """解析工作区程序；普通名称仍交由系统 PATH 查找。"""

    raw_path = Path(raw)
    if raw_path.is_absolute():
        resolved = workspace.resolve(raw, must_exist=True)
        if not resolved.is_file():
            raise SafetyError(f"可执行路径不是文件：{raw}")
        return str(resolved)

    has_separator = "/" in raw or "\\" in raw
    candidates = [workspace.root / raw_path]
    if os.name == "nt" and raw_path.suffix.casefold() != ".exe":
        candidates.append(workspace.root / f"{raw}.exe")

    for candidate in candidates:
        if candidate.is_file():
            return str(workspace.resolve(candidate, must_exist=True))

    if has_separator:
        raise ValueError(f"工作区内找不到可执行程序：{raw}")
    return raw


def _sanitized_environment(source: Mapping[str, str]) -> dict[str, str]:
    """避免把模型 API 凭据继承给目标项目的子进程。"""

    return {
        name: value
        for name, value in source.items()
        if not any(marker in name.upper() for marker in _SECRET_MARKERS)
    }


def _coerce_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    return output.decode(errors="replace") if isinstance(output, bytes) else output


def _truncate(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_OUTPUT_CHARS:
        return value, False
    marker = "\n...[中间输出已截断]...\n"
    available = MAX_OUTPUT_CHARS - len(marker)
    head_chars = available // 2
    tail_chars = available - head_chars
    return value[:head_chars] + marker + value[-tail_chars:], True
