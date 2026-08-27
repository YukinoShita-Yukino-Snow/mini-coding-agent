"""限制在工作区内的 UTF-8 文件工具。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from mini_agent.safety import Workspace

MAX_FILE_BYTES = 256_000
MAX_READ_LINES = 400
SKIPPED_DIRECTORIES = {
    ".git",
    ".idea",
    ".mini-agent",
    ".pytest_cache",
    "__pycache__",
}


def _is_visible_file(name: str) -> bool:
    lowered = name.casefold()
    return not (
        lowered == ".env"
        or (lowered.startswith(".env.") and lowered != ".env.example")
    )


def _is_visible_directory(name: str) -> bool:
    lowered = name.casefold()
    return lowered not in SKIPPED_DIRECTORIES and not lowered.endswith(".egg-info")


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"不是文件：{path.name}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"文件超过 {MAX_FILE_BYTES} 字节限制")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("文件不是有效的 UTF-8 文本") from exc


def list_files(workspace: Workspace, path: str = ".", max_depth: int = 3) -> dict:
    if not 1 <= max_depth <= 6:
        raise ValueError("max_depth 必须在 1 到 6 之间")
    start = workspace.resolve(path, must_exist=True)
    if not start.is_dir():
        raise ValueError(f"不是目录：{path}")

    entries: list[str] = []
    for current, dirs, files in os.walk(start):
        current_path = Path(current)
        depth = len(current_path.relative_to(start).parts)
        dirs[:] = sorted(name for name in dirs if _is_visible_directory(name))
        if depth >= max_depth:
            dirs[:] = []

        if current_path != start:
            entries.append(workspace.relative(current_path) + "/")
        entries.extend(
            workspace.relative(current_path / name)
            for name in sorted(files)
            if _is_visible_file(name)
        )

    return {"path": workspace.relative(start), "entries": entries, "count": len(entries)}


def read_file(
    workspace: Workspace,
    path: str,
    start_line: int = 1,
    end_line: int = 200,
) -> dict:
    if start_line < 1 or end_line < start_line:
        raise ValueError("行号范围无效")
    if end_line - start_line + 1 > MAX_READ_LINES:
        raise ValueError(f"单次最多读取 {MAX_READ_LINES} 行")

    resolved = workspace.resolve(path, must_exist=True)
    lines = _read_text(resolved).splitlines()
    selected = lines[start_line - 1 : end_line]
    numbered = "\n".join(
        f"{number:>5}: {line}" for number, line in enumerate(selected, start=start_line)
    )
    return {
        "path": workspace.relative(resolved),
        "start_line": start_line,
        "end_line": start_line + len(selected) - 1 if selected else start_line - 1,
        "total_lines": len(lines),
        "content": numbered,
    }


def search_text(
    workspace: Workspace,
    query: str,
    path: str = ".",
    max_results: int = 50,
) -> dict:
    if not query:
        raise ValueError("query 不能为空")
    if not 1 <= max_results <= 200:
        raise ValueError("max_results 必须在 1 到 200 之间")

    start = workspace.resolve(path, must_exist=True)
    candidates = [start] if start.is_file() else _iter_text_candidates(start)
    matches: list[dict] = []
    lowered_query = query.casefold()

    for candidate in candidates:
        try:
            content = _read_text(candidate)
        except (OSError, ValueError):
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if lowered_query in line.casefold():
                matches.append(
                    {"path": workspace.relative(candidate), "line": line_number, "text": line[:500]}
                )
                if len(matches) >= max_results:
                    return {"query": query, "matches": matches, "truncated": True}

    return {"query": query, "matches": matches, "truncated": False}


def _iter_text_candidates(root: Path) -> Iterator[Path]:
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if _is_visible_directory(name))
        for name in sorted(files):
            if _is_visible_file(name):
                yield Path(current) / name


def write_file(
    workspace: Workspace,
    path: str,
    content: str,
    overwrite: bool = False,
) -> dict:
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError(f"内容超过 {MAX_FILE_BYTES} 字节限制")

    resolved = workspace.resolve(path)
    if resolved.exists() and not overwrite:
        raise ValueError("文件已存在；请明确设置 overwrite=true 或使用 replace_in_file")
    if resolved.exists() and not resolved.is_file():
        raise ValueError("目标已存在且不是文件")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(resolved, content)
    return {"path": workspace.relative(resolved), "bytes_written": len(encoded)}


def replace_in_file(
    workspace: Workspace,
    path: str,
    old_text: str,
    new_text: str,
    expected_replacements: int = 1,
) -> dict:
    if not old_text:
        raise ValueError("old_text 不能为空")
    if not 1 <= expected_replacements <= 100:
        raise ValueError("expected_replacements 必须在 1 到 100 之间")

    resolved = workspace.resolve(path, must_exist=True)
    content = _read_text(resolved)
    actual = content.count(old_text)
    if actual != expected_replacements:
        raise ValueError(
            f"期望匹配 {expected_replacements} 次，实际匹配 {actual} 次；文件未修改"
        )

    updated = content.replace(old_text, new_text)
    if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError(f"修改后文件将超过 {MAX_FILE_BYTES} 字节限制")
    _atomic_write(resolved, updated)
    return {"path": workspace.relative(resolved), "replacements": actual}


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".mini-agent.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise

