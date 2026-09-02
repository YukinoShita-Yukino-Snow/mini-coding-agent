"""Agent 对话检查点的原子保存与显式恢复。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .context import validate_message_history

SCHEMA_VERSION = 1
MAX_CHECKPOINT_BYTES = 10_000_000
CHECKPOINT_STATUSES = {"running", "completed", "max_steps", "tool_error_limit"}


class CheckpointError(ValueError):
    """检查点不存在、损坏或与当前工作区不兼容。"""


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    path: Path
    workspace: Path
    task: str
    status: str
    messages: list[dict[str, Any]]
    state: dict[str, Any]
    parent_checkpoint: str | None


class CheckpointStore:
    """为一次运行维护一个反复覆盖的最新安全检查点。"""

    def __init__(
        self,
        workspace: str | Path,
        task: str,
        *,
        parent_checkpoint: str | None = None,
    ) -> None:
        if not task.strip():
            raise CheckpointError("检查点任务不能为空")
        self.workspace = Path(workspace).resolve()
        self.directory = _checkpoint_directory(self.workspace, create=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.checkpoint_id = f"checkpoint-{timestamp}-{uuid4().hex[:8]}"
        self.path = self.directory / f"{self.checkpoint_id}.json"
        self.task = task.strip()
        self.parent_checkpoint = parent_checkpoint
        self._created_at = datetime.now(timezone.utc).isoformat()

    def save(self, messages: list[dict[str, Any]], state: dict[str, Any]) -> None:
        safe_messages = validate_message_history(messages)
        safe_state = _validate_state(state)
        status = str(state.get("status", ""))
        if status not in CHECKPOINT_STATUSES:
            raise CheckpointError(f"未知检查点状态：{status}")

        record = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self._created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(self.workspace),
            "task": self.task,
            "status": status,
            "parent_checkpoint": self.parent_checkpoint,
            "messages": safe_messages,
            "state": safe_state,
        }
        temporary = self.path.with_suffix(".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load_latest(cls, workspace: str | Path) -> CheckpointRecord:
        root = Path(workspace).resolve()
        directory = _checkpoint_directory(root, create=False)
        if not directory.is_dir():
            raise CheckpointError("当前工作区没有可恢复的检查点")
        candidates = sorted(directory.glob("checkpoint-*.json"))
        if not candidates:
            raise CheckpointError("当前工作区没有可恢复的检查点")
        return _load_checkpoint(candidates[-1], root)


def _checkpoint_directory(workspace: Path, *, create: bool) -> Path:
    directory = (workspace / ".mini-agent" / "checkpoints").resolve()
    if workspace not in directory.parents:
        raise CheckpointError("检查点目录超出工作区")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _load_checkpoint(path: Path, workspace: Path) -> CheckpointRecord:
    try:
        if path.is_symlink():
            raise CheckpointError("拒绝加载符号链接检查点")
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("检查点文件过大，拒绝加载")
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CheckpointError(f"检查点无法读取：{exc}") from exc
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"检查点无法读取：{exc}") from exc
    if not isinstance(raw, dict):
        raise CheckpointError("检查点根节点必须是对象")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError("检查点版本不受支持")

    checkpoint_id = raw.get("checkpoint_id")
    task = raw.get("task")
    status = raw.get("status")
    parent = raw.get("parent_checkpoint")
    if not isinstance(checkpoint_id, str) or checkpoint_id != path.stem:
        raise CheckpointError("检查点标识无效")
    if not isinstance(task, str) or not task.strip():
        raise CheckpointError("检查点任务无效")
    if status not in CHECKPOINT_STATUSES:
        raise CheckpointError("检查点状态无效")
    if parent is not None and not isinstance(parent, str):
        raise CheckpointError("父检查点标识无效")

    stored_workspace = raw.get("workspace")
    if not isinstance(stored_workspace, str):
        raise CheckpointError("检查点工作区无效")
    if Path(stored_workspace).resolve() != workspace:
        raise CheckpointError("检查点与当前工作区不匹配")

    messages = validate_message_history(raw.get("messages"))
    if messages[1]["content"].strip() != task.strip():
        raise CheckpointError("检查点任务与初始用户消息不一致")
    state = _validate_state(raw.get("state"))
    if state["status"] != status:
        raise CheckpointError("检查点状态前后不一致")
    return CheckpointRecord(
        checkpoint_id=checkpoint_id,
        path=path,
        workspace=workspace,
        task=task,
        status=status,
        messages=messages,
        state=state,
        parent_checkpoint=parent,
    )


def _validate_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError("检查点运行状态必须是对象")
    result: dict[str, Any] = {}
    for name in (
        "steps",
        "tool_calls",
        "successful_tool_calls",
        "prompt_tokens",
        "completion_tokens",
    ):
        item = value.get(name)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise CheckpointError(f"检查点字段 {name} 必须是非负整数")
        result[name] = item
    result["status"] = str(value.get("status", ""))
    return result
