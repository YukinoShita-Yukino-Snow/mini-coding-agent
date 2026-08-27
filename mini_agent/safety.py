"""工作区安全边界。该边界用于减少误操作，并不等同于操作系统沙箱。"""

from __future__ import annotations

from pathlib import Path


class SafetyError(ValueError):
    """操作超出已配置的安全边界。"""


class Workspace:
    """将所有工具路径限制在一个明确的项目目录中。"""

    _BLOCKED_PARTS = {".git", ".idea", ".mini-agent"}

    def __init__(self, root: str | Path) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise SafetyError(f"工作区不存在或不是目录：{resolved}")
        self.root = resolved

    def resolve(self, user_path: str | Path = ".", *, must_exist: bool = False) -> Path:
        path = Path(user_path).expanduser()
        candidate = path.resolve() if path.is_absolute() else (self.root / path).resolve()

        if candidate != self.root and self.root not in candidate.parents:
            raise SafetyError(f"路径超出工作区：{user_path}")
        for part in candidate.relative_to(self.root).parts:
            lowered = part.casefold()
            if lowered in self._BLOCKED_PARTS:
                raise SafetyError(f"禁止访问内部路径：{part}")
            if lowered == ".env" or (
                lowered.startswith(".env.") and lowered != ".env.example"
            ):
                raise SafetyError(f"禁止访问环境凭据文件：{part}")
        if must_exist and not candidate.exists():
            raise SafetyError(f"路径不存在：{user_path}")
        return candidate

    def relative(self, path: str | Path) -> str:
        resolved = self.resolve(path)
        relative = resolved.relative_to(self.root).as_posix()
        return relative or "."

