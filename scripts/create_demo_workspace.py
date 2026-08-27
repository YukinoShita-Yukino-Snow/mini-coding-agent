"""创建全新的 Todo CLI 示例工作区。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence


def create_demo_workspace(target: str | Path) -> Path:
    destination = Path(target).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"目标目录已存在：{destination}")

    template = Path(__file__).resolve().parents[1] / "examples" / "todo_demo_template"
    shutil.copytree(template, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="要创建的全新示例工作区目录。")
    args = parser.parse_args(argv)

    try:
        destination = create_demo_workspace(args.target)
    except FileExistsError as exc:
        parser.error(str(exc))

    print(f"已创建示例工作区：{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

