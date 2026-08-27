"""使用 JSON 保存数据的简易 Todo 命令行程序。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

DEFAULT_DATABASE = Path("tasks.json")


def load_tasks(database: Path) -> list[dict]:
    if not database.exists():
        return []
    return json.loads(database.read_text(encoding="utf-8"))


def save_tasks(database: Path, tasks: list[dict]) -> None:
    database.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def add_task(database: Path, title: str) -> dict:
    tasks = load_tasks(database)
    task = {"id": len(tasks) + 1, "title": title, "completed": False}
    tasks.append(task)
    save_tasks(database, tasks)
    return task


def complete_task(database: Path, task_id: int) -> dict:
    tasks = load_tasks(database)
    for task in tasks:
        if task["id"] == task_id:
            task["completed"] = True
            save_tasks(database, tasks)
            return task
    raise ValueError(f"任务 {task_id} 不存在")


def format_tasks(tasks: list[dict]) -> str:
    if not tasks:
        return "暂无任务。"
    lines = []
    for task in tasks:
        marker = "x" if task["completed"] else " "
        lines.append(f"{task['id']}. [{marker}] {task['title']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="管理本地待办事项。")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="添加任务。")
    add_parser.add_argument("title")
    subparsers.add_parser("list", help="列出任务。")

    done_parser = subparsers.add_parser("done", help="完成任务。")
    done_parser.add_argument("task_id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "add":
        task = add_task(args.database, args.title)
        print(f"已添加任务 {task['id']}。")
    elif args.command == "list":
        print(format_tasks(load_tasks(args.database)))
    elif args.command == "done":
        task = complete_task(args.database, args.task_id)
        print(f"已完成任务 {task['id']}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

