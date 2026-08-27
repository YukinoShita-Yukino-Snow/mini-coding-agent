from pathlib import Path

import pytest

from todo_app import add_task, complete_task, format_tasks, load_tasks


def test_add_and_load_tasks(tmp_path: Path) -> None:
    database = tmp_path / "tasks.json"

    first = add_task(database, "编写代码")
    second = add_task(database, "运行测试")

    assert first["id"] == 1
    assert second["id"] == 2
    assert [task["title"] for task in load_tasks(database)] == ["编写代码", "运行测试"]


def test_complete_task(tmp_path: Path) -> None:
    database = tmp_path / "tasks.json"
    add_task(database, "编写代码")

    completed = complete_task(database, 1)

    assert completed["completed"] is True


def test_complete_missing_task_raises_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不存在"):
        complete_task(tmp_path / "tasks.json", 99)


def test_format_tasks() -> None:
    tasks = [
        {"id": 1, "title": "编写代码", "completed": False},
        {"id": 2, "title": "运行测试", "completed": True},
    ]

    assert format_tasks(tasks) == "1. [ ] 编写代码\n2. [x] 运行测试"

