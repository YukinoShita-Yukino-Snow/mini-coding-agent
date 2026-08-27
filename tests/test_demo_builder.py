from pathlib import Path

import pytest

from scripts.create_demo_workspace import create_demo_workspace


def test_demo_builder_creates_repeatable_workspace(tmp_path: Path) -> None:
    destination = tmp_path / "todo-demo"

    created = create_demo_workspace(destination)

    assert created == destination.resolve()
    assert (destination / "todo_app.py").is_file()
    assert (destination / "test_todo_app.py").is_file()


def test_demo_builder_refuses_to_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "todo-demo"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        create_demo_workspace(destination)

