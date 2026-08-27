from pathlib import Path

import pytest

from mini_agent.safety import Workspace
from mini_agent.tools.filesystem import (
    list_files,
    read_file,
    replace_in_file,
    search_text,
    write_file,
)


def test_write_read_search_and_replace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    written = write_file(workspace, "src/app.py", "alpha\nbeta\n")
    read = read_file(workspace, "src/app.py", start_line=2, end_line=2)
    searched = search_text(workspace, "BETA")
    replaced = replace_in_file(workspace, "src/app.py", "beta", "gamma")

    assert written["path"] == "src/app.py"
    assert read["content"].endswith("beta")
    assert searched["matches"][0]["line"] == 2
    assert replaced["replacements"] == 1
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "alpha\ngamma\n"


def test_replace_requires_exact_occurrence_count(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    target = tmp_path / "values.txt"
    target.write_text("same same", encoding="utf-8")

    with pytest.raises(ValueError, match="实际匹配 2 次"):
        replace_in_file(workspace, "values.txt", "same", "new")

    assert target.read_text(encoding="utf-8") == "same same"


def test_list_files_skips_internal_and_secret_files(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    (tmp_path / ".env.local").write_text("API_KEY=secret", encoding="utf-8")
    (tmp_path / ".env.example").write_text("API_KEY=example", encoding="utf-8")
    (tmp_path / "visible.py").write_text("", encoding="utf-8")

    result = list_files(workspace)

    assert "visible.py" in result["entries"]
    assert ".env.example" in result["entries"]
    assert ".env.local" not in result["entries"]
    assert all(not entry.startswith(".git") for entry in result["entries"])

