from pathlib import Path

import pytest

from mini_agent.safety import SafetyError, Workspace


def test_workspace_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = Workspace(root)

    with pytest.raises(SafetyError, match="超出工作区"):
        workspace.resolve("../secret.txt")


def test_workspace_accepts_paths_inside_root(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    assert workspace.resolve("src/example.py") == (tmp_path / "src" / "example.py").resolve()


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", ".git/config", ".idea/workspace.xml", ".mini-agent/runs/a"],
)
def test_workspace_rejects_secret_and_internal_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(SafetyError):
        Workspace(tmp_path).resolve(path)


def test_workspace_allows_public_env_example(tmp_path: Path) -> None:
    assert Workspace(tmp_path).resolve(".env.example") == (tmp_path / ".env.example").resolve()

