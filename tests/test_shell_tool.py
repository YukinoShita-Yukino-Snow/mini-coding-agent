import os
from pathlib import Path

import pytest

from mini_agent.safety import SafetyError, Workspace
from mini_agent.tools.shell import _resolve_executable, run_command


def test_run_command_captures_output_and_stdin(tmp_path: Path) -> None:
    result = run_command(
        Workspace(tmp_path),
        ["python", "-c", "print(input().upper())"],
        stdin="hello\n",
    )

    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "HELLO"
    assert result["timed_out"] is False


def test_run_command_reports_timeout(tmp_path: Path) -> None:
    result = run_command(
        Workspace(tmp_path),
        ["python", "-c", "import time; time.sleep(2)"],
        timeout_sec=1,
    )

    assert result["timed_out"] is True


def test_run_command_removes_api_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "must-not-leak")

    result = run_command(
        Workspace(tmp_path),
        ["python", "-c", "import os; print(os.getenv('AGENT_API_KEY'))"],
    )

    assert result["stdout"].strip() == "None"


@pytest.mark.skipif(os.name != "nt", reason="Windows .exe 路径解析测试")
def test_resolve_windows_workspace_executable(tmp_path: Path) -> None:
    executable = tmp_path / "fib.exe"
    executable.touch()

    resolved = _resolve_executable(Workspace(tmp_path), "./fib")

    assert resolved == str(executable.resolve())


def test_absolute_executable_outside_workspace_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SafetyError):
        _resolve_executable(Workspace(tmp_path), str(Path(os.__file__).resolve()))
