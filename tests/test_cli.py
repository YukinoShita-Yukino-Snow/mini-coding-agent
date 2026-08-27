import pytest

from mini_agent.cli import build_parser, main


def test_parser_reads_task_and_workspace() -> None:
    args = build_parser().parse_args(["fix the tests", "--workspace", "example"])

    assert args.task == "fix the tests"
    assert args.workspace == "example"


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert "mini-agent 0.1.0" in capsys.readouterr().out


def test_main_reports_missing_environment(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    exit_code = main(["inspect the project", "--workspace", str(tmp_path)])

    assert exit_code == 2
    assert "AGENT_API_KEY" in capsys.readouterr().err

