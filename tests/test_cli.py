import pytest

from mini_agent.checkpoint import CheckpointStore
from mini_agent.cli import build_parser, main
from mini_agent.model_client import ModelReply


def test_parser_reads_task_and_workspace() -> None:
    args = build_parser().parse_args(["fix the tests", "--workspace", "example"])

    assert args.task == "fix the tests"
    assert args.workspace == "example"


def test_parser_accepts_resume_without_new_task() -> None:
    args = build_parser().parse_args(
        ["--workspace", "example", "--resume", "latest", "--max-steps", "10"]
    )

    assert args.task is None
    assert args.resume == "latest"
    assert args.max_steps == 10


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


def test_main_requires_task_without_resume(capsys, tmp_path) -> None:
    exit_code = main(["--workspace", str(tmp_path)])

    assert exit_code == 2
    assert "必须提供任务" in capsys.readouterr().err


def test_main_rejects_resume_with_no_log(capsys, tmp_path) -> None:
    exit_code = main(
        ["--workspace", str(tmp_path), "--resume", "latest", "--no-log"]
    )

    assert exit_code == 2
    assert "不能与 --no-log" in capsys.readouterr().err


def test_main_resumes_latest_checkpoint(monkeypatch, capsys, tmp_path) -> None:
    original = CheckpointStore(tmp_path, "original task")
    original.save(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "original task"},
        ],
        {
            "status": "max_steps",
            "steps": 2,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "prompt_tokens": 2,
            "completion_tokens": 1,
        },
    )

    class FakeClient:
        requests: list[list[dict]] = []

        def __init__(self, _settings) -> None:
            pass

        def complete(self, messages, _tools) -> ModelReply:
            self.requests.append(list(messages))
            return ModelReply(
                content="恢复后完成",
                tool_calls=(),
                assistant_message={"role": "assistant", "content": "恢复后完成"},
                finish_reason="stop",
                usage={"prompt_tokens": 3, "completion_tokens": 2},
            )

    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setattr("mini_agent.cli.OpenAIChatClient", FakeClient)

    exit_code = main(
        ["--workspace", str(tmp_path), "--resume", "latest", "--max-steps", "2"]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[恢复]" in output
    assert "恢复后完成" in output
    assert FakeClient.requests[0][-1]["role"] == "user"
    latest = CheckpointStore.load_latest(tmp_path)
    assert latest.status == "completed"
    assert latest.parent_checkpoint == original.checkpoint_id


def test_main_rejects_completed_checkpoint(monkeypatch, capsys, tmp_path) -> None:
    store = CheckpointStore(tmp_path, "completed task")
    store.save(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "completed task"},
            {"role": "assistant", "content": "done"},
        ],
        {
            "status": "completed",
            "steps": 1,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
        },
    )
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")

    exit_code = main(["--workspace", str(tmp_path), "--resume", "latest"])

    assert exit_code == 2
    assert "已经完成" in capsys.readouterr().err
