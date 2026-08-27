import pytest

from mini_agent.config import ConfigError, Settings


def test_settings_load_required_and_optional_values() -> None:
    settings = Settings.from_env(
        {
            "AGENT_API_KEY": "test-key",
            "AGENT_MODEL": "test-model",
            "AGENT_BASE_URL": "https://example.invalid/v1",
            "AGENT_THINKING_MODE": "disabled",
            "AGENT_MAX_STEPS": "12",
        }
    )

    assert settings.api_key == "test-key"
    assert settings.model == "test-model"
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.thinking_mode == "disabled"
    assert settings.max_steps == 12


@pytest.mark.parametrize("missing", ["AGENT_API_KEY", "AGENT_MODEL"])
def test_settings_require_key_and_model(missing: str) -> None:
    values = {"AGENT_API_KEY": "key", "AGENT_MODEL": "model"}
    del values[missing]

    with pytest.raises(ConfigError):
        Settings.from_env(values)


def test_settings_validate_integer_ranges() -> None:
    with pytest.raises(ConfigError, match="AGENT_MAX_STEPS"):
        Settings.from_env(
            {"AGENT_API_KEY": "key", "AGENT_MODEL": "model", "AGENT_MAX_STEPS": "0"}
        )


def test_settings_validate_thinking_mode() -> None:
    with pytest.raises(ConfigError, match="AGENT_THINKING_MODE"):
        Settings.from_env(
            {
                "AGENT_API_KEY": "key",
                "AGENT_MODEL": "model",
                "AGENT_THINKING_MODE": "sometimes",
            }
        )

