"""从环境变量读取并校验应用配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


class ConfigError(ValueError):
    """缺少必需配置或配置值非法。"""


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    base_url: str | None = None
    thinking_mode: str | None = None
    request_timeout_sec: int = 90
    request_retries: int = 2
    max_steps: int = 25
    max_context_chars: int = 120_000

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if environ is None else environ
        api_key = values.get("AGENT_API_KEY", "").strip()
        model = values.get("AGENT_MODEL", "").strip()
        base_url = values.get("AGENT_BASE_URL", "").strip() or None
        thinking_mode = values.get("AGENT_THINKING_MODE", "").strip().lower() or None

        if not api_key:
            raise ConfigError("缺少环境变量 AGENT_API_KEY")
        if not model:
            raise ConfigError("缺少环境变量 AGENT_MODEL")
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise ConfigError("AGENT_THINKING_MODE 必须为 enabled 或 disabled")

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            thinking_mode=thinking_mode,
            request_timeout_sec=_read_int(values, "AGENT_REQUEST_TIMEOUT_SEC", 90, 1, 600),
            request_retries=_read_int(values, "AGENT_REQUEST_RETRIES", 2, 0, 10),
            max_steps=_read_int(values, "AGENT_MAX_STEPS", 25, 1, 100),
            max_context_chars=_read_int(
                values, "AGENT_MAX_CONTEXT_CHARS", 120_000, 10_000, 2_000_000
            ),
        )


def _read_int(
    values: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return value

