from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    app_name: str = "JARVIS"
    mode: Literal["simulated"] = "simulated"
    ai_enabled: bool = False
    log_level: str = Field(default="INFO", min_length=1)

    @field_validator("mode")
    @classmethod
    def enforce_simulated_mode(cls, value: str) -> str:
        if value != "simulated":
            raise ValueError("Fase 1 only supports simulated mode.")
        return value

    @classmethod
    def from_env(cls, env: dict[str, Any] | None = None) -> "Settings":
        source = os.environ if env is None else env
        payload: dict[str, Any] = {}
        payload["mode"] = source.get("JARVIS_MODE", "simulated")
        payload["ai_enabled"] = source.get("JARVIS_AI_ENABLED", "false").lower() in {"1", "true", "yes"}
        payload["log_level"] = source.get("JARVIS_LOG_LEVEL", "INFO")
        return cls(**payload)
