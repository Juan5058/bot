
from __future__ import annotations

import ipaddress
import math
import os
import re
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_LOG_LEVELS = ("NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    _ENV_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset({"app_name"})

    app_name: str = "JARVIS"
    mode: Literal["simulated"] = "simulated"
    ai_enabled: bool = False
    ai_provider: Literal["simulated", "local", "remote"] = "simulated"
    remote_ai_enabled: bool = Field(default=False, strict=True)

    # Opt-in explícito para conversación local de desarrollo.
    # No sustituye las verificaciones de identidad de los modelos de producción.
    local_development_chat_enabled: bool = Field(default=False, strict=True)

    local_max_context_tokens: int = Field(default=0, strict=True, ge=0)
    local_ram_budget_mb: int | None = Field(default=None, strict=True, gt=0)
    local_vram_budget_mb: int | None = Field(default=None, strict=True, gt=0)
    local_max_system_load: float | None = Field(
        default=None,
        strict=True,
        ge=0.0,
        le=1.0,
    )
    local_max_concurrent_loads: int = Field(
        default=1,
        strict=True,
        ge=1,
        le=16,
    )
    local_max_concurrent_inferences: int = Field(
        default=1,
        strict=True,
        ge=1,
        le=16,
    )
    local_unload_after_inference: bool = Field(default=True, strict=True)
    local_ollama_base_url: str = "http://127.0.0.1:11434"
    local_ollama_model: str | None = Field(default=None, strict=True)
    local_ollama_timeout_seconds: float = Field(
        default=60.0,
        strict=True,
        gt=0,
        le=600,
    )
    log_level: str = Field(default="INFO", min_length=1)

    @field_validator(
        "ai_enabled",
        "remote_ai_enabled",
        "local_development_chat_enabled",
        mode="before",
    )
    @classmethod
    def parse_boolean_settings(cls, value: Any) -> bool:
        if type(value) is bool:
            return value
        if type(value) is str:
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise ValueError(
            "Boolean settings must use an explicit true or false value."
        )

    @field_validator("local_unload_after_inference", mode="before")
    @classmethod
    def parse_local_unload_setting(cls, value: Any) -> bool:
        if type(value) is bool:
            return value
        if type(value) is str:
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise ValueError(
            "local_unload_after_inference must use an explicit true or false value."
        )

    @field_validator(
        "local_max_context_tokens",
        "local_ram_budget_mb",
        "local_vram_budget_mb",
        "local_max_concurrent_loads",
        "local_max_concurrent_inferences",
        mode="before",
    )
    @classmethod
    def parse_local_integer_settings(cls, value: Any) -> int | None:
        if value is None or type(value) is int:
            return value
        if type(value) is str and value.isdecimal():
            return int(value)
        raise ValueError("Local model integer settings must be whole numbers.")

    @field_validator("local_max_system_load", mode="before")
    @classmethod
    def parse_local_system_load(cls, value: Any) -> float | None:
        if value is None or type(value) in {int, float}:
            return value
        if type(value) is str:
            try:
                return float(value)
            except ValueError:
                pass
        raise ValueError("Local system load limit must be a number.")

    @field_validator("local_ollama_base_url", mode="before")
    @classmethod
    def validate_local_ollama_url(cls, value: Any) -> str:
        if type(value) is not str:
            raise ValueError("Ollama URL must be a local HTTP URL.")
        try:
            parsed = urlsplit(value.strip())
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            raise ValueError("Ollama URL must be a local HTTP URL.") from None

        if (
            parsed.scheme != "http"
            or hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.netloc.endswith(":")
        ):
            raise ValueError("Ollama URL must be a local HTTP URL.")

        normalized_host = hostname.lower()
        if normalized_host != "localhost":
            try:
                address = ipaddress.ip_address(normalized_host)
            except ValueError:
                raise ValueError(
                    "Ollama URL must target a loopback address."
                ) from None
            if not address.is_loopback or "%" in normalized_host:
                raise ValueError("Ollama URL must target a loopback address.")

        netloc = parsed.netloc.lower()
        if port is not None and not netloc.rsplit(":", 1)[-1].isdigit():
            raise ValueError("Ollama URL port is invalid.")
        return f"http://{netloc}"

    @field_validator("local_ollama_model", mode="before")
    @classmethod
    def validate_local_ollama_model(cls, value: Any) -> str | None:
        if value is None or value == "":
            return None
        if (
            type(value) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", value) is None
        ):
            raise ValueError("Ollama model name is invalid.")
        return value

    @field_validator("local_ollama_timeout_seconds", mode="before")
    @classmethod
    def parse_ollama_timeout(cls, value: Any) -> float:
        if type(value) in {int, float}:
            parsed = float(value)
        elif type(value) is str:
            try:
                parsed = float(value)
            except ValueError:
                parsed = math.nan
        else:
            parsed = math.nan
        if not math.isfinite(parsed):
            raise ValueError("Ollama timeout must be a finite number.")
        return parsed

    @field_validator("mode")
    @classmethod
    def enforce_simulated_mode(cls, value: str) -> str:
        if value != "simulated":
            raise ValueError("Fase 1 only supports simulated mode.")
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, value: Any) -> str:
        if type(value) is not str:
            raise ValueError(
                "log_level must be one of the supported standard levels."
            )
        normalized = value.strip().upper()
        if normalized not in _LOG_LEVELS:
            raise ValueError(
                "log_level must be one of: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )
        return normalized

    @model_validator(mode="after")
    def validate_provider_selection(self) -> "Settings":
        if self.local_development_chat_enabled and (
            not self.ai_enabled
            or self.ai_provider != "local"
            or self.remote_ai_enabled
        ):
            raise ValueError(
                "Development chat requires enabled local AI and disabled remote AI."
            )

        if self.ai_provider == "remote":
            if not self.ai_enabled or not self.remote_ai_enabled:
                raise ValueError(
                    "The remote provider requires ai_enabled=true, "
                    "ai_provider='remote', and remote_ai_enabled=true."
                )
        elif self.remote_ai_enabled:
            raise ValueError(
                "remote_ai_enabled=true requires ai_enabled=true and "
                "ai_provider='remote'."
            )

        if not self.ai_enabled and self.ai_provider != "simulated":
            raise ValueError(
                "When ai_enabled=false, ai_provider must remain 'simulated'."
            )
        return self

    @classmethod
    def from_env(cls, env: dict[str, Any] | None = None) -> "Settings":
        source = os.environ if env is None else env
        environment_fields = {
            field_name
            for field_name in cls.model_fields
            if field_name not in cls._ENV_EXCLUDED_FIELDS
        }
        known_environment_names = {
            f"JARVIS_{field_name.upper()}" for field_name in environment_fields
        }
        unknown_environment_names = sorted(
            name
            for name in source
            if type(name) is str
            and name.startswith("JARVIS_")
            and name not in known_environment_names
        )
        if unknown_environment_names:
            raise ValueError(
                "Unknown JARVIS environment setting(s): "
                + ", ".join(unknown_environment_names)
            )

        payload = {
            field_name: source.get(
                f"JARVIS_{field_name.upper()}",
                cls.model_fields[field_name].default,
            )
            for field_name in environment_fields
        }
        return cls(**payload)
