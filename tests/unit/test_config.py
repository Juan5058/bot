import pytest
from pydantic import ValidationError

from jarvis.config.settings import Settings


def test_valid_settings_accept_simulated_mode() -> None:
    settings = Settings()
    assert settings.mode == "simulated"
    assert settings.ai_enabled is False


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(mode="live")


def test_missing_ai_credentials_do_not_block_startup() -> None:
    settings = Settings(ai_enabled=False)
    assert settings.mode == "simulated"
    assert settings.ai_enabled is False
