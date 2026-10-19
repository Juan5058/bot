import pytest
from pydantic import ValidationError

from jarvis.ai.simulator import SimulatedAIProvider
from jarvis.app import create_application
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


@pytest.mark.parametrize(
    "level",
    ("NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
)
def test_supported_log_levels_are_accepted(level: str) -> None:
    assert Settings(log_level=level).log_level == level


def test_log_level_is_normalized() -> None:
    assert Settings(log_level="  dEbUg  ").log_level == "DEBUG"


def test_unknown_log_level_is_rejected_without_echoing_input() -> None:
    marker = "PRIVATE_INVALID_LOG_LEVEL"
    with pytest.raises(ValidationError) as error:
        Settings(log_level=marker)

    assert "log_level must be one of:" in str(error.value)
    assert marker not in str(error.value)


def test_unknown_direct_settings_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings(remote_ai_enabeld=True)


def test_from_env_rejects_unknown_jarvis_variables_and_ignores_other_variables() -> None:
    settings = Settings.from_env(
        {
            "PATH": "ignored-system-value",
            "OTHER_APPLICATION_SETTING": "ignored",
            "JARVIS_AI_ENABLED": "false",
        }
    )
    assert settings.ai_enabled is False
    assert settings.remote_ai_enabled is False

    with pytest.raises(ValueError, match="Unknown JARVIS environment setting"):
        Settings.from_env({"JARVIS_UNKNOWN_SETTING": "value"})


def test_misspelled_remote_setting_never_enables_remote_ai() -> None:
    with pytest.raises(ValueError, match="Unknown JARVIS environment setting"):
        Settings.from_env(
            {
                "JARVIS_AI_ENABLED": "true",
                "JARVIS_AI_PROVIDER": "remote",
                "JARVIS_REMOTE_AI_ENABELD": "true",
            }
        )


def test_invalid_environment_settings_fail_before_application_components(
    monkeypatch,
) -> None:
    import jarvis.app as app_module

    monkeypatch.setenv("JARVIS_LOG_LEVEL", "INVALID_LEVEL")

    def unexpected_provider_creation(settings):
        raise AssertionError("Application components were constructed before validation.")

    monkeypatch.setattr(app_module, "_create_ai_provider", unexpected_provider_creation)

    with pytest.raises(ValidationError, match="log_level must be one of:"):
        create_application()


def test_simulated_provider_is_the_default_and_legacy_flag_does_not_select_ai() -> None:
    settings = Settings()
    app = create_application(settings)
    env_settings = Settings.from_env({})

    assert env_settings.mode == "simulated"
    assert env_settings.ai_enabled is False
    assert settings.ai_provider == "simulated"
    assert settings.remote_ai_enabled is False
    assert env_settings.ai_provider == "simulated"
    assert env_settings.remote_ai_enabled is False
    assert env_settings.log_level == "INFO"
    assert type(app.ai_provider) is SimulatedAIProvider
    assert app.authorizer.policy.allowed_permissions == frozenset({"tool.echo_message"})

    legacy_enabled_app = create_application(Settings(ai_enabled=True))
    assert type(legacy_enabled_app.ai_provider) is SimulatedAIProvider
    assert (
        legacy_enabled_app.authorizer.policy.allowed_permissions
        == app.authorizer.policy.allowed_permissions
    )
    assert (
        legacy_enabled_app.authorizer.policy.allowed_tool_permissions
        == app.authorizer.policy.allowed_tool_permissions
    )


def test_from_env_accepts_mode_and_normalizes_log_level_without_enabling_remote_ai() -> None:
    settings = Settings.from_env(
        {
            "JARVIS_MODE": "simulated",
            "JARVIS_LOG_LEVEL": "debug",
        }
    )

    assert settings.mode == "simulated"
    assert settings.log_level == "DEBUG"
    assert settings.ai_enabled is False
    assert settings.ai_provider == "simulated"
    assert settings.remote_ai_enabled is False


@pytest.mark.parametrize(
    "settings",
    (
        {"ai_provider": "unknown"},
        {"ai_provider": "remote"},
        {"remote_ai_enabled": True},
        {"ai_provider": "simulated", "remote_ai_enabled": True},
        {"ai_enabled": False, "ai_provider": "local"},
        {"ai_enabled": False, "ai_provider": "remote", "remote_ai_enabled": True},
        {"ai_provider": "local", "remote_ai_enabled": True},
        {"ai_provider": "remote", "remote_ai_enabled": False},
        {"ai_enabled": True, "ai_provider": "remote", "remote_ai_enabled": False},
    ),
)
def test_invalid_or_incomplete_provider_selection_is_rejected(settings) -> None:
    with pytest.raises(ValidationError):
        Settings(**settings)


def test_remote_provider_requires_explicit_enable_but_remains_unavailable() -> None:
    settings = Settings(
        ai_enabled=True,
        ai_provider="remote",
        remote_ai_enabled=True,
    )
    app = create_application(settings)

    result = app.process("pregunta: ¿qué significa?")

    assert app.ai_provider is None
    assert result.success is False
    assert result.error == "Informative processing is unavailable."


def test_local_provider_selection_is_unavailable_without_fallback() -> None:
    settings = Settings(ai_enabled=True, ai_provider="local")
    app = create_application(settings)

    result = app.process("pregunta: ¿qué significa?")

    assert app.ai_provider is None
    assert result.success is False
    assert result.error == "Informative processing is unavailable."


@pytest.mark.parametrize(
    ("provider", "enabled", "remote_enabled"),
    (("local", True, False), ("remote", True, True)),
)
def test_unimplemented_provider_does_not_fall_back_to_simulator(
    monkeypatch,
    provider,
    enabled,
    remote_enabled,
) -> None:
    import jarvis.app as app_module

    def unexpected_simulator_fallback():
        raise AssertionError("Unimplemented selection fell back to simulation.")

    monkeypatch.setattr(
        app_module,
        "SimulatedAIProvider",
        unexpected_simulator_fallback,
    )

    app = create_application(
        Settings(
            ai_enabled=enabled,
            ai_provider=provider,
            remote_ai_enabled=remote_enabled,
        )
    )
    result = app.process("pregunta: no debe haber fallback")

    assert app.ai_provider is None
    assert result.success is False
    assert result.error == "Informative processing is unavailable."


@pytest.mark.parametrize(
        ("ai_enabled", "ai_provider", "remote_ai_enabled", "valid"),
        (
            (False, "simulated", False, True),
            (True, "simulated", False, True),
            (False, "local", False, False),
            (True, "local", False, True),
            (False, "remote", False, False),
            (False, "remote", True, False),
            (True, "remote", False, False),
            (True, "remote", True, True),
            (True, "simulated", True, False),
            (False, "simulated", True, False),
        ),
)
def test_provider_flag_combinations_are_explicit_and_fail_closed(
        ai_enabled,
        ai_provider,
        remote_ai_enabled,
        valid,
) -> None:
        if not valid:
            with pytest.raises(ValidationError):
                Settings(
                    ai_enabled=ai_enabled,
                    ai_provider=ai_provider,
                    remote_ai_enabled=remote_ai_enabled,
                )
            return

        settings = Settings(
            ai_enabled=ai_enabled,
            ai_provider=ai_provider,
            remote_ai_enabled=remote_ai_enabled,
        )
        if ai_provider == "local":
            app = create_application(settings)
            assert app.ai_provider is None
        elif ai_provider == "remote":
            app = create_application(settings)
            assert app.ai_provider is None
        else:
            assert type(create_application(settings).ai_provider) is SimulatedAIProvider


def test_environment_provider_flags_are_validated_fail_closed() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"JARVIS_REMOTE_AI_ENABLED": "sometimes"})

    with pytest.raises(ValidationError):
        Settings.from_env({"JARVIS_AI_PROVIDER": "unknown"})

    with pytest.raises(ValidationError):
        Settings.from_env({"JARVIS_REMOTE_AI_ENABLED": "true"})

    with pytest.raises(ValidationError):
        Settings.from_env(
            {
                "JARVIS_AI_ENABLED": "false",
                "JARVIS_AI_PROVIDER": "remote",
                "JARVIS_REMOTE_AI_ENABLED": "true",
            }
        )

    remote_settings = Settings.from_env(
        {
            "JARVIS_AI_ENABLED": "true",
            "JARVIS_AI_PROVIDER": "remote",
            "JARVIS_REMOTE_AI_ENABLED": "true",
        }
    )
    assert remote_settings.ai_enabled is True
    assert remote_settings.ai_provider == "remote"
    assert remote_settings.remote_ai_enabled is True
