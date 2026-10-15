from jarvis.app import create_application
from jarvis.config.settings import Settings


def test_text_flow_end_to_end() -> None:
    settings = Settings(mode="simulated", ai_enabled=False)
    app = create_application(settings)

    result = app.process("Hola JARVIS")

    assert result.success is True
    assert result.authorized is True
    assert result.tool_result is not None
    assert result.tool_result["status"] == "simulated_ok"
    assert result.audit_id is not None
