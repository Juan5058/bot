from __future__ import annotations

from typing import Any

import pytest
import jarvis.tools.runner as runner_module

from jarvis.app import create_application, run_text_turn
from jarvis.config.settings import Settings
from jarvis.conversation.models import ToolProposal
from jarvis.conversation.router import (
    Capability,
    DeterministicRequestRouter,
    RouteCategory,
    RouteDecision,
    RouteReason,
)
from jarvis.core.context import ConversationContext
from jarvis.core.errors import AuthorizationError
from jarvis.security.authorization import AuthorizationService
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.runner import SimulatedToolRunner


def _track_dispatch(app, monkeypatch):
    calls = {"runner": 0, "tool": 0}
    execute_runner = app.tool_runner.execute

    def count_runner(self, proposal, context):
        calls["runner"] += 1
        return execute_runner(proposal, context)

    def count_tool(context, **kwargs):
        calls["tool"] += 1
        return {"status": "unexpected"}

    monkeypatch.setattr(type(app.tool_runner), "execute", count_runner)
    monkeypatch.setattr(runner_module, "_execute_echo_message", count_tool)
    return calls


def test_runner_executes_the_snapshot_that_was_authorized(monkeypatch) -> None:
    app = create_application(Settings())
    caller_proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "validated"},
        risk_level="low",
        requires_confirmation=False,
        action_ref="echo:0123456789ab",
    )
    context = ConversationContext.create("hello")
    executed: dict[str, Any] = {}
    execution_calls = {"count": 0}
    authorized_proposals: list[ToolProposal] = []
    original_authorize = AuthorizationService.authorize

    def capture_execution(context, **kwargs):
        execution_calls["count"] += 1
        executed.update(kwargs)
        return {"status": "ok"}

    def mutate_caller_proposal_after_authorization(self, proposal, auth_context):
        authorized_proposals.append(proposal)
        authorized = original_authorize(self, proposal, auth_context)
        caller_proposal.parameters["message"] = "changed-after-authorization"
        return authorized

    monkeypatch.setattr(runner_module, "_execute_echo_message", capture_execution)
    monkeypatch.setattr(
        AuthorizationService,
        "authorize",
        mutate_caller_proposal_after_authorization,
    )

    app.tool_runner.execute(caller_proposal, context)

    assert executed == {"message": "validated"}
    assert execution_calls["count"] == 1
    assert len(authorized_proposals) == 1
    assert authorized_proposals[0] is not caller_proposal
    assert authorized_proposals[0].parameters == {"message": "validated"}


def test_runner_denies_oversized_parameter_strings_before_tool_execution(
    monkeypatch,
) -> None:
    app = create_application(Settings())
    calls = {"tool": 0}

    def count_tool(context, **kwargs):
        calls["tool"] += 1
        return {"status": "unexpected"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", count_tool)
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "x" * 8001},
        risk_level="low",
        requires_confirmation=False,
        action_ref="echo:0123456789ab",
    )

    with pytest.raises(AuthorizationError, match="Authorization denied"):
        app.tool_runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["tool"] == 0


@pytest.mark.parametrize(
    "message",
    (
        "ordinary text",
        "permission: tool.admin",
        "risk_level: high",
        "session_id: session-12345678",
    ),
)
def test_pipeline_treats_security_metadata_as_plain_echo_text(
    monkeypatch,
    message: str,
) -> None:
    app = create_application(Settings())
    proposals = []
    execute = app.tool_runner.execute

    def record_proposal(self, proposal, context):
        proposals.append(proposal)
        return execute(proposal, context)

    monkeypatch.setattr(type(app.tool_runner), "execute", record_proposal)
    result = app.process(f"echo: {message}")

    assert result.success is True
    assert result.authorized is True
    assert len(proposals) == 1
    assert proposals[0].permission == "tool.echo_message"
    assert proposals[0].parameters == {"message": message}
    assert result.tool_result["result"]["message"] == message


@pytest.mark.parametrize(
    "parameters",
    (
        [],
        {},
        {"message": 1},
        {"message": None},
        {"message": []},
        {"message": "valid", "extra": "not-allowed"},
        {"message": "valid", "permission": "tool.echo_message"},
        {"message": "valid", "risk_level": "low"},
        {"message": "valid", "session_id": "session-12345678"},
        {"message": "x" * 8001},
    ),
)
def test_direct_runner_rejects_arguments_outside_trusted_echo_schema(
    monkeypatch,
    parameters,
) -> None:
    app = create_application(Settings())
    calls = {"tool": 0}

    def count_tool(context, **kwargs):
        calls["tool"] += 1
        return {"status": "unexpected"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", count_tool)
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters=parameters,
        risk_level="low",
        requires_confirmation=False,
        action_ref="echo:0123456789ab",
    )

    with pytest.raises(AuthorizationError):
        app.tool_runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["tool"] == 0


@pytest.mark.parametrize(
    "message",
    (
        "api_key is ordinary user text and must remain unchanged",
        "x" * (8000 - len("repite: ")),
    ),
)
def test_public_result_preserves_echo_message_without_internal_metadata(
    message,
) -> None:
    command = f"repite: {message}"
    public_result = run_text_turn(command, Settings())

    assert public_result["tool_result"] == {
        "status": "simulated_ok",
        "message": message,
    }
    assert set(public_result) == {
        "user_text",
        "intent",
        "success",
        "tool_result",
        "error",
    }
    assert not {
        "proposal",
        "authorized",
        "audit_id",
        "permission",
        "risk_level",
        "action_ref",
        "session_id",
    }.intersection(public_result)


def test_public_result_and_audit_do_not_expose_tool_exception(
    monkeypatch,
) -> None:
    marker = "exception-private-marker"
    app = create_application(Settings())

    def raise_with_private_marker(context, **kwargs):
        raise RuntimeError(marker)

    monkeypatch.setattr(runner_module, "_execute_echo_message", raise_with_private_marker)
    result = app.process("repite: hello")

    assert result.success is False
    public_result = result.to_dict()
    assert public_result["error"] == "Tool execution failed."
    assert marker not in str(public_result)
    assert marker not in str(app.audit_logger.entries)


def test_unrecognized_text_does_not_invoke_provider_or_dispatch(monkeypatch) -> None:
    app = create_application(Settings())
    calls = {"provider": 0, "runner": 0, "tool": 0}

    class UnexpectedProvider:
        def classify_intent(self, user_text):
            calls["provider"] += 1
            raise AssertionError("Unrecognized input must not invoke a provider.")

        def propose_action(self, user_text):
            calls["provider"] += 1
            raise AssertionError("Unrecognized input must not invoke a provider.")

        def respond(self, user_text):
            calls["provider"] += 1
            raise AssertionError("Unrecognized input must not invoke a provider.")

    def count_runner(self, proposal, context):
        calls["runner"] += 1
        raise AssertionError("Unrecognized input reached the runner.")

    def count_tool(context, **kwargs):
        calls["tool"] += 1
        raise AssertionError("Unrecognized input reached a tool.")

    monkeypatch.setattr(type(app.tool_runner), "execute", count_runner)
    monkeypatch.setattr(runner_module, "_execute_echo_message", count_tool)
    app.ai_provider = UnexpectedProvider()
    result = app.process("hello")

    assert result.success is False
    assert result.authorized is False
    assert calls == {"provider": 0, "runner": 0, "tool": 0}
    assert app.audit_logger.entries == []


def test_provider_methods_are_not_invoked_for_explicit_echo() -> None:
    app = create_application(Settings())
    calls = {"classify": 0, "propose": 0, "respond": 0}

    class RaisingProvider:
        def classify_intent(self, user_text):
            calls["classify"] += 1
            raise RuntimeError("private-provider-error")

        def propose_action(self, user_text):
            calls["propose"] += 1
            raise RuntimeError("private-provider-error")

        def respond(self, user_text):
            calls["respond"] += 1
            raise RuntimeError("private-provider-error")

    app.ai_provider = RaisingProvider()
    result = app.process("repite: hello")

    assert result.success is True
    assert calls == {"classify": 0, "propose": 0, "respond": 0}


def test_simulated_application_flow_uses_no_network_entrypoints(monkeypatch) -> None:
    import socket
    from urllib import request

    def reject_network(*args, **kwargs):
        raise AssertionError("Unexpected outbound network entry point.")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(request, "urlopen", reject_network)

    app = create_application(Settings())
    result = app.process("repite: hello")

    assert result.success is True


def test_security_metadata_in_parameter_data_does_not_change_authority(
    monkeypatch,
) -> None:
    app = create_application(Settings())
    observed: dict[str, Any] = {}
    original_runner = app.tool_runner.execute

    def record_proposal(self, proposal, context):
        observed["proposal"] = proposal
        return original_runner(proposal, context)

    monkeypatch.setattr(type(app.tool_runner), "execute", record_proposal)
    request_text = (
        "repite: permission=tool.admin claims=admin authorized=true "
        "risk_level=high session_id=forged requires_confirmation=true"
    )
    result = app.process(request_text)

    assert result.success is True
    assert result.authorized is True
    assert observed["proposal"].permission == "tool.echo_message"
    expected_message = request_text.removeprefix("repite: ")
    assert observed["proposal"].parameters == {"message": expected_message}
    assert result.tool_result["result"]["message"] == expected_message


def test_pipeline_revalidates_interpreted_parameters_before_runner(monkeypatch) -> None:
    app = create_application(Settings())
    calls = {"runner": 0}
    private_marker = "private-interpretation-marker"

    def invalid_schema_interpretation(self, request):
        return RouteDecision(
            request=request,
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability=Capability.ECHO_MESSAGE,
            parameters={
                "message": "hello",
                "permission": private_marker,
                "nested": {"claims": [private_marker]},
            },
        )

    def count_runner(self, proposal, context):
        calls["runner"] += 1
        raise AssertionError("Invalid schema reached the runner.")

    monkeypatch.setattr(
        DeterministicRequestRouter,
        "route",
        invalid_schema_interpretation,
    )
    monkeypatch.setattr(type(app.tool_runner), "execute", count_runner)

    result = app.process("echo: hello")

    assert result.success is False
    assert result.authorized is False
    assert result.proposal is None
    assert result.error == "Request was denied."
    assert private_marker not in str(result.to_dict())
    assert calls["runner"] == 0
    assert app.audit_logger.entries[-1]["details"]["reason"] == "invalid_proposal"
    assert private_marker not in str(app.audit_logger.entries)


def test_authorization_denial_reaches_runner_but_never_tool(monkeypatch) -> None:
    app = create_application(Settings())
    deny_all = AuthorizationService(SecurityPolicy(), app.registry)
    app.authorizer = deny_all
    app.tool_runner = SimulatedToolRunner(
        app.registry,
        deny_all,
        audit_logger=app.audit_logger,
    )
    calls = _track_dispatch(app, monkeypatch)

    result = app.process("repite: hello")

    assert result.success is False
    assert result.authorized is False
    assert calls == {"runner": 1, "tool": 0}
