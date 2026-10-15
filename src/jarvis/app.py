from __future__ import annotations

import os

from jarvis.ai.simulator import SimulatedAIProvider
from jarvis.config.settings import Settings
from jarvis.conversation.pipeline import ConversationPipeline
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.registry import ToolRegistry, create_default_registry
from jarvis.tools.runner import SimulatedToolRunner


def create_application(settings: Settings | None = None) -> ConversationPipeline:
    if settings is None:
        settings = Settings.from_env(os.environ)
    else:
        settings = Settings.model_validate(settings.model_dump())
    registry: ToolRegistry = create_default_registry()
    policy = SecurityPolicy.for_application()
    authorization_service = AuthorizationService(policy=policy, registry=registry)
    tool_runner = SimulatedToolRunner(registry=registry, authorization_service=authorization_service, mode=settings.mode)
    return ConversationPipeline(
        ai_provider=SimulatedAIProvider(),
        authorizer=authorization_service,
        registry=registry,
        tool_runner=tool_runner,
        audit_logger=AuditLogger(),
        settings=settings,
    )


def run_text_turn(user_text: str, settings: Settings | None = None) -> dict:
    pipeline = create_application(settings)
    return pipeline.process(user_text).to_dict()
