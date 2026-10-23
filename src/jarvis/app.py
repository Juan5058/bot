from __future__ import annotations

import os

from jarvis.ai.contracts import AIProviderContract
from jarvis.ai.development_chat import DevelopmentOllamaChatProvider
from jarvis.ai.local_models import (
    LocalModelLimits,
    LocalModelManager,
    ManagedLocalAIProvider,
)
from jarvis.ai.selection import AIProviderSelector
from jarvis.ai.simulator import SimulatedAIProvider
from jarvis.config.settings import Settings
from jarvis.conversation.pipeline import ConversationPipeline
from jarvis.conversation.router import DeterministicRequestRouter
from jarvis.internet import InternetServiceSelector
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.registry import ToolRegistry, create_default_registry
from jarvis.tools.runner import SimulatedToolRunner


def _create_ai_provider(settings: Settings) -> AIProviderContract | None:
    if not settings.ai_enabled:
        if settings.ai_provider != "simulated" or settings.remote_ai_enabled:
            raise ValueError(
                "Disabled AI must use the simulated provider with remote AI disabled."
            )
        return SimulatedAIProvider()
    if settings.ai_provider == "simulated":
        return SimulatedAIProvider()
    return None


def create_application(
    settings: Settings | None = None,
    *,
    local_model_manager: LocalModelManager | None = None,
) -> ConversationPipeline:
    if settings is None:
        settings = Settings.from_env(os.environ)
    else:
        settings = Settings.model_validate(settings.model_dump())

    if (
        settings.local_development_chat_enabled
        and local_model_manager is not None
    ):
        raise ValueError(
            "Development chat cannot be combined with a production local model manager."
        )
    if local_model_manager is not None:
        if type(local_model_manager) is not LocalModelManager:
            raise TypeError("Application requires a validated local model manager.")
        if local_model_manager.allows_simulated_resources_for_tests:
            raise ValueError(
                "Application cannot use a manager configured for simulated resources."
            )
        if not settings.ai_enabled or settings.ai_provider != "local":
            raise ValueError(
                "A local model manager requires enabled local AI configuration."
            )
        if local_model_manager.limits != LocalModelLimits.from_settings(settings):
            raise ValueError(
                "Local model manager limits must match validated application settings."
            )
        if not local_model_manager.is_production_ready:
            raise ValueError(
                "Local model dependencies are not verified for production use."
            )

    ai_provider = _create_ai_provider(settings)

    if ai_provider is not None and settings.ai_provider == "simulated":
        provider_catalog: dict[str, AIProviderContract] = {
            "simulated": ai_provider
        }
    elif settings.local_development_chat_enabled:
        provider_catalog = {
            "local": DevelopmentOllamaChatProvider(settings)
        }
    elif local_model_manager is not None:
        provider_catalog = {
            "local": ManagedLocalAIProvider(local_model_manager)
        }
    else:
        provider_catalog = {}

    registry: ToolRegistry = create_default_registry()
    policy = SecurityPolicy.for_application()
    authorization_service = AuthorizationService(
        policy=policy,
        registry=registry,
    )
    audit_logger = AuditLogger()
    tool_runner = SimulatedToolRunner(
        registry=registry,
        authorization_service=authorization_service,
        mode=settings.mode,
        audit_logger=audit_logger,
    )
    return ConversationPipeline(
        ai_provider=ai_provider,
        authorizer=authorization_service,
        registry=registry,
        tool_runner=tool_runner,
        audit_logger=audit_logger,
        settings=settings,
        request_router=DeterministicRequestRouter(),
        provider_selector=AIProviderSelector(provider_catalog),
        internet_service_selector=InternetServiceSelector(),
    )


def run_text_turn(user_text: str, settings: Settings | None = None) -> dict:
    pipeline = create_application(settings)
    return pipeline.process(user_text).to_dict()
