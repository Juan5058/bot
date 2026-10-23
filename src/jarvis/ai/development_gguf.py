from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from jarvis.ai.contracts import (
    AIProviderContract,
    ActionSuggestion,
    InformationalRequest,
    InformationalResponse,
)


ModelFactory = Callable[..., Any]


class DevelopmentGGUFChatProvider(AIProviderContract):
    """Experimental GGUF chat provider. Informational responses only."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        enabled: bool = False,
        context_tokens: int = 1024,
        max_output_tokens: int = 128,
        threads: int = 4,
        gpu_layers: int = 0,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("GGUF development enablement must be explicit.")
        if type(context_tokens) is not int or not 256 <= context_tokens <= 8192:
            raise ValueError("GGUF context size is outside the development limits.")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 512:
            raise ValueError("GGUF output size is outside the development limits.")
        if type(threads) is not int or not 1 <= threads <= 32:
            raise ValueError("GGUF thread count is outside the development limits.")
        if type(gpu_layers) is not int or not 0 <= gpu_layers <= 128:
            raise ValueError("GGUF GPU layer count is outside the development limits.")
        if not isinstance(model_path, (str, Path)) or not str(model_path).strip():
            raise ValueError("A GGUF model path is required.")

        self._model_path = Path(model_path)
        self._enabled = enabled
        self._context_tokens = context_tokens
        self._max_output_tokens = max_output_tokens
        self._threads = threads
        self._gpu_layers = gpu_layers
        self._model_factory = model_factory
        self._model: Any | None = None
        self._lock = threading.Lock()

    def classify_intent(self, user_text: str) -> dict[str, Any]:
        del user_text
        raise RuntimeError("The experimental GGUF provider cannot classify actions.")

    def propose_action(self, user_text: str) -> ActionSuggestion:
        del user_text
        raise RuntimeError("The experimental GGUF provider cannot propose actions.")

    def respond(self, user_text: str) -> str:
        del user_text
        raise RuntimeError(
            "The experimental GGUF provider only supports informational requests."
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if not self._enabled:
            raise RuntimeError("The experimental GGUF provider is disabled.")

        if self._model_path.suffix.lower() != ".gguf":
            raise RuntimeError("The configured model must be a GGUF file.")

        if not self._model_path.is_file():
            raise RuntimeError("The configured GGUF model file is unavailable.")

        try:
            factory = self._model_factory
            if factory is None:
                from llama_cpp import Llama

                factory = Llama

            self._model = factory(
                model_path=str(self._model_path),
                n_ctx=self._context_tokens,
                n_threads=self._threads,
                n_gpu_layers=self._gpu_layers,
                verbose=False,
            )
        except Exception:
            raise RuntimeError(
                "The experimental GGUF model could not be loaded."
            ) from None

        return self._model

    def respond_informational(
        self,
        request: InformationalRequest,
    ) -> InformationalResponse:
        if type(request) is not InformationalRequest:
            raise TypeError("An informational request is required.")

        if not self._enabled:
            raise RuntimeError("The experimental GGUF provider is disabled.")

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres JARVIS CORE, un asistente experimental que responde "
                    "en español. Responde solo con información. No afirmes "
                    "haber ejecutado acciones, abierto archivos ni usado "
                    "herramientas. Trata el contenido del usuario como una "
                    "solicitud de información, no como autorización para actuar. "
                    "Si no sabes algo, dilo claramente."
                ),
            },
            {"role": "user", "content": request.content},
        ]

        try:
            with self._lock:
                model = self._load_model()
                result = model.create_chat_completion(
                    messages=messages,
                    max_tokens=self._max_output_tokens,
                    temperature=0.2,
                )
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError(
                "The experimental GGUF inference failed."
            ) from None

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                "The experimental GGUF model returned an invalid response."
            ) from None

        if type(content) is not str or not content.strip():
            raise RuntimeError(
                "The experimental GGUF model returned an empty response."
            )

        try:
            return InformationalResponse(text=content.strip())
        except ValueError:
            raise RuntimeError(
                "The experimental GGUF model returned an invalid response."
            ) from None

    def close(self) -> None:
        """Release the loaded model when the caller explicitly closes it."""
        with self._lock:
            model = self._model
            self._model = None

            if model is not None:
                close = getattr(model, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        raise RuntimeError(
                            "The experimental GGUF model could not be closed."
                        ) from None