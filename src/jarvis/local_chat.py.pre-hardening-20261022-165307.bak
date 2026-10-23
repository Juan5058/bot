from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "qwen3:1.7b"
API_URL = "http://127.0.0.1:11434/api/chat"
MAX_INPUT_LENGTH = 8000
MAX_HISTORY_MESSAGES = 8
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "Eres JARVIS CORE, un asistente local que responde en español. "
    "Ayuda con preguntas, explicaciones y programación. "
    "No afirmes que puedes ejecutar acciones, acceder a archivos o "
    "usar herramientas si no se te ha proporcionado esa capacidad. "
    "Si no conoces una función real de JARVIS CORE, dilo claramente."
)


def generate_response(messages: list[dict[str, str]]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "num_ctx": 4096,
            "num_predict": 384,
        },
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise RuntimeError("Ollama devolvió un estado HTTP inesperado.")
            result = json.loads(response.read(2_000_000).decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"Ollama no encontró el modelo {MODEL}. "
                f"Comprueba con: ollama list"
            ) from None
        raise RuntimeError(f"Ollama devolvió HTTP {exc.code}.") from None
    except URLError:
        raise RuntimeError(
            "No se pudo conectar con Ollama en 127.0.0.1:11434. "
            "Comprueba que Ollama esté iniciado."
        ) from None
    except (TimeoutError, OSError):
        raise RuntimeError(
            "Ollama no respondió a tiempo o la conexión falló."
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Ollama devolvió una respuesta inválida.") from None

    if type(result) is not dict or result.get("done") is not True:
        raise RuntimeError("La inferencia no terminó correctamente.")

    message = result.get("message")
    if type(message) is not dict:
        raise RuntimeError("Ollama no devolvió un mensaje válido.")

    content = message.get("content")
    if type(content) is not str or not content.strip():
        raise RuntimeError("Ollama devolvió una respuesta vacía.")

    return content.strip()


def main() -> int:
    history: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print(f"JARVIS CORE | Modelo local: {MODEL}")
    print("Escribe /salir para terminar.\n")

    while True:
        try:
            user_text = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJARVIS: Sesión finalizada.")
            return 0

        if user_text.lower() in {"/salir", "/exit", "/quit"}:
            print("JARVIS: Hasta luego.")
            return 0

        if not user_text:
            continue

        if len(user_text) > MAX_INPUT_LENGTH:
            print(f"JARVIS: El mensaje supera {MAX_INPUT_LENGTH} caracteres.\n")
            continue

        history.append({"role": "user", "content": user_text})

        try:
            answer = generate_response(history)
        except RuntimeError as exc:
            history.pop()
            print(f"JARVIS: Error: {exc}\n")
            continue

        history.append({"role": "assistant", "content": answer})
        print(f"\nJARVIS: {answer}\n")

        # Conserva el mensaje de sistema y solo los turnos recientes.
        history = [history[0], *history[-MAX_HISTORY_MESSAGES:]]


if __name__ == "__main__":
    sys.exit(main())
