# JARVIS CORE

Versión inicial del núcleo de JARVIS para Windows, en modo simulado.

## Alcance de la Fase 1

- Entrada por texto
- Configuración validada
- Contratos e interfaces tipadas
- Flujo de conversación básico
- IA simulada determinista
- Autorización central con denegación por defecto
- Registro mínimo de auditoría
- Ejecución de herramientas simuladas
- Pruebas automatizadas

## Requisitos

- Python 3.12

## Ejecución

El modo simulado es la única modalidad habilitada en esta fase.

```bash
pytest
```

## Limitaciones

- No se ejecutan comandos del sistema.
- No se realizan conexiones de red.
- No se usan dispositivos ni Windows reales.
- No se implementan voz, UI ni automatizaciones avanzadas.
