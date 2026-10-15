# Arquitectura de JARVIS CORE

## Objetivo

Construir un núcleo local, seguro y extensible para una aplicación de asistente personal para Windows, con texto como interfaz primaria y ejecución simulada por defecto.

## Capas

- config: validación de configuración y entorno.
- core: contexto, errores e interfaces de contrato.
- conversation: entrada, intención y flujo del usuario.
- ai: proveedores de IA con un simulador determinista.
- security: política, autorización y auditoría.
- tools: definición, registro y ejecución ejecutada en modo simulado.

## Principios

- Denegación por defecto.
- IA propone, seguridad autoriza, herramientas ejecutan.
- Los proveedores son intercambiables.
- El modo simulado es la única ruta implementada.
- La seguridad se valida en el momento de usar la herramienta.
- El contexto de sesión no contiene permisos otorgados; sólo contiene contexto y trazabilidad.
