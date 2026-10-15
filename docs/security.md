# Seguridad de JARVIS CORE

## Reglas

1. La ejecución real está deshabilitada en esta fase.
2. La denegación por defecto aplica a permisos desconocidos o faltantes.
3. La IA no puede autorizar ni ejecutar acciones.
4. Las herramientas solo se ejecutan si están registradas y autorizadas.
5. Cada ejecución requiere validación en tiempo real.
6. Las confirmaciones están ligadas a acciones concretas.
7. La auditoría registra solo metadatos mínimos y no guarda secretos.
8. El contexto de la conversación no puede contener permisos concedidos; la autorización se decide por política y registro.

## Tratamiento de secretos

- No se almacenan claves, tokens ni secretos en el repositorio.
- La auditoría elimina valores sensibles en clave, token, password o secret.
- Los registros quedan reducidos a identificadores y resultados.
