# JARVIS CORE

Versión inicial del núcleo de JARVIS para Windows, en modo simulado.

## Funciones implementadas

- Entrada por texto
- Configuración validada
- Contratos e interfaces tipadas
- Flujo de conversación básico
- IA simulada determinista
- Autorización central con denegación por defecto
- Registro mínimo de auditoría
- Ejecución de herramientas simuladas
- Pruebas automatizadas
- Contrato de proveedor con sugerencias separadas de los datos de seguridad
- Selección explícita del proveedor, con simulador predeterminado y sin fallback para opciones no implementadas
- Catálogo de definiciones confiables separado de los ejecutores; el runner es el punto ordinario de autorización y despacho
- Salida pública del flujo de texto limitada a `user_text`, `intent`, `success`, `tool_result` y `error`

## Requisitos

- Python 3.12

## Ejecución

El modo y proveedor simulados son las únicas modalidades conectadas actualmente. La configuración local/remota se valida, pero no existen adaptadores de inferencia real ni transporte de red.

```bash
pytest
```

## Limitaciones

- No se ejecutan comandos del sistema.
- No se realizan conexiones de red.
- No se usan dispositivos ni Windows reales.
- No se implementan voz, UI ni automatizaciones avanzadas.
- La selección de proveedor es explícita y no tiene fallback: Echo permanece determinista; existe un adaptador informativo local para Ollama por loopback que solo se conecta cuando se inyecta mediante el gestor local. No se crea por defecto y no existe proveedor remoto conectado.
- `web: <consulta>` y `busca en Internet: <consulta>` representan una capacidad separada de búsqueda externa; su política de red deniega por defecto y no hay adaptador ni conexión, independientemente de la configuración de IA remota.
- La activación remota, las credenciales y las reglas de minimización de datos para proveedores reales quedan pendientes.
- El texto de entrada y el mensaje de Echo se devuelven completos y pueden contener información sensible; JARVIS no detecta todos los secretos en texto arbitrario.
- El proceso Python no es un sandbox: la separación de APIs reduce el acceso accidental, pero no aísla frente a código malicioso dentro del proceso.

Consulta [docs/configuration.md](docs/configuration.md) para las opciones actuales.
