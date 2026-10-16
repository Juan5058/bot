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
9. El pipeline crea un contexto por solicitud y lo comparte entre la propuesta y la ejecución. Si el proveedor lo altera, la solicitud se deniega y se restauran sus campos originales; la clasificación solo recibe el texto porque ese es el contrato actual del proveedor.
10. La entrada de texto se limita a 8.000 caracteres y no puede estar vacía. Una clasificación debe ser un diccionario con `intent="text_request"`; las clasificaciones ausentes, malformadas o no admitidas se deniegan antes de solicitar una propuesta.
11. Antes de consultar el autorizador y despachar una herramienta, el runner comprueba que sus parámetros sean un diccionario JSON-compatible acotado (32 niveles, 256 elementos por contenedor y 2.048 nodos como máximo). Esta comprobación valida sin modificar los parámetros; no convierte ni elimina valores.

## Tratamiento de secretos

- No se almacenan claves, tokens ni secretos en el repositorio.
- La auditoría conserva únicamente una lista permitida de metadatos con valores enumerados o identificadores acotados.
- El texto de conversación, los argumentos, los resultados completos, los mensajes de excepción y las estructuras desconocidas no se persisten en los detalles de auditoría.
- Los identificadores de sesión y acción se limitan a caracteres y longitudes controlados; valores inválidos se reemplazan por marcadores fijos y se etiquetan como inválidos en los detalles permitidos.
- Las entradas devueltas por `log()` y `entries` son copias defensivas. Esto evita modificaciones accidentales mediante esas referencias, pero el registro sigue siendo memoria del proceso y no es almacenamiento persistente ni resistente a código arbitrario en el proceso.
- `ConversationResult.to_dict()` serializa explícitamente los campos de la propuesta, excluye `confirmation_token` y redacta parámetros cuyas claves indican secretos conocidos. Rechaza valores de parámetros fuera de los tipos JSON compatibles; no puede detectar secretos bajo claves arbitrarias.
- La auditoría se escribe después de que el runner termina. Si el logger falla después de una operación permitida, esa operación simulada ya ocurrió y el evento podría perderse; si la autorización deniega, el runner no llama a la herramienta antes de que se intente auditar la denegación.

## Confirmación e identidad

- Esta fase no incluye un verificador de confirmación de usuario; por ello, las herramientas registradas con `requires_confirmation=True` se deniegan, aunque la propuesta incluya un token o una acción en `confirmed_actions`.
- Esta fase tampoco incluye un verificador de identidad. Una política con `identity_required=True` deniega la operación; los `claims` no se consideran prueba de identidad.

## Inmutabilidad y límites de proceso

- El catálogo predeterminado confiable se define en el núcleo y solo incluye `EchoTool`, con el permiso `tool.echo_message` y metadatos de riesgo, confirmación e identidad y modo simulado fijados por el núcleo. `ToolRegistry()` siempre usa este catálogo. Los catálogos adicionales solo pueden suministrarse mediante `ToolRegistry.for_trusted_composition`, destinado exclusivamente a código controlado de inicialización y pruebas con un catálogo `dict` explícito; se copian y validan las definiciones y los registros deben congelarse antes de usarse con autorización o ejecución. La aplicación normal no toma catálogos de entradas de usuario, propuestas de IA ni datos externos.
- La ruta de composición personalizada es una convención de arquitectura dentro del proceso, no una barrera frente a código Python arbitrario que pueda invocarla o alterar el proceso.
- La política deniega por defecto y verifica el par exacto de identidad de herramienta y permiso autorizado. Que un permiso aparezca en los metadatos de una implementación no basta para autorizarla.
- El runner y el servicio de autorización exponen sus referencias al registro como propiedades de solo lectura y rechazan por la API ordinaria reasignar o borrar las referencias internas después de construir los objetos. El runner comprueba además al inicio de cada ejecución que ambos siguen usando la misma instancia, antes de obtener o despachar una herramienta. Esto evita cambios accidentales y detecta discrepancias, pero no protege frente a código Python arbitrario que use mecanismos de bajo nivel o manipule el estado interno del proceso.
- Los atributos de una instancia registrada no se pueden reasignar mediante operaciones ordinarias. Esto no impide sustituir el método de una clase ni protege contra código malicioso dentro del proceso.
- Estas defensas reducen errores y cambios accidentales en las rutas normales, pero no sustituyen una frontera de seguridad frente a código arbitrario que ya pueda ejecutar dentro del mismo proceso Python.
- `execution_mode="simulated"` es una comprobación de contrato, no un sandbox: una implementación personalizada de `ToolDefinition` es código Python confiable y podría realizar efectos externos aunque se declare simulada. El registro predeterminado solo incluye `EchoTool`; no se cargan plugins. Para admitir herramientas no confiables se necesitaría aislamiento de proceso y un protocolo de capacidades que esta fase no implementa.
- La autorización sigue siendo centralizada y la ejecución real permanece bloqueada en la fase actual.
