# Config Contract (Schema Mental)

Este documento define el contrato conceptual de configuración para experimentos RAG.
No obliga todavía un formato técnico (YAML/JSON), pero sí fija el lenguaje común.

## 1. Chunking

Campos recomendados:

- `strategy`: token, sentence, semantic, markdown, etc.
- `chunk_size`: tamaño objetivo.
- `chunk_overlap`: solapamiento entre chunks.
- `metadata_policy`: qué metadatos persisten por `chunk_id`.

Preguntas que debe responder:

- ¿Cómo se garantiza trazabilidad de `chunk_id -> doc_id -> dataset_id`?
- ¿Cómo impacta la estrategia en recall y costo?

## 2. Retrieval

Campos recomendados:

- `retriever_type`: dense, sparse, hybrid.
- `index_id`: referencia al índice/versionado.
- `top_k`: cantidad de candidatos recuperados.
- `filters`: restricciones por metadata.
- `reranker`: modelo y criterio (si aplica).

Preguntas que debe responder:

- ¿Qué evidencias entran al prompt y por qué?
- ¿Cómo se controla consistencia entre runs?

## 3. Prompt Contract

Campos recomendados:

- `system_prompt_version`: versión del comportamiento base.
- `citation_policy`: formato de citas requerido.
- `abstention_policy`: cuándo responder “no suficiente evidencia”.
- `output_schema`: estructura de salida (texto libre/JSON estructurado).

Preguntas que debe responder:

- ¿Qué constituye una respuesta válida?
- ¿Cómo validar automáticamente cumplimiento de citas y abstención?

## 4. Identidad del experimento

Campos recomendados:

- `dataset_id`
- `run_id`
- `notes` (hipótesis y cambios respecto al baseline)

Regla clave:

- Todo resultado de evaluación debe poder mapearse de forma unívoca a una configuración y a un dataset concretos.

## 5. Principio operativo

Mantener el contrato simple pero explícito. Si aparece una nueva necesidad recurrente en >=2 experimentos, promoverla al contrato; si no, mantenerla como extensión local.
