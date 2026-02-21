# RAG Experiments Lab

Laboratorio para diseñar, ejecutar y comparar experimentos de Retrieval-Augmented Generation (RAG) de forma repetible.

## 1) Fundaciones del repo

### Repo skeleton + convenciones

Estructura base del repositorio:

- `datasets/`: versiones de datasets de evaluación y/o corpus base.
- `rag_core/`: lógica principal de chunking, retrieval, prompting y contratos compartidos.
- `experiments/`: definiciones de runs y artefactos experimentales por configuración.
- `evaluation/`: métricas, scripts y reportes para comparar resultados.
- `ui/`: visualización de resultados, dashboards o utilidades de inspección.
- `docs/`: guías, contratos conceptuales y decisiones de diseño.

> **Por qué va primero:** sin un esqueleto común, cada experimento tiende a inventar su propia estructura y luego comparar A/B se vuelve ambiguo o costoso.

#### Convenciones de nombres

- `dataset_id`: identificador estable del dataset (ej. `faq_es_v1`).
- `run_id`: identificador único de ejecución (ej. `2026-02-21_hybrid_bm25_k20`).
- `doc_id`: ID de documento fuente dentro del dataset (ej. `doc_000123`).
- `chunk_id`: ID de fragmento derivado de `doc_id` + política de chunking (ej. `doc_000123_c004`).

Reglas prácticas:

1. Evitar espacios y mayúsculas.
2. Preferir formato `snake_case`.
3. Mantener `run_id` legible (fecha + estrategia + parámetro clave).

#### Extensibilidad sin sobrearquitectura

- Dejar **slots explícitos** (carpetas y contratos) para incorporar nuevos retrievers, rerankers y estrategias de prompt.
- Evitar abstraer de más antes de tener al menos 2 variantes reales que justifiquen una capa común.

---

## 2) README pedagógico (esqueleto)

### ¿Qué es este laboratorio?

Un entorno para iterar en RAG con disciplina experimental: misma data, cambios controlados de configuración y comparación sistemática de métricas.

### Objetivo

- Encontrar configuraciones de RAG que mejoren calidad factual y trazabilidad.
- Medir trade-offs entre recuperación, precisión de citas y abstención responsable.

### Cómo ejecutar (flujo sugerido)

1. Preparar o elegir un `dataset_id`.
2. Definir una configuración experimental (chunking + retrieval + prompt contract).
3. Ejecutar una corrida y registrar `run_id`.
4. Evaluar resultados con métricas estandarizadas.

### Cómo comparar A/B

- Mantener fijo el dataset y el set de preguntas.
- Cambiar **una sola dimensión principal** por comparación (ej. chunk size o retriever).
- Reportar métricas lado a lado con el mismo protocolo de evaluación.

### Cómo leer resultados

- **recall@k**: proporción de casos donde el contexto relevante aparece en los top-k recuperados.
- **citation precision**: fracción de citas incluidas por el modelo que realmente sustentan la afirmación.
- **abstención**: capacidad de no inventar respuesta cuando falta evidencia suficiente.

Interpretación rápida:

- Recall alto sin citation precision puede indicar ruido en grounding.
- Citation precision alta con recall bajo puede indicar cobertura insuficiente.
- Buena abstención reduce alucinación, pero en exceso puede afectar utilidad percibida.

### Errores típicos (desde el principio)

1. Comparar runs con datasets distintos sin marcarlo explícitamente.
2. Cambiar múltiples parámetros en A/B y luego no poder atribuir mejoras.
3. Medir solo exactitud textual sin revisar calidad de cita.
4. Ignorar los casos de “no respuesta” al analizar performance.

---

## 3) Modelo de configuración (schema mental)

Ver detalle en [`docs/config_contract.md`](docs/config_contract.md).

Resumen mínimo del contrato conceptual que toda config debe declarar:

- `chunking`: cómo se parte y referencia el contenido.
- `retrieval`: cómo se recupera evidencia (y con qué parámetros).
- `prompt_contract`: qué formato de respuesta y comportamiento de citación/abstención se exige.

> **Por qué fijarlo temprano:** si este contrato no se define al inicio, luego los experimentos no son comparables de manera confiable aunque “parezcan” similares.
