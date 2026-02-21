# Config Contract (Mental Schema)

This document defines the conceptual configuration contract for RAG experiments.
It does not enforce a specific file format yet (YAML/JSON), but it establishes shared language.

## 1. Chunking

Recommended fields:

- `strategy`: token, sentence, semantic, markdown, etc.
- `chunk_size`: target chunk size.
- `chunk_overlap`: overlap between chunks.
- `metadata_policy`: metadata preserved for each `chunk_id`.

Questions this section must answer:

- How is traceability guaranteed from `chunk_id -> doc_id -> dataset_id`?
- How does the strategy affect recall and cost?

## 2. Retrieval

Recommended fields:

- `retriever_type`: dense, sparse, hybrid.
- `index_id`: index/version reference.
- `top_k`: number of retrieved candidates.
- `filters`: metadata-based constraints.
- `reranker`: model and ranking criteria (if used).

Questions this section must answer:

- Which evidence enters the prompt, and why?
- How is consistency controlled across runs?

## 3. Prompt Contract

Recommended fields:

- `system_prompt_version`: version of base behavior.
- `citation_policy`: required citation format.
- `abstention_policy`: when to answer with “insufficient evidence”.
- `output_schema`: output structure (free text or structured JSON).

Questions this section must answer:

- What defines a valid response?
- How can citation and abstention compliance be validated automatically?

## 4. Experiment identity

Recommended fields:

- `dataset_id`
- `run_id`
- `notes` (hypothesis and changes vs baseline)

Key rule:

- Every evaluation result must map unambiguously to one configuration and one dataset.

## 5. Operating principle

Keep the contract simple but explicit. If a new requirement appears repeatedly in two or more experiments, promote it to the contract; otherwise keep it as a local extension.
