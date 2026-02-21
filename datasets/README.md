# Didactic Evaluation Datasets (MVP)

This folder contains three small English datasets for RAG experiments.

## Common format

Each dataset has:

- `docs.json`: source documents with chunk-like sections.
- `questions.json`: evaluation questions with minimal gold labels.

### `docs.json` schema

```json
[
  {
    "doc_id": "string",
    "title": "string",
    "sections": [
      {
        "section_id": "string",
        "heading": "string",
        "content": "string"
      }
    ]
  }
]
```

### `questions.json` schema

```json
[
  {
    "question_id": "string",
    "question": "string",
    "expected_answer": "string",
    "expected_doc_id": "string",
    "expected_section_id": "string"
  }
]
```

## Dataset inventory

1. `dataset1_regulations_versions` — policy/regulation version tracking.
2. `dataset2_biographies_dates` — biographical timelines with dates.
3. `dataset3_hierarchical_manual` — hierarchical operations manual.
