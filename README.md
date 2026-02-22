# RAG Experiments Lab

A lab to design, run, and compare Retrieval-Augmented Generation (RAG) experiments in a repeatable way.

## 1) Repository foundations

### Repository skeleton + conventions

Base repository structure:

- `datasets/`: versions of evaluation datasets and/or source corpora.
- `rag_core/`: core logic for chunking, retrieval, prompting, and shared contracts.
- `experiments/`: run definitions and experiment artifacts per configuration.
- `evaluation/`: metrics, scripts, and reports used to compare results.
- `ui/`: result visualization, dashboards, or inspection tools.
- `docs/`: guides, conceptual contracts, and design decisions.

> **Why this comes first:** without a common skeleton, each experiment tends to invent its own structure, and A/B comparisons become ambiguous and expensive.

#### Naming conventions

- `dataset_id`: stable identifier for a dataset (for example: `faq_en_v1`).
- `run_id`: unique execution identifier (for example: `2026-02-21_hybrid_bm25_k20`).
- `doc_id`: source document ID inside a dataset (for example: `doc_000123`).
- `chunk_id`: chunk ID derived from `doc_id` + chunking policy (for example: `doc_000123_c004`).

Practical rules:

1. Avoid spaces and uppercase letters.
2. Prefer `snake_case`.
3. Keep `run_id` readable (date + strategy + key parameter).

#### Extensibility without over-architecture

- Leave explicit **slots** (folders and contracts) to add retrievers, rerankers, and prompt strategies.
- Avoid premature abstraction until at least two real variants justify a shared layer.

---

## 2) Pedagogical README skeleton

### What is this lab?

An environment to iterate on RAG with experimental discipline: same data, controlled configuration changes, and systematic metric comparison.

### Objective

- Identify RAG configurations that improve factual quality and traceability.
- Measure trade-offs between retrieval coverage, citation quality, and responsible abstention.

### How to run (suggested flow)

1. Prepare or select a `dataset_id`.
2. Define an experiment configuration (chunking + retrieval + prompt contract).
3. Execute a run and record `run_id`.
4. Evaluate outputs with standardized metrics.

### How to compare A/B

- Keep the dataset and question set fixed.
- Change **one primary dimension at a time** per comparison (for example chunk size or retriever).
- Report side-by-side metrics under the same evaluation protocol.

### How to read results

- **recall@k**: share of cases where relevant context appears in top-k retrieved items.
- **citation precision**: fraction of model citations that actually support the corresponding claim.
- **abstention**: ability to avoid answering when evidence is insufficient.

Quick interpretation:

- High recall with low citation precision may indicate grounding noise.
- High citation precision with low recall may indicate insufficient coverage.
- Good abstention reduces hallucinations, but too much abstention can hurt usefulness.

### Common mistakes (from day one)

1. Comparing runs built on different datasets without explicit labeling.
2. Changing multiple variables in an A/B test and losing attribution.
3. Measuring only textual correctness without citation quality checks.
4. Ignoring “no answer” cases during performance analysis.

---

## 3) Configuration model (mental schema)

See details in [`docs/config_contract.md`](docs/config_contract.md).

Minimum conceptual contract every configuration should declare:

- `chunking`: how content is split and referenced.
- `retrieval`: how evidence is retrieved (and with which parameters).
- `prompt_contract`: required response format and citation/abstention behavior.

> **Why define this early:** if this contract is not fixed early, experiments are hard to compare reliably even when they look similar.

## 4) Base pipeline implementation (MVP)

This repository now includes a minimal `rag_core/` implementation that covers:

- Ingestion + normalization with `dataset_id`, `doc_id`, `version`, `section_id` metadata.
- Two chunking strategies: `fixed_size` and `by_headings`.
- Deterministic `chunk_id` generation.
- Persistent vector index versioned by `(dataset + chunking config)` under `experiments/indexes/`.
- Basic retriever with `top_k` and score traces in `experiments/traces/`.
- Context builder + response contract with citation IDs and a simple abstention policy.

Quick run example:

```bash
python -m rag_core.run_pipeline \
  --dataset-id dataset3_hierarchical_manual \
  --docs-path datasets/dataset3_hierarchical_manual/docs.json \
  --chunking-strategy by_headings \
  --question "How do I reset the unit?"
```

## 5) Environment and dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure LLM connection variables:

```bash
cp .env.example .env
# then edit .env with your keys/provider/model
```

The CLI loads `.env` automatically and reports the configured provider/model.

Current implementation uses **LangChain + OpenAI embeddings** for vectorization and retrieval scoring.


## 6) Experiment runner + run records

Run batch A/B experiments over a fixed question set and persist a reproducible run record:

```bash
python -m rag_core.run_experiments \
  --run-id 2026-02-22_chunking_ab \
  --dataset-id dataset1_regulations_versions \
  --docs-path datasets/dataset1_regulations_versions/docs.json \
  --questions-path datasets/dataset1_regulations_versions/questions.json \
  --config-a-chunking-strategy fixed_size \
  --config-b-chunking-strategy by_headings
```

This generates `experiments/run_records/<run_id>/run_record.json` including:

- config snapshot per variant (A/B),
- per-question traces and responses,
- aggregated metrics (`answered`, `abstained`, `citation_hit_rate`),
- `context_sent_to_llm` saved as chunk IDs (not full text) for lean reproducibility.
