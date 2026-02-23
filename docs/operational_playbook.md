# Operational Playbook (with example)

This playbook provides a simple, repeatable way to use this application for a full RAG experimentation cycle.

## Objective

Run an A/B comparison on the same dataset, measure results, and keep reproducible artifacts.

## When to use this playbook

- When you are starting a new experiment.
- When you want to compare two variants (for example, `fixed_size` vs `by_headings`).
- When you need minimal traceability to discuss outcomes with your team.

## Pre-run checklist

1. Install dependencies:

```bash
pip install -e .
```

2. Configure credentials/model:

```bash
cp .env.example .env
# edit .env
```

3. Choose a test dataset (recommended to start):

- `dataset3_hierarchical_manual`

## Standard workflow (5 steps)

### Step 1: Define the hypothesis

Example:

> "`by_headings` should improve citation quality vs `fixed_size` on technical manuals, while keeping abstention behavior reasonable."

Record:

- primary variable being changed,
- target metric,
- success criterion.

### Step 2: Run a one-question smoke test

```bash
python -m rag_core.run_pipeline \
  --dataset-id dataset3_hierarchical_manual \
  --docs-path datasets/dataset3_hierarchical_manual/docs.json \
  --chunking-strategy by_headings \
  --question "How do I reset the unit?"
```

Quick checks:

- pipeline returns an answer,
- citations are present (`chunk_id`),
- abstention is used when evidence is insufficient.

### Step 3: Run a batch A/B experiment

```bash
python -m rag_core.run_experiments \
  --run-id 2026-02-23_manual_chunking_ab \
  --dataset-id dataset3_hierarchical_manual \
  --docs-path datasets/dataset3_hierarchical_manual/docs.json \
  --questions-path datasets/dataset3_hierarchical_manual/questions.json \
  --config-a-chunking-strategy fixed_size \
  --config-b-chunking-strategy by_headings
```

Expected output:

- `experiments/run_records/<run_id>/run_record.json`

### Step 4: Review the minimum metrics

Use these as baseline KPIs:

- `evidence_recall_at_k`
- `citation_precision`
- `answer_correctness`
- `abstention_correctness`

Practical rule:

- do not promote a variant if one metric improves but `citation_precision` or `abstention_correctness` degrades significantly.

### Step 5: Visual inspection in Gradio

```bash
python -m ui.gradio_lab_panel
```

Review in this order:

1. A/B configuration,
2. top-k chunks and scores,
3. answer + citations + abstention,
4. metrics and failed examples.

---

## Full copy/paste example

Case: compare chunking strategies on a technical manual dataset.

1. **Hypothesis**: `by_headings` improves citations.
2. **Run ID**: `2026-02-23_manual_chunking_ab`.
3. **A/B command**: same as Step 3.
4. **Decision rule**:
   - If `citation_precision(B) > citation_precision(A)` and `abstention_correctness` does not drop, adopt B as baseline.
   - Otherwise keep A and open a new experiment changing only `top_k`.

---

## Minimal results log template

```md
# Run review: <run_id>

## Hypothesis
- ...

## Compared configurations
- A: ...
- B: ...

## Metrics (A vs B)
- evidence_recall_at_k: ... vs ...
- citation_precision: ... vs ...
- answer_correctness: ... vs ...
- abstention_correctness: ... vs ...

## Decision
- Winner: A/B
- Reason: ...
- Next experiment: ...
```

## Anti-patterns to avoid

- Changing dataset and strategy at the same time.
- Choosing a winner based only on perceived answer quality without checking citations.
- Not recording `run_id` and losing reproducibility.
