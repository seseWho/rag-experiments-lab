from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DocsIndex = dict[str, dict[str, str | set[str]]]


def validate_docs(docs: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(docs, list):
        return ["docs.json must contain a list"]

    seen_doc_ids: set[str] = set()
    seen_section_ids: set[str] = set()

    for doc_idx, doc in enumerate(docs, start=1):
        item_prefix = f"docs[{doc_idx}]"
        if not isinstance(doc, dict):
            errors.append(f"{item_prefix} must be an object")
            continue

        doc_id = doc.get("doc_id")
        title = doc.get("title")
        sections = doc.get("sections")

        if not isinstance(doc_id, str) or not doc_id.strip():
            errors.append(f"{item_prefix}.doc_id must be a non-empty string")
            doc_id = f"<invalid-doc-{doc_idx}>"
        elif doc_id in seen_doc_ids:
            errors.append(f"Duplicate doc_id: {doc_id}")
        else:
            seen_doc_ids.add(doc_id)

        if not isinstance(title, str) or not title.strip():
            errors.append(f"{item_prefix}.title must be a non-empty string")

        if not isinstance(sections, list):
            errors.append(f"{item_prefix}.sections must be a list")
            continue

        for section_idx, section in enumerate(sections, start=1):
            section_prefix = f"{item_prefix}.sections[{section_idx}]"
            if not isinstance(section, dict):
                errors.append(f"{section_prefix} must be an object")
                continue

            section_id = section.get("section_id")
            heading = section.get("heading")
            text = section.get("text")

            if not isinstance(section_id, str) or not section_id.strip():
                errors.append(f"{section_prefix}.section_id must be a non-empty string")
            elif section_id in seen_section_ids:
                errors.append(f"Duplicate section_id: {section_id}")
            else:
                seen_section_ids.add(section_id)

            if not isinstance(heading, str) or not heading.strip():
                errors.append(f"{section_prefix}.heading must be a non-empty string")

            if not isinstance(text, str):
                errors.append(f"{section_prefix}.text must be a string")

    return errors


def build_docs_index(docs: Any) -> DocsIndex:
    doc_ids: set[str] = set()
    section_ids: set[str] = set()
    section_to_doc: dict[str, str] = {}

    if not isinstance(docs, list):
        return {"doc_ids": doc_ids, "section_ids": section_ids, "section_to_doc": section_to_doc}

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id")
        if isinstance(doc_id, str) and doc_id.strip():
            doc_ids.add(doc_id)
        sections = doc.get("sections")
        if not isinstance(sections, list):
            continue
        for section in sections:
            if not isinstance(section, dict):
                continue
            section_id = section.get("section_id")
            if isinstance(section_id, str) and section_id.strip():
                section_ids.add(section_id)
                if isinstance(doc_id, str) and doc_id.strip():
                    section_to_doc[section_id] = doc_id

    return {"doc_ids": doc_ids, "section_ids": section_ids, "section_to_doc": section_to_doc}


def validate_questions(questions: Any, docs_index: DocsIndex) -> list[str]:
    errors: list[str] = []

    if not isinstance(questions, list):
        return ["questions.json must contain a list"]

    seen_question_ids: set[str] = set()
    doc_ids = docs_index["doc_ids"]
    section_ids = docs_index["section_ids"]
    section_to_doc = docs_index["section_to_doc"]

    for q_idx, question in enumerate(questions, start=1):
        item_prefix = f"questions[{q_idx}]"
        if not isinstance(question, dict):
            errors.append(f"{item_prefix} must be an object")
            continue

        question_id = question.get("question_id")
        question_text = question.get("question_text")

        if not isinstance(question_id, str) or not question_id.strip():
            errors.append(f"{item_prefix}.question_id must be a non-empty string")
        elif question_id in seen_question_ids:
            errors.append(f"Duplicate question_id: {question_id}")
        else:
            seen_question_ids.add(question_id)

        if not isinstance(question_text, str) or not question_text.strip():
            errors.append(f"{item_prefix}.question_text must be a non-empty string")

        expected_answer = question.get("expected_answer")
        expected_doc_id = question.get("expected_doc_id")
        expected_section_id = question.get("expected_section_id")
        unanswerable = question.get("unanswerable")

        if expected_answer is not None and not isinstance(expected_answer, str):
            errors.append(f"{item_prefix}.expected_answer must be a string when provided")
        if expected_doc_id is not None and not isinstance(expected_doc_id, str):
            errors.append(f"{item_prefix}.expected_doc_id must be a string when provided")
        if expected_section_id is not None and not isinstance(expected_section_id, str):
            errors.append(f"{item_prefix}.expected_section_id must be a string when provided")
        if unanswerable is not None and not isinstance(unanswerable, bool):
            errors.append(f"{item_prefix}.unanswerable must be a boolean when provided")

        if isinstance(expected_doc_id, str):
            if expected_doc_id not in doc_ids:
                errors.append(f"{item_prefix}.expected_doc_id references unknown doc_id: {expected_doc_id}")

        if isinstance(expected_section_id, str):
            if expected_section_id not in section_ids:
                errors.append(
                    f"{item_prefix}.expected_section_id references unknown section_id: {expected_section_id}"
                )
            elif isinstance(expected_doc_id, str):
                owner_doc_id = section_to_doc.get(expected_section_id)
                if owner_doc_id is not None and owner_doc_id != expected_doc_id:
                    errors.append(
                        f"{item_prefix}.expected_section_id {expected_section_id} belongs to {owner_doc_id}, not {expected_doc_id}"
                    )

    return errors


def _load_json(path: Path, label: str) -> tuple[Any | None, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"Failed to read {label} at {path}: {exc}"]

    try:
        return json.loads(text), []
    except json.JSONDecodeError as exc:
        return None, [f"Failed to parse {label} at {path}: {exc}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate docs/questions dataset consistency")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--docs-path", default=None)
    parser.add_argument("--questions-path", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.datasets_root) / args.dataset_id
    docs_path = Path(args.docs_path) if args.docs_path else dataset_dir / "docs.json"
    questions_path = Path(args.questions_path) if args.questions_path else dataset_dir / "questions.json"

    docs, docs_load_errors = _load_json(docs_path, "docs.json")
    questions, questions_load_errors = _load_json(questions_path, "questions.json")

    errors = [*docs_load_errors, *questions_load_errors]

    docs_count = 0
    sections_count = 0
    questions_count = 0

    if docs is not None:
        docs_errors = validate_docs(docs)
        errors.extend(docs_errors)
        if isinstance(docs, list):
            docs_count = len(docs)
            sections_count = sum(len(d.get("sections", [])) for d in docs if isinstance(d, dict) and isinstance(d.get("sections"), list))

    docs_index = build_docs_index(docs)

    if questions is not None:
        questions_errors = validate_questions(questions, docs_index)
        errors.extend(questions_errors)
        if isinstance(questions, list):
            questions_count = len(questions)

    result = "PASS" if not errors else "FAIL"
    print(f"docs={docs_count} sections={sections_count} questions={questions_count}")
    print(f"validation={result}")
    if errors:
        for error in errors:
            print(error)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
