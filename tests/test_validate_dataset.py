from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rag_core.validate_dataset import build_docs_index, validate_docs, validate_questions


def _write_dataset(dataset_dir: Path, docs: object, questions: object) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "docs.json").write_text(json.dumps(docs), encoding="utf-8")
    (dataset_dir / "questions.json").write_text(json.dumps(questions), encoding="utf-8")


def test_validate_docs_and_questions_pass() -> None:
    docs = [
        {
            "doc_id": "D1",
            "title": "Doc One",
            "sections": [
                {"section_id": "D1-S1", "heading": "H1", "text": ""},
                {"section_id": "D1-S2", "heading": "H2", "text": "T2"},
            ],
        }
    ]
    questions = [
        {
            "question_id": "Q1",
            "question_text": "What is this?",
            "expected_doc_id": "D1",
            "expected_section_id": "D1-S2",
            "expected_answer": "T2",
        },
        {
            "question_id": "Q2",
            "question_text": "No answer?",
            "unanswerable": True,
        },
    ]

    assert validate_docs(docs) == []
    assert validate_questions(questions, build_docs_index(docs)) == []


def test_validate_docs_duplicate_section_id_fails() -> None:
    docs = [
        {
            "doc_id": "D1",
            "title": "Doc One",
            "sections": [
                {"section_id": "S1", "heading": "A", "text": "T1"},
                {"section_id": "S1", "heading": "B", "text": "T2"},
            ],
        }
    ]

    errors = validate_docs(docs)
    assert any("Duplicate section_id: S1" in error for error in errors)


def test_cli_fails_when_question_references_missing_doc_id(tmp_path: Path) -> None:
    dataset_id = "invalid_missing_doc_ref"
    dataset_dir = tmp_path / "datasets" / dataset_id
    _write_dataset(
        dataset_dir,
        docs=[
            {
                "doc_id": "D1",
                "title": "Doc One",
                "sections": [{"section_id": "D1-S1", "heading": "H", "text": "T"}],
            }
        ],
        questions=[
            {
                "question_id": "Q1",
                "question_text": "Which doc?",
                "expected_doc_id": "D9",
            }
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_core.validate_dataset",
            "--dataset-id",
            dataset_id,
            "--datasets-root",
            str(tmp_path / "datasets"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "validation=FAIL" in proc.stdout
    assert "references unknown doc_id: D9" in proc.stdout


def test_cli_passes_for_valid_dataset(tmp_path: Path) -> None:
    dataset_id = "valid"
    dataset_dir = tmp_path / "datasets" / dataset_id
    _write_dataset(
        dataset_dir,
        docs=[
            {
                "doc_id": "D1",
                "title": "Doc One",
                "sections": [{"section_id": "D1-S1", "heading": "H", "text": ""}],
            }
        ],
        questions=[
            {
                "question_id": "Q1",
                "question_text": "What?",
                "expected_doc_id": "D1",
                "expected_section_id": "D1-S1",
            }
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_core.validate_dataset",
            "--dataset-id",
            dataset_id,
            "--datasets-root",
            str(tmp_path / "datasets"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "validation=PASS" in proc.stdout
