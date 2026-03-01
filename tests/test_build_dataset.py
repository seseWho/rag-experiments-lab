from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rag_core.build_dataset import build_docs_json, load_docs_from_json, parse_markdown_sections


def test_parse_markdown_sections_splits_by_headings() -> None:
    text = "# Intro\nfirst\n## Details\nsecond"
    sections = parse_markdown_sections(text)
    assert sections == [
        {"heading": "Intro", "text": "first\n"},
        {"heading": "Details", "text": "second"},
    ]


def test_build_docs_json_deterministic_ids_and_metadata(tmp_path: Path) -> None:
    b_file = tmp_path / "b.txt"
    a_file = tmp_path / "a.md"
    b_file.write_text("Line 1\r\nLine 2", encoding="utf-8")
    a_file.write_text("# H1\r\nA\r\n## H2\r\nB", encoding="utf-8")

    docs = build_docs_json([a_file, b_file], {"doc_type": "policy", "version": "v1", "date": "2025-01-01"})
    assert [d["doc_id"] for d in docs] == ["D1", "D2"]
    assert docs[0]["sections"][1]["section_id"] == "D1-S2"
    assert docs[1]["sections"][0]["text"] == "Line 1\nLine 2"
    assert docs[0]["doc_type"] == "policy"


def test_load_docs_from_json_keeps_existing_ids_and_fills_missing(tmp_path: Path) -> None:
    in_path = tmp_path / "input.json"
    in_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "DOC-X",
                    "title": "Alpha",
                    "sections": [{"section_id": "SEC-X", "heading": "h", "text": "t"}],
                },
                {"title": "Beta", "sections": [{"heading": "Only", "content": "Body"}]},
            ]
        ),
        encoding="utf-8",
    )

    docs = load_docs_from_json(in_path, {"doc_type": "", "version": "", "date": ""})
    assert docs[0]["doc_id"] == "DOC-X"
    assert docs[0]["sections"][0]["section_id"] == "SEC-X"
    assert docs[1]["doc_id"] == "D2"
    assert docs[1]["sections"][0]["section_id"] == "D2-S1"
    assert docs[1]["sections"][0]["text"] == "Body"


def test_cli_creates_dataset_folder(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "b.txt").write_text("hello", encoding="utf-8")
    (src / "a.md").write_text("# One\ntext", encoding="utf-8")

    out_dir = tmp_path / "datasets" / "demo1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_core.build_dataset",
            "--dataset-id",
            "demo1",
            "--input-path",
            str(src),
            "--out-dir",
            str(out_dir),
            "--doc-type",
            "notes",
        ],
        check=True,
    )

    docs = json.loads((out_dir / "docs.json").read_text(encoding="utf-8"))
    questions = json.loads((out_dir / "questions.json").read_text(encoding="utf-8"))
    assert (out_dir / "README.md").exists()
    assert [d["title"] for d in docs] == ["a", "b"]
    assert questions == []
