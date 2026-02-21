from __future__ import annotations

import json
import re
from pathlib import Path

from .models import NormalizedSection

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text.strip())
    return collapsed


def infer_version_from_title(title: str) -> str:
    lowered = title.lower()
    for token in ["v3", "v2", "v1", "2025", "2024", "2023"]:
        if token in lowered:
            return token.replace("v", "version_") if token.startswith("v") else token
    return "base"


def load_and_normalize_docs(dataset_id: str, docs_path: str | Path) -> list[NormalizedSection]:
    payload = json.loads(Path(docs_path).read_text(encoding="utf-8"))
    normalized: list[NormalizedSection] = []
    for doc in payload:
        doc_id = doc["doc_id"]
        version = infer_version_from_title(doc.get("title", ""))
        for section in doc.get("sections", []):
            normalized.append(
                NormalizedSection(
                    dataset_id=dataset_id,
                    doc_id=doc_id,
                    version=version,
                    section_id=section["section_id"],
                    heading=normalize_text(section.get("heading", "")),
                    content=normalize_text(section.get("content", "")),
                    metadata={
                        "title": normalize_text(doc.get("title", "")),
                        "source": "docs.json",
                    },
                )
            )
    return normalized
