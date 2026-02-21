from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedSection:
    dataset_id: str
    doc_id: str
    version: str
    section_id: str
    heading: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    dataset_id: str
    doc_id: str
    version: str
    section_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceRecord:
    query: str
    top_k: int
    results: list[RetrievalResult]
