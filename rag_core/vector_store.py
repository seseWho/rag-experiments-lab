from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .models import Chunk, RetrievalResult

_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class PersistentVectorStore:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "index.json"
        self._chunks_by_id: dict[str, Chunk] = {}
        self._doc_freq: Counter[str] = Counter()
        self._vectors: dict[str, dict[str, float]] = {}
        self._idf: dict[str, float] = {}
        self._loaded = False

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        term_counts: dict[str, Counter[str]] = {}
        self._doc_freq = Counter()

        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            counts = Counter(tokens)
            term_counts[chunk.chunk_id] = counts
            for tok in counts:
                self._doc_freq[tok] += 1

        n_docs = max(len(chunks), 1)
        self._idf = {
            tok: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for tok, freq in self._doc_freq.items()
        }

        self._vectors = {}
        for chunk_id, counts in term_counts.items():
            self._vectors[chunk_id] = self._tfidf(counts)

        self._save()
        self._loaded = True

    def _tfidf(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        vector = {tok: (cnt / total) * self._idf.get(tok, 0.0) for tok, cnt in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {tok: val / norm for tok, val in vector.items()}

    def _save(self) -> None:
        payload = {
            "chunks": [
                {
                    "dataset_id": c.dataset_id,
                    "doc_id": c.doc_id,
                    "version": c.version,
                    "section_id": c.section_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "metadata": c.metadata,
                }
                for c in self._chunks_by_id.values()
            ],
            "idf": self._idf,
            "vectors": self._vectors,
        }
        self.index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        self._chunks_by_id = {
            raw["chunk_id"]: Chunk(**raw)
            for raw in payload.get("chunks", [])
        }
        self._idf = {k: float(v) for k, v in payload.get("idf", {}).items()}
        self._vectors = {
            chunk_id: {tok: float(weight) for tok, weight in vector.items()}
            for chunk_id, vector in payload.get("vectors", {}).items()
        }
        self._loaded = True

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self._loaded:
            if self.index_file.exists():
                self.load()
            else:
                raise RuntimeError("Index not built or loaded")

        counts = Counter(_tokenize(query))
        query_vec = self._tfidf(counts)

        scored: list[tuple[str, float]] = []
        for chunk_id, vector in self._vectors.items():
            score = sum(query_vec.get(tok, 0.0) * weight for tok, weight in vector.items())
            if score > 0:
                scored.append((chunk_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results: list[RetrievalResult] = []
        for chunk_id, score in scored[:top_k]:
            chunk = self._chunks_by_id[chunk_id]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=chunk.text,
                    metadata={
                        "doc_id": chunk.doc_id,
                        "version": chunk.version,
                        "section_id": chunk.section_id,
                        **chunk.metadata,
                    },
                )
            )
        return results
