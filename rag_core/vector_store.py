from __future__ import annotations

import json
import math
from pathlib import Path

from .embeddings import EmbeddingClient
from .models import Chunk, RetrievalResult


class PersistentVectorStore:
    def __init__(self, index_dir: str | Path, embedding_client: EmbeddingClient):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "index.json"
        self.embedding_client = embedding_client

        self._chunks_by_id: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._loaded = False

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        texts = [c.text for c in chunks]
        embeddings = self.embedding_client.embed_documents(texts)
        self._vectors = {
            chunk.chunk_id: vector
            for chunk, vector in zip(chunks, embeddings, strict=True)
        }
        self._save()
        self._loaded = True

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
            "vectors": self._vectors,
        }
        self.index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        self._chunks_by_id = {raw["chunk_id"]: Chunk(**raw) for raw in payload.get("chunks", [])}
        self._vectors = {
            chunk_id: [float(weight) for weight in vector]
            for chunk_id, vector in payload.get("vectors", {}).items()
        }
        self._loaded = True

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self._loaded:
            if self.index_file.exists():
                self.load()
            else:
                raise RuntimeError("Index not built or loaded")

        query_vec = self.embedding_client.embed_query(query)

        scored: list[tuple[str, float]] = []
        for chunk_id, vector in self._vectors.items():
            score = self._cosine_similarity(query_vec, vector)
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
