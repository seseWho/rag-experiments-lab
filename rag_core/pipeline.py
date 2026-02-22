from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import ChunkingConfig, chunk_sections
from .contracts import ResponseContractResult, enforce_response_contract
from .embeddings import EmbeddingClient
from .ingestion import load_and_normalize_docs
from .models import RetrievalResult, TraceRecord
from .vector_store import PersistentVectorStore


@dataclass(frozen=True)
class PipelineConfig:
    dataset_id: str
    docs_path: str
    chunking_strategy: str
    chunk_size: int = 350
    chunk_overlap: int = 40
    top_k: int = 4
    min_score_threshold: float = 0.06
    index_root: str = "experiments/indexes"
    traces_root: str = "experiments/traces"


class BasePipeline:
    def __init__(self, config: PipelineConfig, embedding_client: EmbeddingClient):
        self.config = config
        self.trace_path = Path(config.traces_root) / f"{config.dataset_id}_{config.chunking_strategy}.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        index_key = f"{config.dataset_id}__{config.chunking_strategy}_s{config.chunk_size}_o{config.chunk_overlap}"
        self.vector_store = PersistentVectorStore(Path(config.index_root) / index_key, embedding_client=embedding_client)

    def build_index(self) -> int:
        sections = load_and_normalize_docs(self.config.dataset_id, self.config.docs_path)
        chunks = chunk_sections(
            sections,
            ChunkingConfig(
                strategy=self.config.chunking_strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            ),
        )
        self.vector_store.build(chunks)
        return len(chunks)

    def query(self, question: str) -> ResponseContractResult:
        results = self.vector_store.search(question, top_k=self.config.top_k)
        record = TraceRecord(query=question, top_k=self.config.top_k, results=results)
        self._append_trace(record)
        return enforce_response_contract(
            query=question,
            results=results,
            min_score_threshold=self.config.min_score_threshold,
        )

    def query_with_results(self, question: str) -> tuple[ResponseContractResult, list[RetrievalResult]]:
        results = self.vector_store.search(question, top_k=self.config.top_k)
        record = TraceRecord(query=question, top_k=self.config.top_k, results=results)
        self._append_trace(record)
        response = enforce_response_contract(
            query=question,
            results=results,
            min_score_threshold=self.config.min_score_threshold,
        )
        return response, results

    def _append_trace(self, record: TraceRecord) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
