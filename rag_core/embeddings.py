from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import sqrt
from typing import Any

from .llm_config import LLMConfig


class EmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class DeterministicTestEmbeddings(EmbeddingClient):
    size: int = 32

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for token in text.lower().split():
            idx = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.size
            vec[idx] += 1.0
        norm = sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_openai_embeddings(config: LLMConfig) -> Any:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=config.model,
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        request_timeout=config.timeout_seconds,
    )
