from __future__ import annotations

from dataclasses import dataclass

from .models import RetrievalResult


@dataclass(frozen=True)
class ResponseContractResult:
    answer: str
    citations: list[str]
    snippets: list[str]
    abstained: bool
    reason: str | None = None


def build_context(results: list[RetrievalResult], max_snippet_chars: int = 220) -> dict[str, list[str]]:
    chunk_ids = [r.chunk_id for r in results]
    snippets = [r.text[:max_snippet_chars] for r in results]
    return {"chunk_ids": chunk_ids, "snippets": snippets}


def enforce_response_contract(
    query: str,
    results: list[RetrievalResult],
    min_score_threshold: float = 0.06,
) -> ResponseContractResult:
    del query
    if not results or results[0].score < min_score_threshold:
        return ResponseContractResult(
            answer="Insufficient evidence to answer with confidence.",
            citations=[],
            snippets=[],
            abstained=True,
            reason="top_score_below_threshold",
        )

    top = results[0]
    answer = f"Based on {top.metadata['doc_id']} / {top.metadata['section_id']}, the most relevant evidence is: {top.text[:180]}"
    context = build_context(results)
    return ResponseContractResult(
        answer=answer,
        citations=context["chunk_ids"],
        snippets=context["snippets"],
        abstained=False,
    )
