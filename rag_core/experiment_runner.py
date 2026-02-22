from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .contracts import ResponseContractResult
from .models import RetrievalResult
from .pipeline import BasePipeline, PipelineConfig


@dataclass(frozen=True)
class BatchQuestion:
    question_id: str
    question: str
    expected_answer: str | None = None
    expected_doc_id: str | None = None
    expected_section_id: str | None = None
    expected_abstain: bool | None = None


@dataclass(frozen=True)
class BatchRunConfig:
    config_id: str
    pipeline: PipelineConfig


@dataclass(frozen=True)
class RunSummary:
    total_questions: int
    answered: int
    abstained: int
    citation_hit_rate: float
    evidence_recall_at_k: float
    citation_precision: float
    answer_correctness: float
    abstention_correctness: float


class ExperimentRunner:
    def __init__(self, pipeline_factory):
        self.pipeline_factory = pipeline_factory

    def run_ab(
        self,
        questions: list[BatchQuestion],
        configs: list[BatchRunConfig],
        run_id: str,
        output_root: str = "experiments/run_records",
    ) -> Path:
        if len(configs) < 2:
            raise ValueError("At least two configs are required for an A/B run")

        output_dir = Path(output_root) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        run_record: dict[str, object] = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "question_count": len(questions),
            "configs": [],
        }

        for cfg in configs:
            pipeline = self.pipeline_factory(cfg.pipeline)
            index_file = Path(cfg.pipeline.index_root) / (
                f"{cfg.pipeline.dataset_id}__{cfg.pipeline.chunking_strategy}_s{cfg.pipeline.chunk_size}_o{cfg.pipeline.chunk_overlap}"
            ) / "index.json"
            if not index_file.exists():
                pipeline.build_index()

            question_records: list[dict[str, object]] = []
            citation_hits = 0
            answered = 0
            citation_precision_acc = 0.0
            answer_correctness_acc = 0.0
            abstention_correct_acc = 0.0

            for item in questions:
                response, retrieval_results = pipeline.query_with_results(item.question)
                if not response.abstained:
                    answered += 1

                evidence_hit = _evidence_hit(item, retrieval_results)
                if evidence_hit:
                    citation_hits += 1

                citation_precision_acc += _citation_precision(item, response, retrieval_results)
                answer_correctness_acc += _answer_correctness(item, response)
                abstention_correct_acc += _abstention_correctness(item, response)

                question_records.append(
                    {
                        "question_id": item.question_id,
                        "question": item.question,
                        "expected": {
                            "answer": item.expected_answer,
                            "doc_id": item.expected_doc_id,
                            "section_id": item.expected_section_id,
                        },
                        "response": self._response_to_dict(response),
                        "trace": self._trace_to_dict(retrieval_results),
                    }
                )

            summary = RunSummary(
                total_questions=len(questions),
                answered=answered,
                abstained=len(questions) - answered,
                citation_hit_rate=(citation_hits / len(questions)) if questions else 0.0,
                evidence_recall_at_k=(citation_hits / len(questions)) if questions else 0.0,
                citation_precision=(citation_precision_acc / len(questions)) if questions else 0.0,
                answer_correctness=(answer_correctness_acc / len(questions)) if questions else 0.0,
                abstention_correctness=(abstention_correct_acc / len(questions)) if questions else 0.0,
            )

            config_record = {
                "config_id": cfg.config_id,
                "config_snapshot": asdict(cfg.pipeline),
                "summary": asdict(summary),
                "questions": question_records,
            }
            run_record["configs"].append(config_record)

        record_file = output_dir / "run_record.json"
        record_file.write_text(json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record_file

    @staticmethod
    def _response_to_dict(response: ResponseContractResult) -> dict[str, object]:
        return {
            "answer": response.answer,
            "citations": response.citations,
            "abstained": response.abstained,
            "reason": response.reason,
            "context_sent_to_llm": response.citations,
        }

    @staticmethod
    def _trace_to_dict(results: list[RetrievalResult]) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": item.chunk_id,
                "score": item.score,
                "doc_id": item.metadata.get("doc_id"),
                "section_id": item.metadata.get("section_id"),
            }
            for item in results
        ]


def load_questions(path: str) -> list[BatchQuestion]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BatchQuestion(**row) for row in payload]


def _evidence_hit(question: BatchQuestion, retrieval_results: list[RetrievalResult]) -> bool:
    if not question.expected_doc_id:
        return False

    for result in retrieval_results:
        if result.metadata.get("doc_id") != question.expected_doc_id:
            continue
        if question.expected_section_id and result.metadata.get("section_id") != question.expected_section_id:
            continue
        return True
    return False


def _citation_precision(
    question: BatchQuestion,
    response: ResponseContractResult,
    retrieval_results: list[RetrievalResult],
) -> float:
    if not response.citations:
        return 1.0 if response.abstained else 0.0

    retrieval_by_chunk = {item.chunk_id: item for item in retrieval_results}
    supported = 0
    for chunk_id in response.citations:
        candidate = retrieval_by_chunk.get(chunk_id)
        if not candidate:
            continue

        doc_ok = True
        section_ok = True
        if question.expected_doc_id:
            doc_ok = candidate.metadata.get("doc_id") == question.expected_doc_id
        if question.expected_section_id:
            section_ok = candidate.metadata.get("section_id") == question.expected_section_id
        if doc_ok and section_ok:
            supported += 1

    return supported / len(response.citations)


def _answer_correctness(question: BatchQuestion, response: ResponseContractResult) -> float:
    expected = question.expected_answer
    if not expected:
        return 2.0 if response.abstained else 0.0

    if response.abstained:
        return 0.0

    expected_norm = _normalize_text(expected)
    answer_norm = _normalize_text(response.answer)
    if expected_norm and expected_norm in answer_norm:
        return 2.0

    expected_tokens = set(expected_norm.split())
    answer_tokens = set(answer_norm.split())
    if not expected_tokens:
        return 0.0

    overlap = len(expected_tokens & answer_tokens) / len(expected_tokens)
    if overlap >= 0.5:
        return 1.0
    return 0.0


def _abstention_correctness(question: BatchQuestion, response: ResponseContractResult) -> float:
    should_abstain = question.expected_abstain
    if should_abstain is None:
        should_abstain = question.expected_answer is None
    return 1.0 if response.abstained == should_abstain else 0.0


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()
