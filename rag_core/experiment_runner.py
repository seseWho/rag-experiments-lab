from __future__ import annotations

import json
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

            for item in questions:
                response, retrieval_results = pipeline.query_with_results(item.question)
                if not response.abstained:
                    answered += 1

                if item.expected_doc_id and any(
                    r.metadata.get("doc_id") == item.expected_doc_id for r in retrieval_results
                ):
                    citation_hits += 1

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
