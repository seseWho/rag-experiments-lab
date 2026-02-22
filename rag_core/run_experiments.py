from __future__ import annotations

import argparse

from .embeddings import build_openai_embeddings
from .experiment_runner import BatchRunConfig, ExperimentRunner, load_questions
from .llm_config import load_llm_config
from .pipeline import BasePipeline, PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B RAG experiments and persist run record")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--docs-path", required=True)
    parser.add_argument("--questions-path", required=True)
    parser.add_argument("--config-a-chunking-strategy", choices=["fixed_size", "by_headings"], required=True)
    parser.add_argument("--config-b-chunking-strategy", choices=["fixed_size", "by_headings"], required=True)
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score-threshold", type=float, default=0.06)
    args = parser.parse_args()

    llm_cfg = load_llm_config()
    if llm_cfg.provider != "openai":
        raise ValueError("This implementation currently supports only LLM_PROVIDER=openai")

    embedding_client = build_openai_embeddings(llm_cfg)

    def factory(cfg: PipelineConfig) -> BasePipeline:
        return BasePipeline(cfg, embedding_client=embedding_client)

    runner = ExperimentRunner(factory)

    cfg_a = BatchRunConfig(
        config_id="A",
        pipeline=PipelineConfig(
            dataset_id=args.dataset_id,
            docs_path=args.docs_path,
            chunking_strategy=args.config_a_chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            min_score_threshold=args.min_score_threshold,
        ),
    )
    cfg_b = BatchRunConfig(
        config_id="B",
        pipeline=PipelineConfig(
            dataset_id=args.dataset_id,
            docs_path=args.docs_path,
            chunking_strategy=args.config_b_chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            min_score_threshold=args.min_score_threshold,
        ),
    )

    questions = load_questions(args.questions_path)
    output = runner.run_ab(questions=questions, configs=[cfg_a, cfg_b], run_id=args.run_id)
    print(f"Saved run record to {output}")


if __name__ == "__main__":
    main()
