from __future__ import annotations

import argparse
from pathlib import Path

from .llm_config import load_llm_config
from .pipeline import BasePipeline, PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run base RAG pipeline")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--docs-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--chunking-strategy", choices=["fixed_size", "by_headings"], default="fixed_size")
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    llm_cfg = load_llm_config()

    cfg = PipelineConfig(
        dataset_id=args.dataset_id,
        docs_path=args.docs_path,
        chunking_strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    pipeline = BasePipeline(cfg)

    index_file = Path(cfg.index_root) / (
        f"{cfg.dataset_id}__{cfg.chunking_strategy}_s{cfg.chunk_size}_o{cfg.chunk_overlap}"
    ) / "index.json"
    if not index_file.exists():
        built = pipeline.build_index()
        print(f"Built index with {built} chunks at {index_file}")

    print(f"LLM provider configured via .env: {llm_cfg.provider} ({llm_cfg.model})")

    response = pipeline.query(args.question)
    print("Answer:", response.answer)
    print("Citations:", response.citations)
    print("Abstained:", response.abstained)


if __name__ == "__main__":
    main()
