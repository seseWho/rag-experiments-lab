from rag_core.chunking import ChunkingConfig, chunk_sections
from rag_core.ingestion import load_and_normalize_docs
from rag_core.pipeline import BasePipeline, PipelineConfig


def test_chunking_strategies_produce_deterministic_chunk_ids():
    sections = load_and_normalize_docs(
        "dataset3_hierarchical_manual",
        "datasets/dataset3_hierarchical_manual/docs.json",
    )
    fixed_chunks = chunk_sections(sections, ChunkingConfig(strategy="fixed_size", chunk_size=120, chunk_overlap=20))
    heading_chunks = chunk_sections(sections, ChunkingConfig(strategy="by_headings"))

    assert fixed_chunks
    assert heading_chunks
    assert fixed_chunks[0].chunk_id == chunk_sections(
        sections, ChunkingConfig(strategy="fixed_size", chunk_size=120, chunk_overlap=20)
    )[0].chunk_id


def test_end_to_end_pipeline_returns_citations_and_traces(tmp_path):
    cfg = PipelineConfig(
        dataset_id="dataset1_regulations_versions",
        docs_path="datasets/dataset1_regulations_versions/docs.json",
        chunking_strategy="fixed_size",
        chunk_size=180,
        chunk_overlap=30,
        top_k=3,
        index_root=str(tmp_path / "indexes"),
        traces_root=str(tmp_path / "traces"),
    )
    pipeline = BasePipeline(cfg)
    built = pipeline.build_index()

    assert built > 0

    response = pipeline.query("What changed in remote work policy for 2025?")
    assert isinstance(response.citations, list)
    assert response.abstained is False or response.reason == "top_score_below_threshold"

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    assert trace_files[0].read_text(encoding="utf-8").strip()
