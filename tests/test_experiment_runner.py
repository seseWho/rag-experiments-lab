import json

from rag_core.embeddings import DeterministicTestEmbeddings
from rag_core.experiment_runner import BatchRunConfig, ExperimentRunner, load_questions
from rag_core.pipeline import BasePipeline, PipelineConfig


def test_ab_run_record_persists_snapshots_traces_and_metrics(tmp_path):
    questions = load_questions("datasets/dataset1_regulations_versions/questions.json")[:3]

    def factory(cfg: PipelineConfig) -> BasePipeline:
        return BasePipeline(cfg, embedding_client=DeterministicTestEmbeddings())

    runner = ExperimentRunner(factory)

    cfg_a = BatchRunConfig(
        config_id="A",
        pipeline=PipelineConfig(
            dataset_id="dataset1_regulations_versions",
            docs_path="datasets/dataset1_regulations_versions/docs.json",
            chunking_strategy="fixed_size",
            chunk_size=180,
            chunk_overlap=30,
            top_k=3,
            index_root=str(tmp_path / "indexes"),
            traces_root=str(tmp_path / "traces"),
        ),
    )
    cfg_b = BatchRunConfig(
        config_id="B",
        pipeline=PipelineConfig(
            dataset_id="dataset1_regulations_versions",
            docs_path="datasets/dataset1_regulations_versions/docs.json",
            chunking_strategy="by_headings",
            top_k=3,
            index_root=str(tmp_path / "indexes"),
            traces_root=str(tmp_path / "traces"),
        ),
    )

    record_file = runner.run_ab(
        questions=questions,
        configs=[cfg_a, cfg_b],
        run_id="test_run_001",
        output_root=str(tmp_path / "run_records"),
    )

    payload = json.loads(record_file.read_text(encoding="utf-8"))

    assert payload["run_id"] == "test_run_001"
    assert payload["question_count"] == 3
    assert len(payload["configs"]) == 2

    first_cfg = payload["configs"][0]
    assert first_cfg["config_snapshot"]["chunking_strategy"] == "fixed_size"
    assert first_cfg["summary"]["total_questions"] == 3
    assert "citation_hit_rate" in first_cfg["summary"]
    assert "evidence_recall_at_k" in first_cfg["summary"]
    assert "citation_precision" in first_cfg["summary"]
    assert "answer_correctness" in first_cfg["summary"]
    assert "abstention_correctness" in first_cfg["summary"]

    first_question = first_cfg["questions"][0]
    assert "trace" in first_question
    assert "response" in first_question
    assert first_question["response"]["context_sent_to_llm"] == first_question["response"]["citations"]

    trace_entry = first_question["trace"][0]
    assert "chunk_id" in trace_entry
    assert "score" in trace_entry
    assert "doc_id" in trace_entry
