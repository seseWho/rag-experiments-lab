from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd


DATASET_ROOT = Path("datasets")
RUN_RECORD_ROOT = Path("experiments/run_records")


def _list_datasets() -> list[str]:
    if not DATASET_ROOT.exists():
        return []
    return sorted([item.name for item in DATASET_ROOT.iterdir() if item.is_dir()])


def _list_run_records(dataset_id: str) -> list[Path]:
    if not RUN_RECORD_ROOT.exists():
        return []

    records: list[Path] = []
    for record in RUN_RECORD_ROOT.glob("*/run_record.json"):
        payload = json.loads(record.read_text(encoding="utf-8"))
        configs = payload.get("configs", [])
        if any(cfg.get("config_snapshot", {}).get("dataset_id") == dataset_id for cfg in configs):
            records.append(record)
    return sorted(records, key=lambda p: p.parent.name, reverse=True)


def _load_latest_record(dataset_id: str) -> dict[str, object] | None:
    records = _list_run_records(dataset_id)
    if not records:
        return None
    return json.loads(records[0].read_text(encoding="utf-8"))


def _config_options(record: dict[str, object]) -> list[str]:
    return [cfg["config_id"] for cfg in record.get("configs", [])]


def _get_config(record: dict[str, object], config_id: str) -> dict[str, object]:
    for cfg in record.get("configs", []):
        if cfg.get("config_id") == config_id:
            return cfg
    raise ValueError(f"Config not found: {config_id}")


def _build_ab_config_table(record: dict[str, object]) -> pd.DataFrame:
    rows = []
    for cfg in record.get("configs", []):
        snap = cfg.get("config_snapshot", {})
        rows.append(
            {
                "config_id": cfg.get("config_id"),
                "dataset_id": snap.get("dataset_id"),
                "chunking_strategy": snap.get("chunking_strategy"),
                "chunk_size": snap.get("chunk_size"),
                "chunk_overlap": snap.get("chunk_overlap"),
                "top_k": snap.get("top_k"),
                "abstain_threshold": snap.get("abstain_threshold"),
            }
        )
    return pd.DataFrame(rows)


def _build_topk_table(record: dict[str, object], config_id: str, question_id: str) -> pd.DataFrame:
    cfg = _get_config(record, config_id)
    for item in cfg.get("questions", []):
        if item.get("question_id") != question_id:
            continue
        rows = []
        for trace in item.get("trace", []):
            rows.append(
                {
                    "chunk_id": trace.get("chunk_id"),
                    "score": trace.get("score"),
                    "doc_id": trace.get("doc_id"),
                    "section_id": trace.get("section_id"),
                }
            )
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["chunk_id", "score", "doc_id", "section_id"])


def _build_answer_view(record: dict[str, object], config_id: str, question_id: str) -> str:
    cfg = _get_config(record, config_id)
    for item in cfg.get("questions", []):
        if item.get("question_id") != question_id:
            continue
        response = item.get("response", {})
        citations = response.get("citations") or []
        citations_text = ", ".join(citations) if citations else "(none)"
        return (
            f"Question: {item.get('question')}\n\n"
            f"Answer: {response.get('answer')}\n"
            f"Citations: {citations_text}\n"
            f"Abstained: {response.get('abstained')}\n"
            f"Reason: {response.get('reason')}"
        )
    return "Question not found"


def _build_metrics_table(record: dict[str, object]) -> pd.DataFrame:
    rows = []
    for cfg in record.get("configs", []):
        summary = cfg.get("summary", {})
        row = {"config_id": cfg.get("config_id")}
        row.update(summary)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if len(frame) == 2:
        delta = {"config_id": "delta(A-B)"}
        numeric_cols = [col for col in frame.columns if col != "config_id"]
        for col in numeric_cols:
            delta[col] = frame.iloc[0][col] - frame.iloc[1][col]
        frame = pd.concat([frame, pd.DataFrame([delta])], ignore_index=True)
    return frame


def _examples_view(record: dict[str, object], config_a: str, config_b: str) -> pd.DataFrame:
    a = _get_config(record, config_a)
    b = _get_config(record, config_b)
    by_qid_a = {q["question_id"]: q for q in a.get("questions", [])}
    by_qid_b = {q["question_id"]: q for q in b.get("questions", [])}

    rows = []
    for qid in sorted(set(by_qid_a) & set(by_qid_b)):
        qa = by_qid_a[qid]
        qb = by_qid_b[qid]
        score_a = qa.get("response", {}).get("abstained")
        score_b = qb.get("response", {}).get("abstained")
        if score_a == score_b and qa.get("response", {}).get("answer") == qb.get("response", {}).get("answer"):
            continue
        rows.append(
            {
                "question_id": qid,
                "question": qa.get("question"),
                "a_answer": qa.get("response", {}).get("answer"),
                "b_answer": qb.get("response", {}).get("answer"),
                "a_abstained": score_a,
                "b_abstained": score_b,
            }
        )
    return pd.DataFrame(rows)


def _question_options(record: dict[str, object], config_id: str) -> list[str]:
    cfg = _get_config(record, config_id)
    return [item["question_id"] for item in cfg.get("questions", [])]


def load_record_state(dataset_id: str):
    record = _load_latest_record(dataset_id)
    if not record:
        empty = pd.DataFrame()
        msg = f"No run record found for dataset: {dataset_id}"
        return None, msg, empty, gr.update(choices=[]), gr.update(choices=[]), empty, "", empty, empty

    config_ids = _config_options(record)
    first_config = config_ids[0]
    question_ids = _question_options(record, first_config)
    first_qid = question_ids[0] if question_ids else ""

    return (
        record,
        f"Loaded run_id={record.get('run_id')} ({record.get('question_count')} questions)",
        _build_ab_config_table(record),
        gr.update(choices=config_ids, value=first_config),
        gr.update(choices=question_ids, value=first_qid),
        _build_topk_table(record, first_config, first_qid),
        _build_answer_view(record, first_config, first_qid),
        _build_metrics_table(record),
        _examples_view(record, config_ids[0], config_ids[1]) if len(config_ids) >= 2 else pd.DataFrame(),
    )


def on_config_change(record: dict[str, object], config_id: str):
    if not record or not config_id:
        return gr.update(choices=[]), pd.DataFrame(), ""
    question_ids = _question_options(record, config_id)
    qid = question_ids[0] if question_ids else ""
    return (
        gr.update(choices=question_ids, value=qid),
        _build_topk_table(record, config_id, qid),
        _build_answer_view(record, config_id, qid),
    )


def on_question_change(record: dict[str, object], config_id: str, question_id: str):
    if not record or not config_id or not question_id:
        return pd.DataFrame(), ""
    return _build_topk_table(record, config_id, question_id), _build_answer_view(record, config_id, question_id)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG Lab Panel") as demo:
        gr.Markdown("# RAG Lab Panel\nInspección rápida: configuración A/B, trazas top-k, respuesta/citas y métricas.")
        state_record = gr.State(value=None)

        with gr.Row():
            dataset = gr.Dropdown(choices=_list_datasets(), label="Dataset", value=_list_datasets()[0] if _list_datasets() else None)
            load_btn = gr.Button("Load latest run")

        status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("A/B Config + Dataset"):
            config_table = gr.Dataframe(label="Config snapshot")

        with gr.Tab("Top-k chunks + scores + metadata"):
            config_selector = gr.Dropdown(label="Config")
            question_selector = gr.Dropdown(label="Question ID")
            topk_table = gr.Dataframe(label="Top-k trace")

        with gr.Tab("Respuesta + citas + abstención"):
            answer_box = gr.Textbox(lines=10, label="Response inspection")

        with gr.Tab("Métricas + delta A/B + ejemplos"):
            metrics_table = gr.Dataframe(label="Summary + delta")
            examples_table = gr.Dataframe(label="Differing examples")

        load_btn.click(
            fn=load_record_state,
            inputs=[dataset],
            outputs=[state_record, status, config_table, config_selector, question_selector, topk_table, answer_box, metrics_table, examples_table],
        )

        config_selector.change(
            fn=on_config_change,
            inputs=[state_record, config_selector],
            outputs=[question_selector, topk_table, answer_box],
        )

        question_selector.change(
            fn=on_question_change,
            inputs=[state_record, config_selector, question_selector],
            outputs=[topk_table, answer_box],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
