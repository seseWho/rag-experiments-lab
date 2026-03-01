from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from rag_core.build_dataset import parse_markdown_sections, write_dataset
from rag_core.validate_dataset import build_docs_index, validate_docs, validate_questions


DATASETS_ROOT = Path("datasets")


DocList = list[dict[str, Any]]
QuestionList = list[dict[str, Any]]


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _build_doc_from_text(doc_id: str, title: str, text: str, is_markdown: bool) -> dict[str, Any]:
    normalized = _normalize_text(text)
    if is_markdown:
        parsed = parse_markdown_sections(normalized)
        sections_source = parsed if parsed else [{"heading": title, "text": normalized}]
    else:
        sections_source = [{"heading": title, "text": normalized}]

    sections = [
        {
            "section_id": f"{doc_id}-S{idx}",
            "heading": section["heading"],
            "text": section["text"],
        }
        for idx, section in enumerate(sections_source, start=1)
    ]

    return {
        "doc_id": doc_id,
        "title": title,
        "doc_type": "",
        "version": "",
        "date": "",
        "sections": sections,
    }


def _build_docs_from_uploads(file_paths: list[str] | None) -> DocList:
    docs: DocList = []
    if not file_paths:
        return docs

    sorted_paths = sorted([Path(p) for p in file_paths], key=lambda p: p.name.lower())
    for idx, path in enumerate(sorted_paths, start=1):
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        docs.append(_build_doc_from_text(f"D{idx}", path.stem, text, is_markdown=(suffix == ".md")))

    return docs


def _docs_preview_choices(docs: DocList) -> tuple[list[str], str | None]:
    choices = [f"{doc['doc_id']} | {doc['title']}" for doc in docs]
    return choices, choices[0] if choices else None


def _sections_preview(doc: dict[str, Any]) -> str:
    lines: list[str] = []
    for section in doc.get("sections", []):
        text = str(section.get("text", ""))
        snippet = (text[:180] + "...") if len(text) > 180 else text
        lines.append(
            f"- {section.get('section_id')} | {section.get('heading')}\n  {snippet}"
        )
    return "\n".join(lines)


def _export_controls(dataset_id: str, docs: DocList) -> tuple[dict[str, Any], str]:
    if not dataset_id.strip():
        return gr.update(interactive=False), "Dataset ID is required before export."
    if not docs:
        return gr.update(interactive=False), "Add at least one document before export."
    return gr.update(interactive=True), "Ready to export dataset."


def ingest_uploaded_docs(dataset_id: str, file_paths: list[str] | None):
    docs = _build_docs_from_uploads(file_paths)
    choices, value = _docs_preview_choices(docs)
    doc_ids = [doc["doc_id"] for doc in docs]
    btn_update, status_msg = _export_controls(dataset_id, docs)

    return (
        docs,
        json.dumps(docs, indent=2, ensure_ascii=False),
        gr.Dropdown(choices=choices, value=value),
        _sections_preview(docs[0]) if docs else "",
        gr.Dropdown(choices=doc_ids, value=doc_ids[0] if doc_ids else None),
        gr.Dropdown(choices=[], value=None),
        btn_update,
        status_msg,
    )


def add_pasted_doc(dataset_id: str, current_docs: DocList, title: str, text: str, is_markdown: bool):
    if not title.strip() or not text.strip():
        btn_update, status_msg = _export_controls(dataset_id, current_docs)
        return (
            current_docs,
            json.dumps(current_docs, indent=2, ensure_ascii=False),
            gr.Dropdown(),
            "",
            gr.Dropdown(),
            gr.Dropdown(),
            btn_update,
            "Title and text are required to add a pasted document.",
        )

    docs = list(current_docs)
    doc_id = f"D{len(docs) + 1}"
    docs.append(_build_doc_from_text(doc_id, title.strip(), text, is_markdown=is_markdown))
    choices, value = _docs_preview_choices(docs)
    doc_ids = [doc["doc_id"] for doc in docs]
    btn_update, status_msg = _export_controls(dataset_id, docs)

    return (
        docs,
        json.dumps(docs, indent=2, ensure_ascii=False),
        gr.Dropdown(choices=choices, value=value),
        _sections_preview(docs[0]),
        gr.Dropdown(choices=doc_ids, value=doc_ids[0]),
        gr.Dropdown(choices=[s["section_id"] for s in docs[0]["sections"]], value=docs[0]["sections"][0]["section_id"]),
        btn_update,
        f"Added document '{title.strip()}'. {status_msg}",
    )


def update_doc_preview(selected_label: str, docs: DocList) -> str:
    if not selected_label or not docs:
        return ""
    selected_doc_id = selected_label.split("|", maxsplit=1)[0].strip()
    for doc in docs:
        if doc["doc_id"] == selected_doc_id:
            return _sections_preview(doc)
    return ""


def update_section_choices(selected_doc_id: str, docs: DocList) -> dict[str, Any]:
    if not selected_doc_id:
        return gr.Dropdown(choices=[], value=None)
    for doc in docs:
        if doc["doc_id"] == selected_doc_id:
            section_ids = [section["section_id"] for section in doc.get("sections", [])]
            return gr.Dropdown(choices=section_ids, value=section_ids[0] if section_ids else None)
    return gr.Dropdown(choices=[], value=None)


def add_question(
    questions: QuestionList,
    question_text: str,
    expected_answer: str,
    expected_doc_id: str | None,
    expected_section_id: str | None,
    unanswerable: bool,
):
    if not question_text.strip():
        return questions, json.dumps(questions, indent=2), "question_text is required."

    next_id = f"Q{len(questions) + 1}"
    question = {
        "question_id": next_id,
        "question_text": question_text.strip(),
    }
    if expected_answer.strip():
        question["expected_answer"] = expected_answer.strip()
    if expected_doc_id:
        question["expected_doc_id"] = expected_doc_id
    if expected_section_id:
        question["expected_section_id"] = expected_section_id
    if unanswerable:
        question["unanswerable"] = True

    updated = list(questions) + [question]
    return updated, json.dumps(updated, indent=2), f"Added question {next_id}."


def export_dataset(dataset_id: str, docs: DocList, questions: QuestionList) -> str:
    if not dataset_id.strip():
        return "FAIL\nDataset ID is required."
    if not docs:
        return "FAIL\nAt least one document is required."

    errors = validate_docs(docs)
    errors.extend(validate_questions(questions, build_docs_index(docs)))
    if errors:
        return "FAIL\n" + "\n".join(errors)

    out_dir = DATASETS_ROOT / dataset_id.strip()
    write_dataset(out_dir=out_dir, docs=docs, questions=questions)

    docs_errors = validate_docs(json.loads((out_dir / "docs.json").read_text(encoding="utf-8")))
    questions_errors = validate_questions(
        json.loads((out_dir / "questions.json").read_text(encoding="utf-8")),
        build_docs_index(json.loads((out_dir / "docs.json").read_text(encoding="utf-8"))),
    )
    post_errors = docs_errors + questions_errors
    if post_errors:
        return "FAIL\n" + "\n".join(post_errors)

    return f"PASS\nExported dataset to {out_dir}"


def on_dataset_id_change(dataset_id: str, docs: DocList):
    out_path = f"datasets/{dataset_id.strip()}" if dataset_id.strip() else "datasets/<dataset_id>"
    btn_update, status_msg = _export_controls(dataset_id, docs)
    return out_path, btn_update, status_msg


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Dataset Builder") as demo:
        gr.Markdown("# Dataset Builder\nCreate docs + questions manually and export to `datasets/<dataset_id>/`.")

        docs_state = gr.State(value=[])
        questions_state = gr.State(value=[])

        with gr.Row():
            dataset_id = gr.Textbox(label="dataset_id", placeholder="my_dataset_v1")
            output_dir = gr.Textbox(label="Output directory", value="datasets/<dataset_id>", interactive=False)

        with gr.Tab("1) Documents"):
            files_input = gr.File(file_count="multiple", file_types=[".txt", ".md"], type="filepath", label="Upload .txt/.md files")
            ingest_btn = gr.Button("Ingest uploaded files")

            pasted_title = gr.Textbox(label="Pasted doc title")
            pasted_text = gr.Textbox(label="Pasted text", lines=8)
            pasted_markdown = gr.Checkbox(label="Treat pasted text as markdown", value=True)
            add_pasted_btn = gr.Button("Add pasted document")

            docs_json_preview = gr.Textbox(label="Docs JSON preview", lines=10)
            doc_selector = gr.Dropdown(label="Preview doc")
            sections_preview = gr.Textbox(label="Sections preview", lines=10)

        with gr.Tab("2) Questions"):
            question_text = gr.Textbox(label="question_text (required)")
            expected_answer = gr.Textbox(label="expected_answer (optional)")
            expected_doc_id = gr.Dropdown(label="expected_doc_id (optional)")
            expected_section_id = gr.Dropdown(label="expected_section_id (optional)")
            unanswerable = gr.Checkbox(label="unanswerable", value=False)
            add_question_btn = gr.Button("Add question")
            question_status = gr.Textbox(label="Question editor status", interactive=False)
            questions_json_preview = gr.Textbox(label="Questions JSON preview", lines=10)

        with gr.Tab("3) Export"):
            export_btn = gr.Button("Export dataset", interactive=False)
            export_status = gr.Textbox(label="Export + validation status", lines=8, interactive=False)

        dataset_id.change(
            fn=on_dataset_id_change,
            inputs=[dataset_id, docs_state],
            outputs=[output_dir, export_btn, export_status],
        )

        ingest_btn.click(
            fn=ingest_uploaded_docs,
            inputs=[dataset_id, files_input],
            outputs=[docs_state, docs_json_preview, doc_selector, sections_preview, expected_doc_id, expected_section_id, export_btn, export_status],
        )

        add_pasted_btn.click(
            fn=add_pasted_doc,
            inputs=[dataset_id, docs_state, pasted_title, pasted_text, pasted_markdown],
            outputs=[docs_state, docs_json_preview, doc_selector, sections_preview, expected_doc_id, expected_section_id, export_btn, export_status],
        )

        doc_selector.change(fn=update_doc_preview, inputs=[doc_selector, docs_state], outputs=[sections_preview])

        expected_doc_id.change(fn=update_section_choices, inputs=[expected_doc_id, docs_state], outputs=[expected_section_id])

        add_question_btn.click(
            fn=add_question,
            inputs=[questions_state, question_text, expected_answer, expected_doc_id, expected_section_id, unanswerable],
            outputs=[questions_state, questions_json_preview, question_status],
        )

        export_btn.click(fn=export_dataset, inputs=[dataset_id, docs_state, questions_state], outputs=[export_status])

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    main()
