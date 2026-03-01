from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)


def list_source_files(input_path: str | Path) -> list[Path]:
    """Return deterministic source files (.txt/.md) sorted by filename.

    If *input_path* is a file, it must be .txt or .md and is returned as a
    single-item list.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError("Input file must be .txt or .md when not using JSON mode")
        return [path]

    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    return sorted(files, key=lambda p: p.name)


def parse_markdown_sections(text: str) -> list[dict[str, str]]:
    """Split markdown text into sections by heading lines (#..######).

    Returns a list of {"heading": str, "text": str} preserving content. If no
    headings are found, the caller can fallback to a single section.
    """

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []

    sections: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end]
        if body.startswith("\n"):
            body = body[1:]
        sections.append({"heading": heading, "text": body})
    return sections


def build_docs_json(files: list[Path], metadata: dict[str, str]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for doc_index, path in enumerate(files, start=1):
        doc_id = f"D{doc_index}"
        title = path.stem
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")

        if path.suffix.lower() == ".md":
            parsed_sections = parse_markdown_sections(text)
            if parsed_sections:
                sections_source = parsed_sections
            else:
                sections_source = [{"heading": title, "text": text}]
        else:
            sections_source = [{"heading": title, "text": text}]

        sections: list[dict[str, str]] = []
        for sec_index, section in enumerate(sections_source, start=1):
            sections.append(
                {
                    "section_id": f"{doc_id}-S{sec_index}",
                    "heading": section["heading"],
                    "text": section["text"],
                }
            )

        docs.append(
            {
                "doc_id": doc_id,
                "title": title,
                "doc_type": metadata.get("doc_type", ""),
                "version": metadata.get("version", ""),
                "date": metadata.get("date", ""),
                "sections": sections,
            }
        )
    return docs


def _coerce_simple_json_doc(doc: dict[str, Any], doc_id: str) -> dict[str, Any]:
    title = str(doc.get("title") or doc_id)
    doc_type = str(doc.get("doc_type") or "")
    version = str(doc.get("version") or "")
    date = str(doc.get("date") or "")

    if isinstance(doc.get("sections"), list):
        raw_sections = doc["sections"]
        sections = []
        for idx, raw in enumerate(raw_sections, start=1):
            heading = str(raw.get("heading") or title)
            section_text = str(raw.get("text") if raw.get("text") is not None else raw.get("content") or "")
            section_id = str(raw.get("section_id") or f"{doc_id}-S{idx}")
            sections.append({"section_id": section_id, "heading": heading, "text": section_text})
    else:
        section_text = str(doc.get("text") if doc.get("text") is not None else doc.get("content") or "")
        sections = [{"section_id": f"{doc_id}-S1", "heading": title, "text": section_text}]

    return {
        "doc_id": str(doc.get("doc_id") or doc_id),
        "title": title,
        "doc_type": doc_type,
        "version": version,
        "date": date,
        "sections": sections,
    }


def load_docs_from_json(input_file: str | Path, metadata: dict[str, str]) -> list[dict[str, Any]]:
    """Load docs from JSON input.

    Behavior:
    - If a document already resembles target schema (has 'sections' list of
      objects), preserve provided IDs and fill missing IDs deterministically.
    - Otherwise coerce each entry from a simple schema where doc-level text or
      section 'content' is accepted, converting to target schema.
    - CLI metadata defaults are only applied when a field is missing/empty.
    """

    payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of documents")

    docs: list[dict[str, Any]] = []
    for doc_index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each document in JSON input must be an object")

        default_doc_id = f"D{doc_index}"
        if isinstance(item.get("sections"), list):
            doc_id = str(item.get("doc_id") or default_doc_id)
            title = str(item.get("title") or doc_id)
            doc_type = str(item.get("doc_type") or metadata.get("doc_type", ""))
            version = str(item.get("version") or metadata.get("version", ""))
            date = str(item.get("date") or metadata.get("date", ""))

            sections = []
            for sec_index, section in enumerate(item["sections"], start=1):
                if not isinstance(section, dict):
                    raise ValueError("Section entries must be objects")
                section_text = section.get("text")
                if section_text is None:
                    section_text = section.get("content", "")
                sections.append(
                    {
                        "section_id": str(section.get("section_id") or f"{doc_id}-S{sec_index}"),
                        "heading": str(section.get("heading") or title),
                        "text": str(section_text),
                    }
                )
            doc = {
                "doc_id": doc_id,
                "title": title,
                "doc_type": doc_type,
                "version": version,
                "date": date,
                "sections": sections,
            }
        else:
            doc = _coerce_simple_json_doc(item, default_doc_id)
            if not doc.get("doc_type"):
                doc["doc_type"] = metadata.get("doc_type", "")
            if not doc.get("version"):
                doc["version"] = metadata.get("version", "")
            if not doc.get("date"):
                doc["date"] = metadata.get("date", "")

        for section in doc["sections"]:
            section["text"] = str(section["text"]).replace("\r\n", "\n").replace("\r", "\n")
        docs.append(doc)

    return docs


def write_dataset(out_dir: str | Path, docs: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = output_dir / "docs.json"
    questions_path = output_dir / "questions.json"
    readme_path = output_dir / "README.md"

    docs_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    questions_path.write_text(json.dumps(questions, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Dataset: {output_dir.name}

This dataset was generated by `python -m rag_core.build_dataset`.

## Files
- `docs.json`: document corpus used by retrieval pipelines.
- `questions.json`: evaluation questions (currently empty placeholder).

## Run experiments

```bash
python -m rag_core.run_experiments \\
  --run-id <run_id> \\
  --dataset-id {output_dir.name} \\
  --docs-path {docs_path.as_posix()} \\
  --questions-path {questions_path.as_posix()} \\
  --config-a-chunking-strategy fixed_size \\
  --config-b-chunking-strategy by_headings
```
"""
    readme_path.write_text(readme, encoding="utf-8")

    print(f"Created {docs_path}")
    print(f"Created {questions_path}")
    print(f"Created {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dataset folder from source files or JSON")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--doc-type", default="")
    parser.add_argument("--version", default="")
    parser.add_argument("--date", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("datasets") / args.dataset_id
    metadata = {"doc_type": args.doc_type, "version": args.version, "date": args.date}

    input_path = Path(args.input_path)
    if input_path.suffix.lower() == ".json":
        docs = load_docs_from_json(input_path, metadata)
    else:
        files = list_source_files(input_path)
        docs = build_docs_json(files, metadata)

    write_dataset(out_dir=out_dir, docs=docs, questions=[])
    print(f"Dataset '{args.dataset_id}' ready at {out_dir}")


if __name__ == "__main__":
    main()
