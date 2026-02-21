from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .models import Chunk, NormalizedSection


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    chunk_size: int = 350
    chunk_overlap: int = 40


def _deterministic_chunk_id(
    dataset_id: str,
    doc_id: str,
    section_id: str,
    strategy: str,
    idx: int,
) -> str:
    material = f"{dataset_id}|{doc_id}|{section_id}|{strategy}|{idx}"
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}_{section_id}_{strategy}_{idx:03d}_{digest}"


def chunk_fixed_size(sections: list[NormalizedSection], cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    size = max(cfg.chunk_size, 1)
    overlap = max(min(cfg.chunk_overlap, size - 1), 0)
    step = size - overlap

    for section in sections:
        text = section.content
        if not text:
            continue
        cursor = 0
        idx = 0
        while cursor < len(text):
            part = text[cursor : cursor + size]
            if not part.strip():
                break
            chunks.append(
                Chunk(
                    dataset_id=section.dataset_id,
                    doc_id=section.doc_id,
                    version=section.version,
                    section_id=section.section_id,
                    chunk_id=_deterministic_chunk_id(
                        section.dataset_id,
                        section.doc_id,
                        section.section_id,
                        cfg.strategy,
                        idx,
                    ),
                    text=part,
                    metadata={
                        **section.metadata,
                        "heading": section.heading,
                        "chunk_strategy": cfg.strategy,
                        "chunk_index": idx,
                    },
                )
            )
            cursor += step
            idx += 1
    return chunks


def chunk_by_headings(sections: list[NormalizedSection], cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, section in enumerate(sections):
        text = f"{section.heading}\n{section.content}".strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                dataset_id=section.dataset_id,
                doc_id=section.doc_id,
                version=section.version,
                section_id=section.section_id,
                chunk_id=_deterministic_chunk_id(
                    section.dataset_id,
                    section.doc_id,
                    section.section_id,
                    cfg.strategy,
                    idx,
                ),
                text=text,
                metadata={
                    **section.metadata,
                    "heading": section.heading,
                    "chunk_strategy": cfg.strategy,
                    "chunk_index": idx,
                },
            )
        )
    return chunks


def chunk_sections(sections: list[NormalizedSection], cfg: ChunkingConfig) -> list[Chunk]:
    if cfg.strategy == "fixed_size":
        return chunk_fixed_size(sections, cfg)
    if cfg.strategy == "by_headings":
        return chunk_by_headings(sections, cfg)
    raise ValueError(f"Unsupported strategy: {cfg.strategy}")
