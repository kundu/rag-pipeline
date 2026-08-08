"""Deterministic character-window chunking. No LLM involvement.

Chunk boundaries are a pure function of (document bytes, policy chunk size,
policy overlap): files are processed in sorted filename order and split into
fixed-size windows, so the same inputs always produce byte-identical
chunks.json output.
"""
from __future__ import annotations

import json
from pathlib import Path


def chunk_documents(
    documents_dir: Path, chunk_size: int, overlap: int
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap_chars must be >= 0 and < chunk_size_chars")

    chunks: list[dict] = []
    for doc_path in sorted(documents_dir.glob("*.txt")):
        text = doc_path.read_text(encoding="utf-8")
        n = len(text)
        start = 0
        i = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(
                {
                    "chunk_id": f"{doc_path.stem}::c{i:04d}",
                    "document_name": doc_path.name,
                    "start_char": start,
                    "end_char": end,
                    "text": text[start:end],
                }
            )
            if end >= n:
                break
            start = end - overlap
            i += 1
    return chunks


def write_chunks(chunks: list[dict], out_path: Path) -> None:
    out_path.write_text(
        json.dumps(chunks, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
