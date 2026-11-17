"""Chunk usable Milestone 1 documents into JSONL for a vector store.

Usage:
    python3 -I chunking.py milestone1_combined_fixed.json chunks.jsonl

Rules:
- Only documents with status "usable" (and included_in_index true) are chunked.
- Text is split on blank lines into paragraphs, packed up to MAX_CHARS.
- Paragraphs longer than MAX_CHARS are split on sentence ends, then hard-cut.
- Consecutive chunks share OVERLAP_CHARS of trailing text, so a fact at a boundary is not lost.
- Every chunk carries the full document metadata plus its position (chunk_index, chunk_count).
"""
import json
import re
import sys

MAX_CHARS = 800
OVERLAP_CHARS = 100

META_KEYS = [
    "doc_id", "title", "topic", "category", "crop_type", "subtopic",
    "publisher", "publisher_type", "source_url", "date_published",
    "date_accessed", "status", "content_basis", "quality_notes", "language",
]


def split_long(text):
    """Split a too-long paragraph on sentence ends, then hard-cut anything still too long."""
    sentences = re.split(r"(?<=[.;:!?])\s+", text)
    pieces, buf = [], ""
    for sent in sentences:
        if len(sent) > MAX_CHARS:
            if buf:
                pieces.append(buf)
                buf = ""
            for i in range(0, len(sent), MAX_CHARS):
                pieces.append(sent[i:i + MAX_CHARS])
            continue
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) <= MAX_CHARS:
            buf = candidate
        else:
            pieces.append(buf)
            buf = sent
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units = []
    for p in paragraphs:
        units.extend([p] if len(p) <= MAX_CHARS else split_long(p))

    chunks, buf = [], ""
    for u in units:
        candidate = f"{buf}\n\n{u}" if buf else u
        if len(candidate) <= MAX_CHARS:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            tail = buf[-OVERLAP_CHARS:] if buf else ""
            buf = f"{tail}\n\n{u}".strip() if tail else u
            if len(buf) > MAX_CHARS:
                buf = u
    if buf:
        chunks.append(buf)
    return chunks


def main(src, dst):
    data = json.load(open(src, encoding="utf-8"))
    docs = data["sections"]["2_documents_and_faqs"]["documents"]
    written = 0
    with open(dst, "w", encoding="utf-8") as out:
        for doc in docs:
            meta = doc["metadata"]
            if meta.get("status") != "usable" or not meta.get("included_in_index"):
                continue
            pieces = chunk_text(doc["text"])
            for i, piece in enumerate(pieces):
                record = {
                    "chunk_id": f'{meta["doc_id"]}_c{i + 1:02d}',
                    "text": piece,
                    "metadata": {k: meta.get(k) for k in META_KEYS}
                    | {"chunk_index": i + 1, "chunk_count": len(pieces),
                       "text_file": meta.get("text_file")},
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} chunks to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
