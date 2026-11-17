"""Load the real agricultural knowledge base CSV chunks into ChromaDB.

Input: raw_kb/*.csv (chunked CSVs from the project's data collection workbook)
  - records CSV  (record_id, information, ...)  -> one chunk per record
  - FAQ CSV      (faq_id, question, answer, ...) -> one chunk per FAQ
  - sources CSV  (source_id, source_name, ...)   -> metadata only (not indexed)
  - gaps / categorisation CSVs                   -> project notes, not indexed

Each chunk keeps its source_id, URL, verification note and reliability note as
metadata, so answers can show where a fact came from and what still needs checking.

Usage:
    python build_real_index.py
"""

import csv
import json
from pathlib import Path

import chromadb

from embedder import FarmEmbeddingFunction

BASE = Path(__file__).parent
RAW = BASE / "raw_kb"
STORE = BASE / "chroma_store"
COLLECTION = "agri_kb"


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if any((v or "").strip() for v in r.values())], reader.fieldnames or []


def main() -> None:
    records, faqs, sources = [], [], {}
    skipped = []

    for path in sorted(RAW.glob("*.csv")):
        rows, header = read_rows(path)
        if "record_id" in header:
            records.extend(rows)
        elif "faq_id" in header:
            faqs.extend(rows)
        elif "source_id" in header and "source_name" in header:
            for r in rows:
                sources[r["source_id"]] = r
        else:
            skipped.append(path.name)

    print(f"Records: {len(records)}  FAQs: {len(faqs)}  Sources: {len(sources)}  Skipped (notes/summary): {skipped}")

    chunks = []
    for r in records:
        src = sources.get(r["source_id"], {})
        chunks.append({
            "id": r["record_id"],
            "text": f"{r['subtopic']}: {r['information']}",
            "meta": {
                "kind": "record",
                "topic": r["topic"],
                "category": r["category"],
                "crop_type": r["crop_type"],
                "language": r["language"],
                "source_id": r["source_id"],
                "source_name": r["source_name"],
                "source_url": r["source_url"],
                "document_date": r["document_date"],
                "verification_note": r["verification_note"],
                "reliability_note": src.get("reliability_note", ""),
            },
        })
    for q in faqs:
        src = sources.get(q["source_id"], {})
        chunks.append({
            "id": q["faq_id"],
            "text": f"Q: {q['question']} A: {q['answer']}",
            "meta": {
                "kind": "faq",
                "topic": q["category"],
                "category": q["category"],
                "crop_type": q["crop_type"],
                "language": "en",
                "source_id": q["source_id"],
                "source_name": src.get("source_name", ""),
                "source_url": q["source_url"],
                "document_date": "",
                "verification_note": "",
                "reliability_note": src.get("reliability_note", ""),
                "related_record_ids": q["related_record_ids"],
            },
        })

    client = chromadb.PersistentClient(path=str(STORE))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=FarmEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        b = chunks[i:i + BATCH]
        collection.add(
            ids=[c["id"] for c in b],
            documents=[c["text"] for c in b],
            metadatas=[c["meta"] for c in b],
        )

    (BASE / "kb_chunks_index.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Indexed {collection.count()} chunks into '{COLLECTION}' at {STORE}")


if __name__ == "__main__":
    main()
