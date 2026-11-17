"""Load chunks, generate embeddings, and store them in a persistent ChromaDB collection.

Usage:
    python build_index.py [chunks.json]

Chunks are JSON objects with: id, text, source, language, topic, crops.
Re-running is safe: existing ids are upserted, not duplicated.
"""

import json
import sys
from pathlib import Path

import chromadb

from embedder import FarmEmbeddingFunction

BASE = Path(__file__).parent
CHROMA_DIR = BASE / "chroma_store"
COLLECTION = "agri_kb"
BATCH = 64


def main(chunk_file: Path) -> None:
    chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {chunk_file.name}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=FarmEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "source": c.get("source", ""),
                "language": c.get("language", ""),
                "topic": c.get("topic", ""),
                "crops": ",".join(c.get("crops", [])),
            } for c in batch],
        )

    print(f"Collection '{COLLECTION}' now holds {collection.count()} vectors at {CHROMA_DIR}")


if __name__ == "__main__":
    default = BASE / "data" / "sample_chunks.json"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
