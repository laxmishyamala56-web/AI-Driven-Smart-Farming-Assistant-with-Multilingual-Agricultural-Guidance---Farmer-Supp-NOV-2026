"""Test searches against the real knowledge base index.

Each test has a question and the record or FAQ id that should rank in the top 3.
Usage:
    python search_real.py
"""

from pathlib import Path

import chromadb

from embedder import FarmEmbeddingFunction

BASE = Path(__file__).parent

TESTS = [
    ("How much does PM-KISAN pay each year?", ["Q005", "R007"]),
    ("What is the Kisan Call Centre helpline number?", ["Q002", "R002"]),
    ("When do I get a new Soil Health Card?", ["Q019", "R023"]),
    ("When should I spray for brown planthoppers in rice?", ["Q022", "R044"]),
    ("How do I pay crop insurance premium under PMFBY?", ["Q013", "R017"]),
    ("Where can I check daily mandi prices?", ["Q017", "R025"]),
    ("When should I set up pheromone traps for pink bollworm in cotton?", ["Q023", "R046"]),
    ("Can a government employee get PM-KISAN?", ["Q007", "R009"]),
]


def main() -> None:
    client = chromadb.PersistentClient(path=str(BASE / "chroma_store"))
    collection = client.get_collection(name="agri_kb", embedding_function=FarmEmbeddingFunction())
    print(f"Index holds {collection.count()} chunks\n")

    hits = 0
    for query, expected in TESTS:
        res = collection.query(query_texts=[query], n_results=3)
        ids = res["ids"][0]
        found = any(e in ids for e in expected)
        hits += found
        print(f"Q: {query}")
        for rank, (cid, dist, meta, doc) in enumerate(zip(ids, res["distances"][0], res["metadatas"][0], res["documents"][0]), 1):
            flag = " <- expected" if cid in expected else ""
            print(f"  {rank}. {cid} [{meta['kind']}, {meta['source_id']}] dist={dist:.3f}{flag}")
            print(f"     {doc[:110]}")
        print(f"  -> {'PASS' if found else 'FAIL'}\n")

    print(f"{hits}/{len(TESTS)} queries had an expected chunk in the top 3")


if __name__ == "__main__":
    main()
