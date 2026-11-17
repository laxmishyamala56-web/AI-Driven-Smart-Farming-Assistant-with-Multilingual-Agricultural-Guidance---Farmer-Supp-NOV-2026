"""Run test queries against the ChromaDB index and print the top matches.

Usage:
    python search_test.py
"""

from pathlib import Path

import chromadb

from embedder import FarmEmbeddingFunction

BASE = Path(__file__).parent
COLLECTION = "agri_kb"

# (query, language of query, what a good top hit should be about)
TESTS = [
    ("how do I control whitefly on cotton", "en", "pest_control"),
    ("मिट्टी की जांच कैसे कराएं", "hi", "soil_health"),
    ("ರಾಗಿ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ", "kn", "market_price"),
    ("PM-KISAN income support instalments", "en", "subsidy_scheme"),
]


def main() -> None:
    client = chromadb.PersistentClient(path=str(BASE / "chroma_store"))
    collection = client.get_collection(name=COLLECTION, embedding_function=FarmEmbeddingFunction())
    print(f"Index holds {collection.count()} vectors\n")

    passed = 0
    for query, lang, expected_topic in TESTS:
        res = collection.query(query_texts=[query], n_results=3)
        print(f"Q [{lang}] {query}")
        for rank, (doc_id, dist, meta, doc) in enumerate(zip(
            res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]
        ), start=1):
            print(f"  {rank}. {doc_id}  topic={meta['topic']}  lang={meta['language']}  dist={dist:.3f}")
            print(f"     {doc[:90]}...")
        top_topic = res["metadatas"][0][0]["topic"]
        ok = top_topic == expected_topic
        passed += ok
        print(f"  -> top topic '{top_topic}' expected '{expected_topic}': {'PASS' if ok else 'FAIL'}\n")

    print(f"{passed}/{len(TESTS)} queries returned the expected topic at rank 1")


if __name__ == "__main__":
    main()
