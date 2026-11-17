"""Embedding function used by the index and the search test.

PRODUCTION CHOICE: a multilingual sentence-transformers model, e.g.
    intfloat/multilingual-e5-small   (small, fast, covers Hindi and Kannada)
    BAAI/bge-m3                      (larger, stronger on Indic languages)

This sandbox cannot reach Hugging Face, so the default here is an OFFLINE
stand-in: hashed character n-grams with L2 normalisation. It matches shared
words and word fragments across scripts, but it does NOT understand meaning
(e.g. "whitefly" vs "सफेद मक्खी" will not match). Use it only to test the
pipeline. Swap in the real model on your machine with USE_MODEL=1.
"""

import hashlib
import math
import os
import re

from chromadb import Documents, EmbeddingFunction, Embeddings

DIM = 2048
MODEL_NAME = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")


def _ngrams(text: str, n_min: int = 3, n_max: int = 5):
    text = re.sub(r"\s+", " ", text.lower().strip())
    tokens = re.findall(r"\w+", text)
    for tok in tokens:
        padded = f"<{tok}>"
        for n in range(n_min, n_max + 1):
            for i in range(max(1, len(padded) - n + 1)):
                yield padded[i:i + n]


def hashed_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for gram in _ngrams(text):
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        vec[h % DIM] += 1.0 if (h >> 1) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FarmEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma-compatible embedding function.

    With USE_MODEL=1 it loads a sentence-transformers model; otherwise it uses
    the offline hashed stand-in above.
    """

    def __init__(self):
        self._model = None
        if os.environ.get("USE_MODEL") == "1":
            from sentence_transformers import SentenceTransformer  # needs torch
            self._model = SentenceTransformer(MODEL_NAME)

    def name(self) -> str:
        return f"farm-{'st' if self._model else 'hashed'}"

    def __call__(self, input: Documents) -> Embeddings:
        if self._model is not None:
            # e5 models expect a "passage: " / "query: " prefix
            prefixed = [f"passage: {t}" for t in input] if "e5" in MODEL_NAME else list(input)
            return self._model.encode(prefixed, normalize_embeddings=True).tolist()
        return [hashed_embed(t) for t in input]

    def embed_query(self, input: Documents) -> Embeddings:
        if self._model is not None:
            qs = [f"query: {t}" if "e5" in MODEL_NAME else t for t in input]
            return self._model.encode(qs, normalize_embeddings=True).tolist()
        return [hashed_embed(t) for t in input]
