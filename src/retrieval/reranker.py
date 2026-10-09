"""
reranker.py — Cross-encoder reranking of retrieved passages.
"""
from __future__ import annotations
from functools import lru_cache
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from config import RERANK_MODEL


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str = RERANK_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(query: str, docs: list, top_k: int = 4) -> list:
    """
    Score each (query, passage) pair with a cross-encoder.
    Returns docs sorted descending by relevance score, top_k only.
    """
    if not docs:
        return docs
    encoder = _get_cross_encoder()
    pairs = [(query, d.page_content) for d in docs]
    scores = encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    result = []
    for doc, score in ranked[:top_k]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        result.append(doc)
    return result
