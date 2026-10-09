"""
hybrid.py — Reciprocal Rank Fusion (RRF) over multiple ranked lists.
"""
from __future__ import annotations
from langchain_core.documents import Document
from config import RRF_K


def rrf_fuse(
    ranked_lists: list,
    k: int = RRF_K,
    final_k: int = 4,
) -> list:
    """
    Fuse N ranked lists with Reciprocal Rank Fusion.
    Returns `final_k` unique Documents ordered by fused score.
    """
    scores: dict = {}
    doc_map: dict = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content[:120]  # dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ordered[:final_k]]
