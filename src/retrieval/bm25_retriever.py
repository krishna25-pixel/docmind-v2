"""
bm25_retriever.py — Sparse keyword retrieval using BM25 (rank_bm25).

BM25 corpus is bootstrapped from Pinecone: all vector metadata is fetched
(in batches) to reconstruct the document texts in memory.
"""
from __future__ import annotations
import os
import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from pinecone import Pinecone
from config import PINECONE_INDEX, PINECONE_NAMESPACE


def _tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())


def _fetch_all_from_pinecone() -> list[Document]:
    """
    Fetch every stored vector's text + metadata from Pinecone.
    Uses list() → fetch() in batches of 100 (free-tier safe).
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX)

    docs = []
    # list() returns paginated ID batches
    for id_batch in index.list(namespace=PINECONE_NAMESPACE):
        response = index.fetch(ids=id_batch, namespace=PINECONE_NAMESPACE)
        for _id, vec in response.vectors.items():
            meta = vec.metadata or {}
            text = meta.pop("text", "")  # langchain-pinecone stores text as "text"
            if text:
                docs.append(Document(page_content=text, metadata=meta))
    return docs


class BM25Retriever:
    """Thin BM25 wrapper that mirrors the .invoke() interface."""

    def __init__(self, docs: list, k: int = 6):
        self.docs = docs
        self.k = k
        corpus = [_tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_pinecone(cls, k: int = 6) -> "BM25Retriever":
        """Bootstrap BM25 corpus by fetching all docs from Pinecone."""
        docs = _fetch_all_from_pinecone()
        return cls(docs, k=k)

    def invoke(self, query: str) -> list:
        if not self.bm25 or not self.docs:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [self.docs[i] for i in top_idx]
