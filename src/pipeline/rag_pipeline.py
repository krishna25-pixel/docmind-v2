"""
rag_pipeline.py — End-to-end RAG pipeline: hybrid retrieve → rerank → generate.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from langchain_core.documents import Document

from src.retrieval.vector_store import load_vectorstore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.reranker import rerank
from src.citations.formatter import format_context, extract_sources
from src.generation.llm import build_chain
from config import TOP_K, FINAL_K, MEMORY_WINDOW


@dataclass
class PipelineResult:
    answer: str
    sources: list
    docs: list
    history: list
    latency_ms: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Stateful pipeline with conversation memory.
    One instance per Streamlit session.
    """

    def __init__(self, groq_api_key=None):
        self._key = groq_api_key
        self._vectordb = None
        self._bm25 = None
        self._chain = None
        self.history: list = []  # [{role, content}, ...]

    # ── lazy init ────────────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._vectordb is None:
            self._vectordb = load_vectorstore()
        if self._bm25 is None:
            self._bm25 = BM25Retriever.from_pinecone(k=TOP_K)
        if self._chain is None:
            self._chain = build_chain(groq_api_key=self._key)

    def reload(self):
        """Call after new documents are indexed to refresh retrievers."""
        self._vectordb = None
        self._bm25 = None

    # ── history helpers ──────────────────────────────────────────────────────
    def _history_str(self) -> str:
        window = self.history[-MEMORY_WINDOW:]
        if not window:
            return "(none)"
        lines = []
        for m in window:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    # ── main entry ───────────────────────────────────────────────────────────
    def ask(self, question: str) -> PipelineResult:
        self._ensure_loaded()
        latency: dict = {}

        # 1. Hybrid retrieval
        t0 = time.perf_counter()
        vec_retriever = self._vectordb.as_retriever(search_kwargs={"k": TOP_K})
        vec_docs = vec_retriever.invoke(question)
        bm25_docs = self._bm25.invoke(question)
        fused = rrf_fuse([vec_docs, bm25_docs], final_k=FINAL_K * 2)
        latency["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 2. Cross-encoder rerank
        t0 = time.perf_counter()
        final_docs = rerank(question, fused, top_k=FINAL_K)
        latency["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 3. Generate
        t0 = time.perf_counter()
        context = format_context(final_docs)
        answer = self._chain.invoke({
            "context": context,
            "history": self._history_str(),
            "question": question,
        })
        latency["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 4. Update memory
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return PipelineResult(
            answer=answer,
            sources=extract_sources(final_docs),
            docs=final_docs,
            history=list(self.history),
            latency_ms=latency,
        )
