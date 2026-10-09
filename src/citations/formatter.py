"""
formatter.py — Format retrieved docs into a context string and citation list.
"""
from langchain_core.documents import Document


def format_context(docs: list) -> str:
    """Build numbered context block for the LLM prompt."""
    parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        rerank_score = d.metadata.get("rerank_score", "")
        score_tag = f" [score: {rerank_score}]" if rerank_score != "" else ""
        parts.append(f"[{i}] [{source}, p.{page}]{score_tag}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def extract_sources(docs: list) -> list:
    """Unique sorted source filenames from retrieved docs."""
    return sorted({d.metadata.get("source", "unknown") for d in docs})
