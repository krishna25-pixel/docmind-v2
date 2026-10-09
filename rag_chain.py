"""
rag_chain.py — V2 backward-compat shim.
Kept for CLI usage: python rag_chain.py "your question"
"""
import sys
from src.pipeline.rag_pipeline import RAGPipeline


def answer_question(question: str, groq_api_key=None):
    pipeline = RAGPipeline(groq_api_key=groq_api_key)
    result = pipeline.ask(question)
    return result.answer, result.sources, result.docs


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    answer, sources, _ = answer_question(q)
    print("\nANSWER:\n", answer)
    print("\nSOURCES:", sources)
