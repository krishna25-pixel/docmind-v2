"""
embedder.py — Shared embedding model instance (cached).
"""
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """Return a cached HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(model_name=model_name)
