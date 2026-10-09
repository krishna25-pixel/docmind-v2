"""
config.py — Central configuration for DocMind V2.
Edit here to tune chunking, retrieval, and model settings.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

# ── Pinecone ───────────────────────────────────────────────────────────────
PINECONE_INDEX = "docmind"          # index name in your Pinecone project
PINECONE_NAMESPACE = "default"      # namespace within the index
EMBEDDING_DIM = 384                 # all-MiniLM-L6-v2 output dimension

# ── Embedding ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = 6          # candidates from each retriever before fusion
FINAL_K = 4        # passages passed to LLM after reranking

# Reciprocal Rank Fusion smoothing constant
RRF_K = 60

# ── Reranker ───────────────────────────────────────────────────────────────
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Generation ─────────────────────────────────────────────────────────────
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0

# ── Conversation memory ────────────────────────────────────────────────────
MEMORY_WINDOW = 6   # last N messages kept in context
