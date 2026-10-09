"""
vector_store.py — Build or load Pinecone vector store.

Pinecone free tier: 1 serverless index, ~100k vectors, no expiry.
Set PINECONE_API_KEY in .env (or Streamlit secrets).
"""
import os
import time
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.embeddings.embedder import get_embeddings
from config import PINECONE_INDEX, PINECONE_NAMESPACE, EMBEDDING_DIM


def _get_pinecone_client() -> Pinecone:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY missing. Set in .env or Streamlit secrets.")
    return Pinecone(api_key=api_key)


def _ensure_index(pc: Pinecone) -> None:
    """Create the index if it doesn't exist yet."""
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)


def build_vectorstore(chunks):
    """Embed chunks and upsert into Pinecone. Returns PineconeVectorStore."""
    pc = _get_pinecone_client()
    _ensure_index(pc)
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=PINECONE_INDEX,
        namespace=PINECONE_NAMESPACE,
    )
    return vectorstore


def load_vectorstore():
    """Load an existing Pinecone index as a LangChain vector store."""
    _get_pinecone_client()  # validates key early
    return PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=get_embeddings(),
        namespace=PINECONE_NAMESPACE,
    )


def index_has_vectors() -> bool:
    """Return True if the Pinecone index exists and has at least one vector."""
    try:
        pc = _get_pinecone_client()
        existing = [i.name for i in pc.list_indexes()]
        if PINECONE_INDEX not in existing:
            return False
        stats = pc.Index(PINECONE_INDEX).describe_index_stats()
        total = stats.get("total_vector_count", 0)
        return total > 0
    except Exception:
        return False
