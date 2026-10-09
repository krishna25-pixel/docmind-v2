"""
ingest.py — V2 CLI entry point. Wraps src modules.

Usage:
    python ingest.py                    # ingest every PDF in ./data
    python ingest.py path/to/file.pdf   # ingest single file
"""
import os
import sys
import glob

from src.ingestion.pdf_loader import load_pdfs
from src.chunking.chunker import chunk_documents
from src.retrieval.vector_store import build_vectorstore
from config import DATA_DIR


def main():
    if len(sys.argv) > 1:
        pdf_paths = sys.argv[1:]
    else:
        pdf_paths = sorted(glob.glob(str(DATA_DIR / "*.pdf")))

    if not pdf_paths:
        print(f"No PDFs found. Put files in ./{DATA_DIR}/ or pass a path.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s).")
    docs = load_pdfs(pdf_paths)
    print(f"Loaded {len(docs)} page(s).")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")

    build_vectorstore(chunks)
    print("Done. Vector store persisted.")


if __name__ == "__main__":
    main()
