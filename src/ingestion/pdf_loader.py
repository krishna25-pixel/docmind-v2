"""
pdf_loader.py — Load PDFs using PyPDFLoader; tag source metadata.
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(paths: list) -> list:
    """Load one or more PDFs. Returns list of LangChain Documents."""
    docs = []
    for path in paths:
        path = Path(path)
        loader = PyPDFLoader(str(path))
        file_docs = loader.load()
        for d in file_docs:
            d.metadata["source"] = path.name
            d.metadata.setdefault("page", d.metadata.get("page", 0))
        docs.extend(file_docs)
    return docs
