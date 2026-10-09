# DocMind V2 — Hybrid RAG Document Q&A

Ask questions over your own PDF documents and get grounded, source-cited answers —
with hallucination guardrails, hybrid retrieval, cross-encoder reranking, and
conversation memory. All running on free-tier tools.

## What's new in V2

| Feature | V1 | V2 |
|---|---|---|
| Retrieval | Vector-only (top-4) | **Hybrid: Vector + BM25 → RRF fusion** |
| Reranking | None | **Cross-encoder (ms-marco-MiniLM-L-6-v2)** |
| Memory | None | **Sliding-window conversation history** |
| Debug panel | Retrieved chunks | **Latencies + rerank scores + passages** |
| Architecture | Monolithic | **Modular `src/` packages** |

## Architecture

```
PDFs → pdf_loader → chunker → embedder → ChromaDB (persist)
                                 ↓
User query ──┬── Vector retriever (top-6)  ──┐
             └── BM25 retriever   (top-6)  ──┴── RRF fusion
                                                      ↓
                                           Cross-encoder rerank (top-4)
                                                      ↓
                                         Grounded prompt + history
                                                      ↓
                                            Groq LLM → Answer + citations
```

## Project structure

```
docmind/
├── app.py                      # Streamlit UI (V2)
├── ingest.py                   # CLI ingestion entry point
├── rag_chain.py                # CLI Q&A entry point (backward-compat shim)
├── config.py                   # Central config (chunk size, TOP_K, models…)
├── requirements.txt
├── .env                        # GROQ_API_KEY=...
├── data/                       # Drop PDFs here for CLI ingestion
├── chroma_store/               # Auto-created vector store
└── src/
    ├── ingestion/
    │   └── pdf_loader.py       # PyPDF → LangChain Documents
    ├── chunking/
    │   └── chunker.py          # RecursiveCharacterTextSplitter
    ├── embeddings/
    │   └── embedder.py         # Cached HuggingFace embedder
    ├── retrieval/
    │   ├── vector_store.py     # Chroma build/load
    │   ├── bm25_retriever.py   # Sparse BM25 (rank_bm25)
    │   ├── hybrid.py           # Reciprocal Rank Fusion
    │   └── reranker.py         # CrossEncoder reranker
    ├── generation/
    │   ├── prompts.py          # System prompt template
    │   └── llm.py              # Groq chain builder
    ├── citations/
    │   └── formatter.py        # Context string + source extraction
    └── pipeline/
        └── rag_pipeline.py     # Stateful end-to-end pipeline (memory)
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key:** https://console.groq.com/keys

3. **Create `.env`:**
   ```
   GROQ_API_KEY=your_key_here
   ```

## Usage

### Web UI (recommended)
```bash
streamlit run app.py
```
Upload PDFs → **Index documents** → ask questions.  
Toggle **Show debug panel** in the sidebar to see latencies and rerank scores.

### CLI
```bash
python ingest.py                          # ingest all PDFs in ./data
python ingest.py path/to/file.pdf         # ingest single file
python rag_chain.py "What is this about?" # one-shot question
```

## Tuning (`config.py`)

| Setting | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 500 | Passage length in chars |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `TOP_K` | 6 | Candidates from each retriever |
| `FINAL_K` | 4 | Passages passed to LLM after reranking |
| `RRF_K` | 60 | RRF smoothing constant |
| `MEMORY_WINDOW` | 6 | Last N messages in conversation context |
| `LLM_MODEL` | openai/gpt-oss-20b | Groq model to use |
