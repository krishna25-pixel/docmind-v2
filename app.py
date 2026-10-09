"""
app.py — DocMind V2 Streamlit UI.

Full single-page UI without sidebar:
  • Settings popover (⚙️) on top right with Debug panel toggle & Clear conversation
  • File upload (+) popover just above/next to the chat input to upload & auto-index PDFs
  • Hybrid retrieval (vector + BM25) fused with RRF
  • Cross-encoder reranking
  • Conversation memory
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.ingestion.pdf_loader import load_pdfs
from src.chunking.chunker import chunk_documents
from src.retrieval.vector_store import build_vectorstore, index_has_vectors
from src.pipeline.rag_pipeline import RAGPipeline
from src.citations.formatter import format_context

load_dotenv()

# Bridge Streamlit Cloud secrets into os.environ
try:
    if hasattr(st, "secrets"):
        for key in ["GROQ_API_KEY", "PINECONE_API_KEY"]:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key]).strip()
except Exception:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

st.set_page_config(
    page_title="DocMind",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0A0E16;
    --surface: #121826;
    --surface-2: #171F30;
    --border: rgba(201, 162, 75, 0.16);
    --border-strong: rgba(201, 162, 75, 0.45);
    --gold: #C9A24B;
    --gold-bright: #E4C468;
    --text: #EDEEF2;
    --text-muted: #8B93A7;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide Streamlit top-right elements (Fork, GitHub icon, MainMenu) */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header [data-testid="stToolbar"] { display: none; visibility: hidden; }
[data-testid="stToolbar"] { display: none; visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stHeader"] { background: transparent; }

/* Completely hide sidebar */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 18% -10%, #151E33 0%, var(--bg) 55%);
}

.dm-hero { padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.dm-hero h1 { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 2.3rem; letter-spacing: -0.02em; color: var(--text); margin: 0; display: flex; align-items: center; gap: 0.6rem; }
.dm-hero .mark { color: var(--gold); font-size: 1.8rem; }
.dm-hero p { font-family: 'Inter', sans-serif; color: var(--text-muted); font-size: 0.98rem; margin: 0.4rem 0 0 0; }
.dm-rule { height: 1px; width: 64px; background: linear-gradient(90deg, var(--gold), transparent); margin-top: 0.8rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--gold-bright), var(--gold));
    color: #1A1200; font-family: 'Sora', sans-serif; font-weight: 600;
    border: none; border-radius: 8px; letter-spacing: 0.01em;
    transition: filter 0.15s ease, transform 0.15s ease;
}
.stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); }
.stButton > button:disabled { background: var(--surface-2); color: var(--text-muted); }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
}
[data-testid="stChatInput"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--border-strong); box-shadow: 0 0 0 3px rgba(201, 162, 75, 0.12);
}

[data-testid="stCaptionContainer"], .stCaption { color: var(--text-muted) !important; }
[data-testid="stExpander"] { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; }

.dm-empty {
    border: 1px dashed var(--border-strong); border-radius: 12px; padding: 2.5rem 2rem;
    text-align: center; color: var(--text-muted); font-family: 'Inter', sans-serif;
}
.dm-empty .mark { color: var(--gold); font-size: 1.6rem; display: block; margin-bottom: 0.6rem; }
.dm-notice {
    background: var(--surface-2); border: 1px solid var(--border-strong);
    border-radius: 10px; padding: 1rem 1.25rem; color: var(--text); font-size: 0.92rem;
}
.dm-latency {
    font-family: 'Inter', sans-serif; font-size: 0.78rem; color: var(--text-muted);
    margin-top: 0.3rem;
}

/* Plus button for upload near bottom */
.dm-upload-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = RAGPipeline(groq_api_key=GROQ_API_KEY)

pipeline: RAGPipeline = st.session_state["pipeline"]

# ---------------------------------------------------------------------------
# Header Row: Title & Top-Right Settings
# ---------------------------------------------------------------------------
col_title, col_settings = st.columns([0.88, 0.12], vertical_alignment="bottom")

with col_title:
    st.markdown("""
    <div class="dm-hero">
        <h1><span class="mark">◈</span> DocMind <span style="font-size:1rem;color:#8B93A7;font-weight:400;letter-spacing:0;">V2</span></h1>
        <p>Hybrid retrieval · Cross-encoder reranking · Conversation memory · Grounded answers</p>
        <div class="dm-rule"></div>
    </div>
    """, unsafe_allow_html=True)

with col_settings:
    with st.popover("⚙️ Settings", use_container_width=True):
        st.session_state["show_debug"] = st.toggle(
            "Show debug panel", value=st.session_state["show_debug"]
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            pipeline.history = []
            st.rerun()

if not GROQ_API_KEY or not PINECONE_API_KEY:
    missing = []
    if not GROQ_API_KEY:
        missing.append("<code>GROQ_API_KEY</code> — free at <a href='https://console.groq.com/keys' style='color:#C9A24B;'>console.groq.com</a>")
    if not PINECONE_API_KEY:
        missing.append("<code>PINECONE_API_KEY</code> — free at <a href='https://app.pinecone.io' style='color:#C9A24B;'>app.pinecone.io</a>")
    st.markdown(f"""
    <div class="dm-notice">
        <strong>Setup required.</strong> Add these keys to your <code>.env</code> file or Streamlit Secrets, then restart:<br><br>
        {'<br>'.join(f'• {m}' for m in missing)}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Chat Feed
# ---------------------------------------------------------------------------
has_index = index_has_vectors()

if not has_index and not st.session_state["messages"]:
    st.markdown("""
    <div class="dm-empty">
        <span class="mark">◈</span>
        Your library is empty. Click <strong>➕ Upload & Index PDF</strong> below to get started.
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))
        if st.session_state["show_debug"] and msg.get("debug"):
            dbg = msg["debug"]
            lat = dbg.get("latency_ms", {})
            st.markdown(
                f'<div class="dm-latency">'
                f'retrieve {lat.get("retrieval_ms","?")} ms · '
                f'rerank {lat.get("rerank_ms","?")} ms · '
                f'llm {lat.get("llm_ms","?")} ms'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Retrieved passages + rerank scores"):
                st.text(format_context(dbg["docs"]))

# ---------------------------------------------------------------------------
# Upload / Index popover near Chat Input
# ---------------------------------------------------------------------------
with st.container():
    col_upload, _ = st.columns([0.3, 0.7])
    with col_upload:
        with st.popover("➕ Upload & Index PDF", use_container_width=True):
            st.markdown("##### Add Research Papers")
            uploaded_files = st.file_uploader(
                "Upload PDFs", type=["pdf"], accept_multiple_files=True,
                label_visibility="collapsed"
            )
            if st.button("Index documents", type="primary", disabled=not uploaded_files, use_container_width=True):
                with st.spinner("Reading and indexing into Pinecone…"):
                    tmp_dir = tempfile.mkdtemp()
                    tmp_paths = []
                    for uf in uploaded_files:
                        path = os.path.join(tmp_dir, uf.name)
                        with open(path, "wb") as f:
                            f.write(uf.getbuffer())
                        tmp_paths.append(path)

                    docs = load_pdfs(tmp_paths)
                    chunks = chunk_documents(docs)
                    build_vectorstore(chunks)
                    pipeline.reload()

                st.success(f"Indexed {len(uploaded_files)} doc(s) · {len(chunks)} passages!")
                st.rerun()

# ---------------------------------------------------------------------------
# Chat Input & Response Handling
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question about your documents…")

if question:
    if not has_index:
        st.error("Upload and index at least one document first via ➕ Upload & Index PDF.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            result = pipeline.ask(question)

        st.markdown(result.answer)
        st.caption("Sources: " + ", ".join(result.sources))

        if st.session_state["show_debug"]:
            lat = result.latency_ms
            st.markdown(
                f'<div class="dm-latency">'
                f'retrieve {lat.get("retrieval_ms","?")} ms · '
                f'rerank {lat.get("rerank_ms","?")} ms · '
                f'llm {lat.get("llm_ms","?")} ms'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Retrieved passages + rerank scores"):
                st.text(format_context(result.docs))

    st.session_state["messages"].append({
        "role": "assistant",
        "content": result.answer,
        "sources": result.sources,
        "debug": {"latency_ms": result.latency_ms, "docs": result.docs},
    })
    st.rerun()
