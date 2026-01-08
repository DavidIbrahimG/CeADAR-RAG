import streamlit as st
import subprocess
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

from rag.pipeline import answer

# Repo root (since this file is in repo root)
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
INDEX_META_PATH = CHROMA_PATH / "index_meta.json"

# ---------- Helpers ----------
def index_exists() -> bool:
    return CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir())

def write_index_meta(success: bool, stdout_text: str = "", stderr_text: str = ""):
    """
    Store minimal metadata to make the UI feel production-ready.
    We keep it simple and robust: timestamp + success flag.
    """
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "stdout_tail": stdout_text[-2000:] if stdout_text else "",
        "stderr_tail": stderr_text[-2000:] if stderr_text else "",
    }
    INDEX_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def read_index_meta():
    if INDEX_META_PATH.exists():
        try:
            return json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def build_index():
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, "-m", "ingestion.build_index"],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
        env=env
    )

# ---------- Page ----------
st.set_page_config(page_title="CeADAR RAG", layout="wide")
st.title("CeADAR RAG")
st.caption("Chat memory improves retrieval, while answers remain grounded ONLY in retrieved document context.")

# ---------- Auto-build on first run ----------
if not index_exists():
    st.info("Index not found. Building it now (first run only)...")
    with st.spinner("Building index..."):
        result = build_index()
    if result.returncode == 0:
        write_index_meta(True, stdout_text=result.stdout)
        st.success("Index built. You can now ask questions.")
    else:
        write_index_meta(False, stderr_text=result.stderr)
        st.error("Auto index build failed.")
        st.code(result.stderr)
        st.stop()

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example prompts (great for interviewers)
EXAMPLE_PROMPTS = [
    "What is self-attention and why is it useful?",
    "What are the key components of the Transformer architecture?",
    "What are the main themes or obligations discussed in the EU AI Act document?",
]

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Status")
    meta = read_index_meta()
    if index_exists():
        st.success("Index: ready")
    else:
        st.warning("Index: missing")

    if meta and meta.get("built_at_utc"):
        st.caption(f"Last built (UTC): {meta['built_at_utc']}")

    st.divider()

    st.subheader("Indexing")
    st.write("If you changed documents, rebuild the index.")
    if st.button("Build/Rebuild Index"):
        with st.spinner("Building index..."):
            result = build_index()
            if result.returncode == 0:
                write_index_meta(True, stdout_text=result.stdout)
                st.success("Index built successfully.")
                # Keep stdout behind an expander to avoid clutter
                with st.expander("Build logs", expanded=False):
                    st.code(result.stdout)
            else:
                write_index_meta(False, stderr_text=result.stderr)
                st.error("Index build failed.")
                with st.expander("Error logs", expanded=True):
                    st.code(result.stderr)

    st.divider()

    st.subheader("Retrieval Settings")
    top_k = st.slider("Top-K chunks", 2, 8, 4)
    show_context = st.toggle("Show retrieved sources", value=True)
    show_rewrite = st.toggle("Show rewritten retrieval query", value=False)

    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Example prompts (main area) ----------
# These buttons help interviewers start quickly
st.markdown("### Try an example question")
cols = st.columns(len(EXAMPLE_PROMPTS))
for i, p in enumerate(EXAMPLE_PROMPTS):
    if cols[i].button(p):
        st.session_state.messages.append({"role": "user", "content": p})
        st.rerun()

st.divider()

# ---------- Chat History Display ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

        if m["role"] == "assistant" and show_rewrite and m.get("rewritten_query"):
            st.caption(f"Rewritten retrieval query: {m['rewritten_query']}")

        # Sources displayed as expanders (clean UI)
        if m["role"] == "assistant" and show_context and m.get("sources"):
            st.markdown("### Sources Used (Top Matches)")
            for s in m["sources"]:
                source_file = s.get("source_file", "unknown")
                page = s.get("page", None)
                dist = s.get("distance", None)
                label = f"[{s.get('rank', '?')}] {source_file}"
                if page is not None:
                    label += f" (page {page})"
                if dist is not None:
                    label += f" — distance: {float(dist):.4f}"

                with st.expander(label, expanded=False):
                    st.code(s.get("text_preview", ""))

# ---------- Chat Input ----------
user_input = st.chat_input("Ask a question about the provided documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            try:
                res = answer(
                    user_input,
                    top_k=top_k,
                    history=st.session_state.messages
                )

                st.write(res["answer"])

                if show_rewrite:
                    st.caption(f"Rewritten retrieval query: {res.get('rewritten_query', '')}")

                if show_context:
                    st.markdown("### Sources Used (Top Matches)")
                    for s in res.get("sources", []):
                        source_file = s.get("source_file", "unknown")
                        page = s.get("page", None)
                        dist = s.get("distance", None)
                        label = f"[{s.get('rank', '?')}] {source_file}"
                        if page is not None:
                            label += f" (page {page})"
                        if dist is not None:
                            label += f" — distance: {float(dist):.4f}"

                        with st.expander(label, expanded=False):
                            st.code(s.get("text_preview", ""))

                # Save assistant message WITH sources (persists across reruns)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res.get("sources", []),
                    "rewritten_query": res.get("rewritten_query", "")
                })

            except Exception as e:
                err = str(e)
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
