import streamlit as st
import subprocess
from rag.pipeline import answer

st.set_page_config(page_title="CeADAR RAG", layout="wide")
st.title("CeADAR RAG Prototype")
st.caption("Chat memory improves retrieval, while answers remain grounded ONLY in retrieved document context.")
st.sidebar.caption("BUILD: sources-persist-v2")

# ---------- Session State ----------
if "messages" not in st.session_state:
    # Each message can optionally include:
    # - sources: list of retrieved chunks metadata
    # - rewritten_query: retrieval query derived from conversation
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "...", "sources": [...]}]

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Indexing")
    st.write("If you changed documents, rebuild the index.")
    if st.button("Build/Rebuild Index"):
        with st.spinner("Building index..."):
            result = subprocess.run(["python", "-m", "ingestion.build_index"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Index built successfully.")
                st.code(result.stdout)
            else:
                st.error("Index build failed.")
                st.code(result.stderr)

    st.subheader("Retrieval Settings")
    top_k = st.slider("Top-K chunks", 2, 8, 4)

    show_context = st.toggle("Show retrieved sources", value=True)
    show_rewrite = st.toggle("Show rewritten retrieval query", value=False)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat History Display ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

        # Re-render sources for assistant messages on rerun 
        if (
            m["role"] == "assistant"
            and show_context
            and m.get("sources")
        ):
            st.markdown("### Sources Used (Top Matches)")
            for s in m["sources"]:
                meta = f"**[{s.get('rank', '?')}]** `{s.get('source_file', 'unknown')}`"
                if s.get("page") is not None:
                    meta += f" (page {s['page']})"
                if "distance" in s:
                    meta += f" — similarity distance: `{float(s['distance']):.4f}`"
                st.markdown(meta)
                st.code(s.get("text_preview", ""))

        if (
            m["role"] == "assistant"
            and show_rewrite
            and m.get("rewritten_query")
        ):
            st.caption(f"Rewritten retrieval query: {m['rewritten_query']}")

# ---------- Chat Input ----------
user_input = st.chat_input("Ask a question about the provided documents...")

if user_input:
    # Save + show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            try:
                res = answer(
                    user_input,
                    top_k=top_k,
                    history=st.session_state.messages
                )

                st.write(res["answer"])

                # Optional: show rewritten query for THIS turn
                if show_rewrite:
                    st.caption(f"Rewritten retrieval query: {res.get('rewritten_query', '')}")

                # Optional: show sources for THIS turn
                if show_context:
                    st.markdown("### Sources Used (Top Matches)")
                    for s in res.get("sources", []):
                        meta = f"**[{s.get('rank', '?')}]** `{s.get('source_file', 'unknown')}`"
                        if s.get("page") is not None:
                            meta += f" (page {s['page']})"
                        if "distance" in s:
                            meta += f" — similarity distance: `{float(s['distance']):.4f}`"
                        st.markdown(meta)
                        st.code(s.get("text_preview", ""))

                # Saving assistant message WITH sources so they persist across reruns
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
