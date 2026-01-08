import streamlit as st
import subprocess
from rag.pipeline import answer

st.set_page_config(page_title="CeADAR RAG", layout="wide")
st.title("CeADAR RAG Prototype (Groq + Open Embeddings)")
st.caption("Chat memory improves retrieval, while answers remain grounded ONLY in retrieved document context.")

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]

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

# ---------- Chat Input ----------
user_input = st.chat_input("Ask a question about the provided documents...")

if user_input:
    # Show user message
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

                if show_rewrite:
                    st.caption(f"Rewritten retrieval query: {res['rewritten_query']}")

                if show_context:
                    st.markdown("### Sources Used (Top Matches)")
                    for s in res["sources"]:
                        meta = f"**[{s['rank']}]** `{s['source_file']}`"
                        if s["page"] is not None:
                            meta += f" (page {s['page']})"
                        meta += f" — similarity distance: `{s['distance']:.4f}`"
                        st.markdown(meta)
                        st.code(s["text_preview"])

                
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})

            except Exception as e:
                err = str(e)
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
