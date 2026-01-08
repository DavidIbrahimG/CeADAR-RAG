from typing import List, Dict, Any
from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, GROQ_MODEL

def build_context(docs: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        md = d.get("metadata", {}) or {}
        src = md.get("source_file", "unknown")
        page = md.get("page", None)
        header = f"[{i}] source={src}" + (f", page={page}" if page is not None else "")
        blocks.append(f"{header}\n{d['text']}")
    return "\n\n---\n\n".join(blocks)

def generate(query: str, docs: List[Dict[str, Any]]) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env")

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.2,
    )

    context = build_context(docs)

    system = (
    "You are a factual assistant.\n"
    "Use ONLY the provided context.\n"
    "Respond with clear, factual statements grounded in the context.\n"
    "Cite sources using [1], [2], etc.\n"
    "If some parts of the question are not supported by the context, "
    "simply omit them or end the answer naturally.\n"
    "Do NOT explain what you do not know.\n"
    "Do NOT list missing information.\n"
    "Do NOT mention limitations or lack of knowledge explicitly.\n"
    )



    user = f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:"

    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return resp.content
