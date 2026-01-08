from typing import List, Dict
from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, GROQ_MODEL

PRONOUNS = {"it", "this", "that", "they", "those", "these", "he", "she", "them", "its", "their"}

def _last_user_topic(history: List[Dict[str, str]]) -> str:
    """
    Pull a simple 'topic' from the last user message that isn't just a pronoun.
    This is a lightweight, reliable fallback for coreference like 'it'.
    """
    # Walk backwards through user messages
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        text = (m.get("content") or "").strip()
        if not text:
            continue
        # If message is longer than a few chars, treat it as topic-ish
        return text
    return ""

def rewrite_query(user_query: str, history: List[Dict[str, str]]) -> str:
    """
    Converts a follow-up question into a standalone search query using chat history.
    Retrieval uses this rewritten query; generation uses original user question.
    """

    uq = (user_query or "").strip()
    if not history or not uq:
        return uq

    # --- Rule-based coreference fallback for short pronoun questions ---
    # Example: "why is it important" → "why is self-attention important"
    tokens = [t.strip("?.!,").lower() for t in uq.split()]
    if any(t in PRONOUNS for t in tokens) and len(tokens) <= 6:
        topic = _last_user_topic(history[:-1])  # exclude current question
        if topic:
            # Build a better standalone query without overthinking
            return f"{uq} (referring to: {topic})"

    # --- LLM rewriter (more precise) ---
    if not GROQ_API_KEY:
        return uq

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.0,
    )

    # Keep only the last few turns to prevent noise
    recent = history[-8:]
    transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent if m.get("content")])

    system = (
        "Rewrite the user's latest message into a SINGLE standalone retrieval query.\n"
        "Resolve references like 'it/that/this' using the conversation.\n"
        "Do NOT ask for more context.\n"
        "Do NOT answer.\n"
        "Output ONLY the rewritten query text."
    )

    user = (
        f"Conversation:\n{transcript}\n\n"
        f"Latest user message:\n{uq}\n\n"
        "Standalone retrieval query:"
    )

    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    rewritten = (resp.content or "").strip()

    # Safety fallback
    if not rewritten or len(rewritten) > 300:
        return uq
    return rewritten
