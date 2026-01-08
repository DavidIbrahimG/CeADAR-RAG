from typing import List, Dict
from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, GROQ_MODEL

def rewrite_query(user_query: str, history: List[Dict[str, str]]) -> str:
    """
    Converts a follow-up question into a standalone query using chat history.
    This improves retrieval while keeping generation grounded in retrieved context.
    """
    if not GROQ_API_KEY or not history:
        return user_query

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.0,
    )

    
    recent = history[-6:]  # last 6 turns max
    transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])

    system = (
        "Rewrite the user's latest question into a SINGLE standalone search query.\n"
        "Use the conversation context only to resolve references (e.g., 'it', 'that section').\n"
        "Do NOT answer the question. Output ONLY the rewritten query."
    )

    user = f"Conversation:\n{transcript}\n\nLatest question:\n{user_query}\n\nStandalone query:"

    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    rewritten = (resp.content or "").strip()

    
    if not rewritten or len(rewritten) > 300:
        return user_query
    return rewritten
