from typing import Dict, Any, List, Optional

from config.settings import TOP_K, CHROMA_DIR, COLLECTION_NAME
from vector_store.embeddings import get_embedding_function
from vector_store.chroma_store import get_or_create_collection
from rag.retriever import retrieve, to_sources
from rag.generator import generate
from rag.query_rewriter import rewrite_query

def load_collection():
    embedding_fn = get_embedding_function()
    return get_or_create_collection(
        persist_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

def answer(query: str, top_k: int = TOP_K, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    collection = load_collection()

    # memory to rewrite query ONLY for retrieval
    rewritten = rewrite_query(query, history or [])
    docs = retrieve(collection, rewritten, top_k=top_k)

    # Generation remains grounded ONLY in retrieved docs
    ans = generate(query, docs)

    return {
        "answer": ans,
        "sources": to_sources(docs),
        "rewritten_query": rewritten
    }
