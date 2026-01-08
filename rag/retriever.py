from typing import List, Dict, Any

def retrieve(collection, query: str, top_k: int) -> List[Dict[str, Any]]:
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs: List[Dict[str, Any]] = []
    for i in range(len(res["documents"][0])):
        docs.append({
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i] or {},
            "distance": res["distances"][0][i],
            "rank": i + 1
        })
    return docs

def to_sources(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        md = d["metadata"]
        preview = d["text"]
        out.append({
            "rank": d["rank"],
            "source_file": md.get("source_file", "unknown"),
            "page": md.get("page", None),
            "distance": float(d["distance"]),
            "text_preview": (preview[:350] + "...") if len(preview) > 350 else preview
        })
    return out
