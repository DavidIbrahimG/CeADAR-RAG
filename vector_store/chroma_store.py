from typing import List, Optional, Dict, Any
import chromadb

def get_or_create_collection(persist_dir: str, collection_name: str, embedding_function):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    return collection

def add_documents_to_collection(collection, docs: List[Any]):
    """
    docs are LangChain Documents (or compatible objects) with:
      - page_content
      - metadata
    """
    texts = [d.page_content for d in docs]
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for i, d in enumerate(docs):
        md = dict(d.metadata or {})
        metadatas.append(md)
        # unique deterministic IDs
        ids.append(f"{md.get('source_file','doc')}-{md.get('page', 'na')}-{i}")

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
