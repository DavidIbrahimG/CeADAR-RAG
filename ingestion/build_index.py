import os
import shutil

from config.settings import RAW_DIR, CHROMA_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.loaders import load_documents
from ingestion.chunking import chunk_documents
from vector_store.embeddings import get_embedding_function
from vector_store.chroma_store import get_or_create_collection, add_documents_to_collection

def main():
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"Missing raw data folder: {RAW_DIR}")

    # clean rebuild
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    docs = load_documents(RAW_DIR)
    if not docs:
        raise RuntimeError("No .pdf or .docx found in data/raw")

    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)

    embedding_fn = get_embedding_function()
    collection = get_or_create_collection(
        persist_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    add_documents_to_collection(collection, chunks)

    print(f"✅ Built index | docs={len(docs)} chunks={len(chunks)} db={CHROMA_DIR}")

if __name__ == "__main__":
    main()
