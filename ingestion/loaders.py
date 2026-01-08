import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def load_documents(raw_dir: str) -> List[Document]:
    docs: List[Document] = []

    for filename in os.listdir(raw_dir):
        path = os.path.join(raw_dir, filename)
        _, ext = os.path.splitext(filename.lower())

        if ext == ".pdf":
            loaded = PyPDFLoader(path).load()
        elif ext == ".docx":
            loaded = Docx2txtLoader(path).load()
        else:
            continue

        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["source_file"] = filename
        docs.extend(loaded)

    return docs
