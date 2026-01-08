from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

RAW_DIR = str(BASE_DIR / "data" / "raw")
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ceadar_docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
