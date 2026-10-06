# pipelines/retrieval.py

import os
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from core.services.rag.llm_backend import get_embeddings

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR))
DB_PATH = DATA_DIR / "chroma_db"
COLLECTION = "evie_docs"
TOP_K = int(os.getenv("RAG_TOP_K", "5"))


@lru_cache(maxsize=1)
def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=str(DB_PATH),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
