# pipelines/retrieval.py

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from core.services.rag.llm_backend import CF_AI_EMBED_MODEL, get_embeddings, rerank

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR))
DOCS_PATH = BASE_DIR / "docs"
COLLECTION = "evie_docs"

CANDIDATES = int(os.getenv("RAG_CANDIDATES", "20"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Reranker scores, measured on evaluation/questions.json: the best passage scores at most
# 0.03 for off-topic messages and usually above 0.1 for real questions, while useful
# passages of a relevant answer (lists, calendars) can score lower on their own. So the
# best passage decides whether anything is relevant, and a low floor trims the rest.
MIN_RELEVANCE = float(os.getenv("RAG_MIN_RELEVANCE", "0.05"))
MIN_PASSAGE_RELEVANCE = float(os.getenv("RAG_MIN_PASSAGE_RELEVANCE", "0.01"))

# Bump when the way documents are read or split changes, to force a rebuild.
INDEX_VERSION = "2"


def _index_id():
    """Fingerprint of everything the index is built from: documents, embedding model
    and ingestion version. A change gives a new index directory, so a deploy with new
    documents rebuilds the index while the previous container keeps using the old one."""
    digest = hashlib.sha256(f"{INDEX_VERSION}|{CF_AI_EMBED_MODEL}".encode())
    for path in sorted(DOCS_PATH.glob("*")):
        if path.suffix in (".pdf", ".txt"):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


DB_PATH = DATA_DIR / f"chroma_{_index_id()}"


@lru_cache(maxsize=1)
def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=str(DB_PATH),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def search(query):
    """Passages that actually answer the query, best first (possibly none).

    Embedding similarity alone cannot tell relevant from irrelevant: a one-word
    message like "ok" scores as high as a real question. So the vector search only
    proposes candidates and a cross-encoder decides which ones are relevant."""
    candidates = get_vectorstore().similarity_search(query, k=CANDIDATES)
    if not candidates:
        return []
    scores = rerank(query, [doc.page_content for doc in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    if ranked[0][0] < MIN_RELEVANCE:
        return []
    return [doc for score, doc in ranked[:TOP_K] if score >= MIN_PASSAGE_RELEVANCE]
