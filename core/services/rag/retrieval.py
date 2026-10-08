# pipelines/retrieval.py

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from core.services.rag.llm_backend import CF_AI_EMBED_MODEL, get_embeddings, rerank

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR))
DOCS_PATH = BASE_DIR / "docs"
COLLECTION = "evie_docs"

CANDIDATES = int(os.getenv("RAG_CANDIDATES", "20"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Reranker scores, measured on evaluation/questions.json: the best passage scores at most
# 0.03 for off-topic messages and usually above 0.1 for real questions, so the best
# passage decides whether anything is relevant at all.
MIN_RELEVANCE = float(os.getenv("RAG_MIN_RELEVANCE", "0.05"))
# Reciprocal rank fusion constant: lower values give the top ranks more weight.
RRF_K = 10

# Bump when the way documents are read or split changes, to force a rebuild.
INDEX_VERSION = "5"


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
    """Passages that answer the query, best first (possibly none).

    Embedding similarity alone cannot tell relevant from irrelevant: a one-word message
    like "ok" scores as high as a real question. So a cross-encoder decides whether
    anything is relevant. It is a poor judge of the order, though, on French legal text
    or questions framed as "what does the code say about ...", where the vector search
    ranks the right article first; the final order fuses both rankings (RRF), which
    finds the source page for 43 of the 50 evaluation questions against 35 for the
    cross-encoder order alone."""
    candidates = get_vectorstore().similarity_search(query, k=CANDIDATES)
    if not candidates:
        return []
    try:
        scores = rerank(query, [doc.page_content for doc in candidates])
    except Exception:
        # Better a plain similarity ranking than an error while Workers AI recovers.
        logger.warning("Reranker unavailable, falling back to similarity order", exc_info=True)
        return candidates[:TOP_K]
    by_score = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    if scores[by_score[0]] < MIN_RELEVANCE:
        return []
    score_rank = {i: rank for rank, i in enumerate(by_score)}
    # Candidates come in vector-search order, so a candidate's index is its vector rank.
    fused = sorted(
        range(len(candidates)),
        key=lambda i: 1 / (RRF_K + score_rank[i]) + 1 / (RRF_K + i),
        reverse=True,
    )
    return [candidates[i] for i in fused[:TOP_K]]
