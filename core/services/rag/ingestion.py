# pipelines/ingestion.py

from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.services.rag.retrieval import BASE_DIR, DB_PATH, get_vectorstore

DOCS_PATH = BASE_DIR / "docs"


def load_documents():
    documents = []
    for pdf in sorted(DOCS_PATH.glob("*.pdf")):
        with fitz.open(pdf) as doc:
            for page in doc:
                text = page.get_text().strip()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf.name, "page": page.number + 1},
                    ))
    if not documents:
        raise ValueError(f"No documents found in {DOCS_PATH}")
    return documents


def split_documents(documents):
    # Recursive splitter enforces the size limit; CharacterTextSplitter only splits on
    # blank lines and happily returns page-sized chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)


def vectorstore_is_populated():
    return DB_PATH.exists() and get_vectorstore()._collection.count() > 0


def run_ingestion(force=False):
    """Embed the PDFs in docs/ into Chroma. Skipped when the index already has data."""
    store = get_vectorstore()
    if not force and store._collection.count() > 0:
        print(f"Vector DB already has {store._collection.count()} chunks")
        return
    if force and store._collection.count() > 0:
        store.reset_collection()

    chunks = split_documents(load_documents())
    print(f"Embedding {len(chunks)} chunks...")
    # Chroma's add_documents calls the embedding client in batches of chunk_size.
    store.add_documents(chunks)
    print(f"Ingestion complete: {store._collection.count()} chunks stored")
