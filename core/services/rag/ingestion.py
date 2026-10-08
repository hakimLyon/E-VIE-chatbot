# pipelines/ingestion.py

import fcntl
import re
import shutil

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.services.rag.retrieval import DATA_DIR, DB_PATH, DOCS_PATH, get_vectorstore

READY_MARKER = DB_PATH / ".complete"


def _read_pages(pdf):
    """Text of each page. A `<name>.ocr.txt` next to the PDF (pages separated by form
    feeds, as written by `ocrmypdf --sidecar`) takes precedence: some PDFs embed fonts
    without a Unicode map and their text layer extracts as gibberish."""
    sidecar = pdf.with_suffix(".ocr.txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").split("\f")
    with fitz.open(pdf) as doc:
        return [page.get_text() for page in doc]


def load_documents():
    documents = []
    for pdf in sorted(DOCS_PATH.glob("*.pdf")):
        for number, text in enumerate(_read_pages(pdf), start=1):
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text).strip()  # words cut at line ends
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf.name, "page": number},
                ))
    if not documents:
        raise ValueError(f"No documents found in {DOCS_PATH}")
    return documents


def is_usable(text):
    """False for chunks that only add noise to retrieval: page numbers, running heads,
    figure tables, or text from fonts without a Unicode map."""
    compact = " ".join(text.split())
    if len(compact) < 80:
        return False
    if sum(c.isalpha() for c in compact) / len(compact) < 0.6:
        return False
    garbled = sum(1 for c in compact if "\x80" <= c <= "\x9f")
    return garbled / len(compact) < 0.01


def split_documents(documents):
    # Recursive splitter enforces the size limit; CharacterTextSplitter only splits on
    # blank lines and happily returns page-sized chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    usable = [chunk for chunk in chunks if is_usable(chunk.page_content)]
    print(f"{len(chunks)} chunks, {len(chunks) - len(usable)} dropped as noise")
    return usable


def _remove_old_indexes():
    """Delete indexes built from older documents, except the newest one: during a
    rolling deploy the previous container still reads it until this one is healthy."""
    old = sorted(
        (path for path in DATA_DIR.glob("chroma*") if path.is_dir() and path != DB_PATH),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in old[1:]:
        shutil.rmtree(path, ignore_errors=True)


def run_ingestion(force=False):
    """Embed the documents in docs/ into Chroma, unless this exact index is complete.

    The lock keeps several processes (gunicorn workers, the startup command) from
    building the same index at once; the marker tells a finished index from one
    interrupted halfway."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / ".ingest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = get_vectorstore()
        if READY_MARKER.exists() and not force:
            print(f"Vector DB {DB_PATH.name} ready ({store._collection.count()} chunks)")
            return
        if store._collection.count():
            store.reset_collection()

        chunks = split_documents(load_documents())
        print(f"Embedding {len(chunks)} chunks into {DB_PATH.name}...")
        # Chroma's add_documents calls the embedding client in batches of chunk_size.
        store.add_documents(chunks)
        READY_MARKER.write_text(f"{store._collection.count()} chunks\n")
        print(f"Ingestion complete: {store._collection.count()} chunks stored")
        _remove_old_indexes()
