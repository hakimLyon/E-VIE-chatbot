# pipelines/ingestion.py

import fcntl
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.services.rag.llm_backend import get_ingestion_embeddings
from core.services.rag.retrieval import DATA_DIR, DB_PATH, DOCS_PATH, get_vectorstore

READY_MARKER = DB_PATH / ".complete"


def _read_pages(pdf):
    """Text of each page. A `<name>.ocr.txt` next to the PDF (pages separated by form
    feeds, as written by `ocrmypdf --sidecar`) takes precedence: some PDFs embed fonts
    without a Unicode map and their text layer extracts as gibberish, and a PDF whose
    images were compressed for display can keep the exact text of its original there."""
    sidecar = pdf.with_suffix(".ocr.txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").split("\f")
    with fitz.open(pdf) as doc:
        return [page.get_text() for page in doc]


# Pages that hold no knowledge but attract searches for names and organisations:
# credits, acknowledgements, tables of contents, copyright pages, reference lists.
_MATTER_HEADING = re.compile(
    r"^(acknowledg(e)?ments?|bibliography|references|reference list|(table of )?contents|"
    r"photo(graph)?s? credits?|photos|table des matières|sommaire|remerciements|"
    r"bibliographie|références( bibliographiques)?|notes|crédits photos?)(\s+[ivxlc\d]+)?\W*$",
    re.IGNORECASE,
)
# Lists of people or abbreviations, whose heading is often followed by the list itself
# on the same line ("Équipe de rédaction: Richard P. Allan (Royaume-Uni), ..."). A list
# of names and countries otherwise matches any "who ...?" question.
_LIST_HEADING = re.compile(
    r"^(équipe de rédaction|auteurs principaux|drafting authors|lead authors|contributing authors|"
    r"task force members|liste des abréviations|list of abbreviations|abbreviations|"
    r"sigles et abréviations|acronymes)\b",
    re.IGNORECASE,
)
_REFERENCE_LINE = re.compile(r"^[A-Z][A-Za-z'\-]+,\s*[A-Z]\.|\(\d{4}[a-z]?\)|\b(19|20)\d{2}[a-z]?\.\s")


def _reference_density(lines):
    return sum(bool(_REFERENCE_LINE.search(line)) for line in lines) / len(lines) if lines else 0.0


def _page_kind(text, page_number):
    """'references', 'matter' (credits, contents, acknowledgements) or None."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(_LIST_HEADING.match(line) for line in lines[:5]):
        return "matter"
    for line in lines[:5]:  # below the running head
        heading = _MATTER_HEADING.match(line)
        if heading:
            is_references = re.match(r"(ref|réf|biblio|notes)", heading.group(1), re.IGNORECASE)
            return "references" if is_references else "matter"
    # Only near the start: content pages can carry a copyright line in their footer.
    if page_number <= 4 and re.search(r"\bISBN\b|all rights reserved|reuse policy|©", text, re.IGNORECASE):
        return "matter"
    if len(lines) > 8 and _reference_density(lines) > 0.35:
        return "references"
    return None


def skipped_pages(texts):
    """Page numbers to leave out of the index. A reference list often repeats its heading
    on every other page only, and wrapped French references give a lower density, so a
    page with many reference-like lines next to a reference page belongs to the list."""
    kinds = {number: _page_kind(text, number) for number, text in enumerate(texts, start=1)}
    densities = {
        number: _reference_density([line.strip() for line in text.splitlines() if line.strip()])
        for number, text in enumerate(texts, start=1)
    }
    grew = True
    while grew:  # a page added to a list can bring in its own neighbour
        grew = False
        for number in kinds:
            near_references = "references" in (kinds.get(number - 1), kinds.get(number + 1))
            if kinds[number] is None and near_references and densities[number] >= 0.15:
                kinds[number] = "references"
                grew = True
    return {number for number, kind in kinds.items() if kind}


def load_documents():
    documents, skipped = [], 0
    for pdf in sorted(DOCS_PATH.glob("*.pdf")):
        # words cut at line ends are joined back
        texts = [re.sub(r"(\w)-\n(\w)", r"\1\2", text).strip() for text in _read_pages(pdf)]
        to_skip = skipped_pages(texts)
        skipped += len(to_skip)
        for number, text in enumerate(texts, start=1):
            if not text or number in to_skip:
                continue
            documents.append(Document(
                page_content=text,
                metadata={"source": pdf.name, "page": number},
            ))
    if not documents:
        raise ValueError(f"No documents found in {DOCS_PATH}")
    print(f"{len(documents)} pages, {skipped} skipped as credits, contents or references")
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


def _embed(texts, batch_size=16, workers=4):
    """Embeddings of all texts, with several Workers AI requests in flight: a few thousand
    chunks then take well under a minute at startup instead of several."""
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(get_ingestion_embeddings().embed_documents, batches)
    return [vector for batch in results for vector in batch]


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
        texts = [chunk.page_content for chunk in chunks]
        vectors = _embed(texts)
        for start in range(0, len(chunks), 1000):  # Chroma caps the size of one insert
            part = slice(start, start + 1000)
            store._collection.add(
                ids=[str(uuid4()) for _ in texts[part]],
                embeddings=vectors[part],
                documents=texts[part],
                metadatas=[chunk.metadata for chunk in chunks[part]],
            )
        READY_MARKER.write_text(f"{store._collection.count()} chunks\n")
        print(f"Ingestion complete: {store._collection.count()} chunks stored")
        _remove_old_indexes()
