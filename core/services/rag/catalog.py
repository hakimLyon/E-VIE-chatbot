"""Titles and descriptions of the documents in docs/, for citations and the documents panel."""

import json
from functools import lru_cache

import fitz  # PyMuPDF

from core.services.rag.retrieval import DOCS_PATH


@lru_cache(maxsize=1)
def documents():
    """One entry per PDF in docs/, in the order of docs/catalog.json; a PDF missing from
    the catalog still appears, titled by its file name."""
    catalog_path = DOCS_PATH / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []
    known = {entry["file"]: entry for entry in catalog}
    files = [entry["file"] for entry in catalog if (DOCS_PATH / entry["file"]).is_file()]
    files += sorted(path.name for path in DOCS_PATH.glob("*.pdf") if path.name not in known)

    entries = []
    for name in files:
        info = known.get(name, {})
        with fitz.open(DOCS_PATH / name) as doc:
            pages = doc.page_count
        entries.append({
            "file": name,
            "title": info.get("title") or name.rsplit(".", 1)[0],
            "publisher": info.get("publisher", ""),
            "year": info.get("year"),
            "language": info.get("language", ""),
            "summary": info.get("summary", ""),
            "pages": pages,
            "url": f"/documents/{name}",
        })
    return entries


def source(file_name, page):
    """Citation of one page: title, page number and a link that opens the PDF there."""
    title = next((entry["title"] for entry in documents() if entry["file"] == file_name), file_name)
    return {"title": title, "page": page, "url": f"/documents/{file_name}#page={page}"}
