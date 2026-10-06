# pipelines/ingestion.py

import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

#DOCS_PATH = "docs"
#DB_PATH = "db/chroma_db"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

DOCS_PATH = BASE_DIR / "docs"
DB_PATH = BASE_DIR / "db" / "chroma_db"
def load_documents():
    loader = DirectoryLoader(
        path=DOCS_PATH,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No documents found in docs/")

    return documents


def split_documents(documents):

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


def create_vectorstore(chunks):

    embeddings = OllamaEmbeddings(model="bge-m3")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    return vectorstore


def run_ingestion():

    if os.path.exists(DB_PATH):
        print("Vector DB already exists")
        return

    docs = load_documents()

    chunks = split_documents(docs)

    create_vectorstore(chunks)

    print("Ingestion complete")