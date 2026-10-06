# pipelines/retrieval.py

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DB_PATH = "db/chroma_db"


def get_retriever():

    embeddings = OllamaEmbeddings(model="bge-m3")

    # db = Chroma(
    #     persist_directory=DB_PATH,
    #     embedding_function=embeddings
    # )
    
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}  
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    return retriever