# core/services/ai_services.py

from core.services.rag.ingestion import run_ingestion
from core.services.rag.rag_chat import ask_question

rag_initialized = False


def rag_query(question):

    global rag_initialized

    if not rag_initialized:
        run_ingestion()
        rag_initialized = True

    answer = ask_question(question)

    return answer