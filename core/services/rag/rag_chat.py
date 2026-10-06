# pipelines/rag_chat.py

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.services.rag.llm_backend import get_llm
from core.services.rag.retrieval import get_retriever

HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", "6"))

REWRITE_SYSTEM = """Rewrite the user's question so that it becomes a standalone
question that can be understood without the chat history. Keep the user's language.
Return ONLY the rewritten question."""

ANSWER_SYSTEM = """You are E-VIE, an assistant on environmental protection.
Answer using ONLY the provided context. Reply in the same language as the question.
If the context does not contain the answer, say so briefly instead of guessing."""


def _history_messages(history):
    """history is a list of [question, answer] pairs kept in the user's session."""
    messages = []
    for question, answer in history[-HISTORY_TURNS:]:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


def ask_question(question, history=None):
    history = history or []
    llm = get_llm()

    # -------- Rewrite question if history exists --------
    if history:
        rewrite_messages = (
            [SystemMessage(content=REWRITE_SYSTEM)]
            + _history_messages(history)
            + [HumanMessage(content=question)]
        )
        standalone_question = llm.invoke(rewrite_messages).content.strip() or question
    else:
        standalone_question = question

    # -------- Retrieve documents --------
    docs = get_retriever().invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # -------- Answer --------
    prompt = f"""Context:
{context}

Question:
{question}"""

    messages = [SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=prompt)]
    return llm.invoke(messages).content.strip()
