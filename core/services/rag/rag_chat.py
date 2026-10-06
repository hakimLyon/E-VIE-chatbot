# pipelines/rag_chat.py

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama.llms import OllamaLLM

# from pipelines.retrieval import get_retriever
from core.services.rag.retrieval import get_retriever
retriever = get_retriever()

llm = OllamaLLM(model="ministral-3:latest")

chat_history = []


def ask_question(question):

    global chat_history

    print("\nUser Question:", question)

    # -------- Rewrite question if history exists --------

    if len(chat_history) > 0:

        rewrite_messages = [
            SystemMessage(
                content="""Rewrite the user's question so that it becomes a standalone
question that can be understood without the chat history.
Return ONLY the rewritten question."""
            )
        ] + chat_history + [HumanMessage(content=question)]

        standalone_question = llm.invoke(rewrite_messages).strip()

    else:

        standalone_question = question

    print("Standalone Question:", standalone_question)

    # -------- Retrieve documents --------

    docs = retriever.invoke(standalone_question)

    context = "\n\n".join([doc.page_content for doc in docs])

    print("\nRetrieved Context:\n", context)

    # -------- Build final prompt --------

    prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}
"""

    messages = [
        SystemMessage(content="You answer questions using retrieved documents."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)   # already a string

    # -------- Save conversation --------

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response