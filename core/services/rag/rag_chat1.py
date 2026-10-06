# pipelines/rag_chat.py

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama.llms import OllamaLLM

from pipelines.retrieval import get_retriever

retriever = get_retriever()

llm = OllamaLLM(model="ministral-3:latest")

chat_history = []


def ask_question(question):

    global chat_history

    # ---- Rewrite question using history ----

    if chat_history:

        messages = [
            SystemMessage(
                content="Rewrite the user question to be standalone using the chat history."
            )
        ] + chat_history + [HumanMessage(content=question)]

        standalone_question = llm.invoke(messages).content

    else:
        standalone_question = question

    # ---- Retrieve docs ----

    docs = retriever.invoke(standalone_question)

    context = "\n".join([doc.page_content for doc in docs])
    print(context)
    # ---- Final prompt ----

    prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{question}
"""

    messages = [
        SystemMessage(content="You answer questions using retrieved documents."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages).content

    # ---- Update memory ----

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response