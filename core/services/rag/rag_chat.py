# pipelines/rag_chat.py

import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.services.rag.llm_backend import get_llm, get_rewrite_llm
from core.services.rag.retrieval import get_retriever

HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", "6"))

REWRITE_SYSTEM = """Tu reformules des questions, tu n'y réponds jamais.
Réécris le dernier message de l'utilisateur pour qu'il soit compréhensible seul, sans
l'historique de la conversation, en gardant sa langue et sa forme de question.
Si le message n'est pas une question (salutation, remerciement...), renvoie-le tel quel.
Renvoie UNIQUEMENT la question réécrite, sans explication.

Exemple :
Historique : "Qu'est-ce que le droit à un environnement sain ?" / "C'est le droit à un air pur..."
Message : "Et les écoles, quel est leur rôle ?"
Réponse : Quel est le rôle des écoles dans le droit à un environnement sain ?"""

ANSWER_SYSTEM = """Tu es E-VIE, l'assistant d'une plateforme sur la protection de l'environnement.
Tu réponds toujours en français, sauf si l'utilisateur écrit clairement dans une autre langue.

Règles :
1. Si le message est une salutation ou une simple conversation (bonjour, merci, ok, ça va...),
   réponds brièvement et chaleureusement. Présente-toi en une phrase seulement si c'est le
   premier message de la conversation ; ensuite, ne répète pas ta présentation : un simple
   « ok » ou « merci » appelle une courte phrase et une proposition de continuer.
2. Si le contexte fourni permet de répondre, réponds en t'appuyant UNIQUEMENT sur lui,
   de façon claire et structurée.
3. Si le contexte ne permet pas de répondre, dis-le en une phrase, sans inventer, et propose
   des sujets sur lesquels tu peux aider : protection de l'environnement, droit à un
   environnement sain, bien-être humain et environnement, rôle des écoles et des
   communautés face aux risques environnementaux."""


def _history_messages(history):
    """history is a list of [question, answer] pairs kept in the user's session."""
    messages = []
    for question, answer in history[-HISTORY_TURNS:]:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer[:1500]))
    return messages


_SMALL_TALK = re.compile(
    r"^(ok(ay)?|d'accord|merci( beaucoup)?|super|parfait|bonjour|bonsoir|salut|hello|hi|hey|"
    r"ça va( \?)?|ca va( \?)?|oui|non|au revoir|bye|à bientôt|a bientot|cool|top|bien)[ !.]*$",
    re.IGNORECASE,
)


def is_small_talk(message):
    """Greetings and acknowledgements: no point searching the PDFs for them."""
    return bool(_SMALL_TALK.match(message.strip()))


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
        standalone_question = get_rewrite_llm().invoke(rewrite_messages).content.strip() or question
    else:
        standalone_question = question

    # -------- Retrieve documents --------
    if is_small_talk(question):
        context = "(aucun : message conversationnel, réponds sans documents)"
    else:
        docs = get_retriever().invoke(standalone_question)
        context = "\n\n".join(doc.page_content for doc in docs)

    # -------- Answer --------
    prompt = f"""Contexte :
{context}

Message de l'utilisateur :
{question}"""

    # The answer model also sees the recent exchanges, so it knows what it already
    # said (no repeated introductions) and can keep the thread coherent.
    messages = (
        [SystemMessage(content=ANSWER_SYSTEM)]
        + _history_messages(history)
        + [HumanMessage(content=prompt)]
    )
    return llm.invoke(messages).content.strip()
