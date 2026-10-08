# pipelines/rag_chat.py

import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.services.rag.llm_backend import get_llm, get_rewrite_llm
from core.services.rag.retrieval import search

HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", "6"))

LANGUAGE_NAMES = {"fr": "français", "en": "English"}

TOPICS = {
    "fr": "la protection de l'environnement, le droit à un environnement sain, le lien entre "
          "environnement et bien-être, ou le rôle des écoles et des communautés face aux "
          "risques environnementaux",
    "en": "environmental protection, the right to a healthy environment, the link between "
          "the environment and human well-being, or the role of schools and communities "
          "in facing environmental risks",
}

# Fixed replies: the research on RAG (e.g. the RGB benchmark) shows models are poor at
# declining on their own when nothing relevant was found, so these cases never reach it.
UNCLEAR_REPLY = {
    "fr": "Je n'ai pas bien compris votre message. Que souhaitez-vous savoir sur la "
          "protection de l'environnement ?",
    "en": "Sorry, I didn't quite understand your message. What would you like to know "
          "about environmental protection?",
}
NO_DOCUMENTS_REPLY = {
    "fr": "Je n'ai pas trouvé d'information sur ce sujet dans nos documents. Je peux vous "
          f"aider sur {TOPICS['fr']}.",
    "en": "I couldn't find information on this topic in our documents. I can help with "
          f"{TOPICS['en']}.",
}

# The documents are in English and the reranker only judges English queries reliably
# (a French question scored 0.003 on the very passage its English version scored 0.999),
# so every question is searched in English. The same call resolves follow-ups.
SEARCH_QUERY_SYSTEM = """You turn the user's last message into a search query for English documents.
Write it as one standalone question in English. Use the conversation only to resolve a
genuine follow-up (pronouns, "and ...?", "why?", "yes"). If the message is a new topic,
translate it on its own and do not mix in the earlier topic. Never answer it.
Return only the English question.

Examples:
Conversation: "Qu'est-ce que le droit à un environnement sain ?" / "C'est le droit à un air pur..."
Message: "Et les écoles, quel est leur rôle ?"
Query: What is the role of schools in the right to a healthy environment?
Message: "cuisine"
Query: What is cooking?"""

ANSWER_SYSTEM = """Tu es E-VIE, l'assistant d'une plateforme sur la protection de l'environnement.

Règles :
- Réponds dans la langue indiquée par la ligne « Langue de réponse ». En français, vouvoie.
- Appuie-toi uniquement sur les extraits fournis, sans rien inventer. S'ils ne couvrent
  qu'une partie de la question, réponds à cette partie et dis ce qui manque.
- Ne parle jamais d'« extraits », de « contexte » ni de « documents fournis » : présente
  l'information directement (au besoin, dis « d'après nos documents »).
- Sois clair et structuré (listes, gras) sans être plus long que nécessaire."""

SMALL_TALK_SYSTEM = f"""Tu es E-VIE, l'assistant d'une plateforme sur la protection de
l'environnement. Tu peux aider sur : {TOPICS['fr']}.

Le dernier message est une formule de politesse, une salutation ou une question sur toi.
Réponds en une ou deux phrases, dans la langue indiquée par la ligne « Langue de réponse »
(en français, vouvoie). Présente-toi seulement si la conversation commence ; sinon ne
répète pas ta présentation. Termine en proposant ton aide."""

_SMALL_TALK = re.compile(
    r"^(ok(ay)?|d'accord|merci( beaucoup)?|thanks?( a lot)?|thank you( very much)?|super|"
    r"parfait|génial|bonjour|bonsoir|salut|coucou|hello|hi|hey|ça va|ca va|how are you|"
    r"au revoir|bye|goodbye|à bientôt|a bientot|cool|top|bien|nice|great|good|lol|mdr|haha|"
    r"qui es[- ]tu|who are you|que peux[- ]tu faire|what can you do|aide|help)[\s!.?]*$",
    re.IGNORECASE,
)
# Answers to a question the assistant just asked: small talk only when nothing precedes.
_YES_NO = re.compile(r"^(oui|non|yes|no)[\s!.?]*$", re.IGNORECASE)
# The rewrite model sometimes answers the user instead of rewriting the message.
_NOT_A_QUERY = re.compile(
    r"^(je ne|désolé|desole|pardon|merci|i'm sorry|i am sorry|i (do not|don't|cannot|can't)|"
    r"sorry|thank|as an ai)",
    re.IGNORECASE,
)

_FR_WORDS = set(
    "le la les des du de un une est et en que qui quoi quel quelle quels quelles comment "
    "pourquoi pour dans sur avec je tu il elle nous vous ils elles ce cette ces mon ma mes "
    "ton ta tes son sa ses leur leurs notre nos votre vos pas ne au aux sont peut peux "
    "faire fait aussi plus bonjour bonsoir salut merci oui non".split()
)
_EN_WORDS = set(
    "the an is are was were what how why which who of to in for with and or do does did "
    "can could should would will i you he she we they my your his her our their this that "
    "these those it its be have has not hello hi hey thanks thank yes please about from".split()
)


def detect_language(text):
    """'fr', 'en', or None when the text gives no clue (e.g. "ok", "pollution")."""
    lowered = text.lower()
    words = re.findall(r"[a-zàâäçéèêëîïôöùûüÿœ]+", lowered)
    fr = sum(word in _FR_WORDS for word in words) + 2 * bool(re.search(r"[àâçéèêëîïôùûœ]", lowered))
    en = sum(word in _EN_WORDS for word in words)
    if fr != en:
        return "fr" if fr > en else "en"
    return None


def conversation_language(question, history):
    """Language of the latest message that has one, French by default."""
    for text in [question] + [past_question for past_question, _ in reversed(history)]:
        language = detect_language(text)
        if language:
            return language
    return "fr"


def route(question, history):
    """'small_talk', 'unclear' or 'question'. Deciding from the message itself is as good
    as asking a model (see the adaptive-retrieval studies) and costs nothing."""
    text = question.strip()
    if _SMALL_TALK.match(text) or (_YES_NO.match(text) and not history):
        return "small_talk"
    if not re.search(r"[^\W\d_]{3,}", text):  # no word of 3 letters or more: "ko", "?", "a b"
        return "unclear"
    return "question"


def _history_messages(history):
    """history is a list of [question, answer] pairs kept in the user's session."""
    messages = []
    for question, answer in history[-HISTORY_TURNS:]:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer[:1500]))
    return messages


def search_query(question, history):
    """The message as a standalone English question, or the message itself when the
    rewrite fails or looks wrong (an answer, an apology, a long text)."""
    try:
        query = get_rewrite_llm().invoke(
            [SystemMessage(content=SEARCH_QUERY_SYSTEM)]
            + _history_messages(history[-3:])
            + [HumanMessage(content=question)]
        ).content.strip().strip('"«» ')
    except Exception:
        return question
    if not query or _NOT_A_QUERY.match(query) or len(query) > 3 * len(question) + 150:
        return question
    return query


def ask_question(question, history=None):
    history = history or []
    language = conversation_language(question, history)
    kind = route(question, history)

    if kind == "unclear":
        return UNCLEAR_REPLY[language]

    if kind == "small_talk":
        prompt = (
            f"Langue de réponse : {LANGUAGE_NAMES[language]}\n"
            f"Début de conversation : {'non' if history else 'oui'}\n\n"
            f"Message de l'utilisateur :\n{question}"
        )
        messages = (
            [SystemMessage(content=SMALL_TALK_SYSTEM)]
            + _history_messages(history)
            + [HumanMessage(content=prompt)]
        )
        return get_llm().invoke(messages).content.strip()

    docs = search(search_query(question, history))
    if not docs:
        return NO_DOCUMENTS_REPLY[language]

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    prompt = (
        f"Extraits :\n{context}\n\n"
        f"Langue de réponse : {LANGUAGE_NAMES[language]}\n\n"
        f"Message de l'utilisateur :\n{question}"
    )
    # The answer model also sees the recent exchanges, so it can keep the thread coherent.
    messages = (
        [SystemMessage(content=ANSWER_SYSTEM)]
        + _history_messages(history)
        + [HumanMessage(content=prompt)]
    )
    return get_llm().invoke(messages).content.strip()
