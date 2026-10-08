# pipelines/rag_chat.py

import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.services.rag.catalog import source
from core.services.rag.llm_backend import get_llm, get_rewrite_llm
from core.services.rag.retrieval import search

HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", "6"))
MAX_SOURCES = 3

LANGUAGE_NAMES = {"fr": "French", "en": "English"}
# The reply language goes last and in that language: the model follows it far more
# reliably there than in a French system prompt placed before the excerpts.
REPLY_IN = {"fr": "Répondez en français, en vouvoyant.", "en": "Reply in English."}

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
SERVICE_BUSY_REPLY = {
    "fr": "Le service d'intelligence artificielle met trop de temps à répondre. Veuillez réessayer dans un instant.",
    "en": "The AI service is taking too long to respond. Please try again in a moment.",
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
Write it as one standalone question in English about the subject itself, not about a
document: drop "what does the code / report / document say about" and "according to the
report", but keep what narrows the subject down (country, sector, group of people).
Use the conversation only to resolve a genuine follow-up (pronouns, "and ...?", "why?",
"yes"). If the message is a new topic, translate it on its own and do not mix in the
earlier topic. If it asks for nothing (thanks, agreement, a reaction such as "oh i see"),
reply exactly NONE. Never answer it. Reply with the question or NONE only, no label.

Acronyms used in the documents: CDN = contribution déterminée au niveau national
(nationally determined contribution, NDC); GIEC = IPCC; CDB = Convention on Biological
Diversity; GMV = Great Green Wall; EIES = environmental and social impact study;
ODD = Sustainable Development Goals; CNULCD = UNCCD; PNUE = UNEP; FAO = FAO.

Examples (message -> reply):
"Et les écoles, quel est leur rôle ?", after a question on the right to a healthy
environment -> What is the role of schools in the right to a healthy environment?
"Que dit la loi malienne sur l'eau à propos des forages ?" -> What rules apply to
drilling boreholes and wells in Mali?
"According to the UNEP report, how fast are species disappearing?" -> How fast are
species disappearing?
"cuisine" -> What is cooking?
"oui", after the assistant asked "Voulez-vous des exemples concrets de gestion des
déchets ?" -> What are concrete examples of waste management?
"ah je vois, merci" -> NONE"""

ANSWER_SYSTEM = """You are E-VIE, the assistant of a platform about environmental protection.

Rules:
- Write the whole reply in the language required by the last line of the message.
- Rely only on the excerpts provided and invent nothing. If they cover only part of the
  question, answer that part and say what is missing.
- Never mention "excerpts", "context" or "provided documents": present the information
  directly (if needed, say "according to our documents" / "d'après nos documents").
- Be clear and structured (lists, bold) without being longer than needed."""

SMALL_TALK_SYSTEM = f"""You are E-VIE, the assistant of a platform about environmental protection.
You can help with: {TOPICS['en']}.

The user's last message is a courtesy, a greeting, a reaction or a question about you.
Reply in one or two sentences, in the language required by the last line of the message.
Introduce yourself only if the conversation is starting; otherwise do not repeat your
introduction. End by offering your help."""

_SMALL_TALK = re.compile(
    r"^(ok(ay)?|d'accord|merci( beaucoup)?|thanks?( a lot)?|thank you( very much)?|super|"
    r"parfait|génial|bonjour|bonsoir|salut|coucou|hello|hi|hey|ça va|ca va|how are you|"
    r"au revoir|bye|goodbye|à bientôt|a bientot|cool|top|bien|nice|great|good|lol|mdr|haha|"
    r"qui es[- ]tu|who are you|que peux[- ]tu faire|what can you do|aide|help|"
    r"test(ing)?( ?\d+)?)[\s!.?]*$",
    re.IGNORECASE,
)
# Replies to an offer the assistant ended its answer with ("Voulez-vous des exemples ?").
_ACCEPT = re.compile(
    r"^(oui|yes|yeah|yep|sure|volontiers)( (please|merci|svp|s'il vous plaît))?[\s!.]*$", re.IGNORECASE
)
_DECLINE = re.compile(r"^(non|no|nope)( (merci|thanks|thank you))?[\s!.]*$", re.IGNORECASE)
# Words of messages that only acknowledge or react, however they are phrased:
# "oh i see", "ok merci", "got it", "c'est noté", "great, thanks!".
_ACKNOWLEDGEMENT_WORDS = set(
    "oh ah eh hmm hm ok okay okk d accord daccord merci beaucoup bien très tres bon super "
    "parfait génial genial top cool nice great good wow waouh i see je vois got it understood "
    "compris noté note c est ça ca makes sense thanks thank you so alright all right sure yeah "
    "yep yes oui non no nope entendu intéressant interessant interesting excellent bravo lol "
    "mdr haha".split()
)
# Labels the rewrite model sometimes copies from the examples ("Query:", "->").
_LABEL = re.compile(r"^\s*(->|→|query\s*:|output\s*:|requête\s*:|requete\s*:)\s*", re.IGNORECASE)
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


def _offer(history):
    """The question the assistant ended its last answer with, if any."""
    if not history:
        return None
    last_sentence = re.split(r"(?<=[.!?])\s+", history[-1][1].strip())[-1]
    return last_sentence if last_sentence.endswith("?") else None


def is_acknowledgement(text):
    """True when the message asks for nothing: no question mark, and nothing but
    interjections and politeness words."""
    if "?" in text:
        return False
    words = re.findall(r"[a-zàâäçéèêëîïôöùûüÿœ]+", text.lower())
    return bool(words) and all(word in _ACKNOWLEDGEMENT_WORDS for word in words)


def route(question, history):
    """'small_talk', 'unclear' or 'question'. Deciding from the message itself is as good
    as asking a model (see the adaptive-retrieval studies) and costs nothing."""
    text = question.strip()
    if _ACCEPT.match(text):
        return "question" if _offer(history) else "small_talk"
    if _DECLINE.match(text) or _SMALL_TALK.match(text) or is_acknowledgement(text):
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
    """The message as a standalone English question; None when it asks for nothing
    (the rewrite model says NONE); the message itself when the rewrite fails or looks
    wrong (an answer, an apology, a long text)."""
    try:
        query = get_rewrite_llm().invoke(
            [SystemMessage(content=SEARCH_QUERY_SYSTEM)]
            + _history_messages(history[-3:])
            + [HumanMessage(content=question)]
        ).content.strip()
    except Exception:
        return question
    query = _LABEL.sub("", query).strip().strip('"«» ')
    if query.upper().rstrip(".!") == "NONE":
        return None
    if not query or _NOT_A_QUERY.match(query) or len(query) > 3 * len(question) + 150:
        return question
    return query


def _small_talk_reply(question, history, language):
    prompt = (
        f"Conversation starting: {'no' if history else 'yes'}\n\n"
        f"Message:\n{question}\n\n"
        f"{REPLY_IN[language]}"
    )
    messages = (
        [SystemMessage(content=SMALL_TALK_SYSTEM)]
        + _history_messages(history)
        + [HumanMessage(content=prompt)]
    )
    return get_llm().invoke(messages).content.strip()


def ask_question(question, history=None):
    """{"answer": text, "sources": pages the answer was written from (may be empty)}."""
    history = history or []
    language = conversation_language(question, history)
    kind = route(question, history)

    if kind == "unclear":
        return {"answer": UNCLEAR_REPLY[language], "sources": []}

    # "oui" to an offer: search for what was offered. Phrased as a request, because the
    # rewrite model reads a bare "Would you like more details?" as asking for nothing.
    to_search = f"Yes, tell me more: {_offer(history)}" if _ACCEPT.match(question.strip()) else question
    query = search_query(to_search, history) if kind == "question" else None
    if query is None:  # small talk, or a message the rewrite model says asks for nothing
        return {"answer": _small_talk_reply(question, history, language), "sources": []}

    docs = search(query)
    if not docs:
        return {"answer": NO_DOCUMENTS_REPLY[language], "sources": []}

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    prompt = (
        f"Excerpts:\n{context}\n\n"
        f"Message:\n{question}\n\n"
        f"{REPLY_IN[language]}"
    )
    # The answer model also sees the recent exchanges, so it can keep the thread coherent.
    messages = (
        [SystemMessage(content=ANSWER_SYSTEM)]
        + _history_messages(history)
        + [HumanMessage(content=prompt)]
    )
    answer = get_llm().invoke(messages).content.strip()

    pages = []
    for doc in docs:  # best passages first
        page = (doc.metadata["source"], doc.metadata["page"])
        if page not in pages:
            pages.append(page)
    return {"answer": answer, "sources": [source(*page) for page in pages[:MAX_SOURCES]]}
