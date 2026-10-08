"""Cloudflare Workers AI client factories.

Workers AI exposes an OpenAI-compatible API, so the regular LangChain OpenAI
classes work once they are pointed at the account endpoint. Everything is
read from the environment so the backend can be swapped without code changes.
"""

import os
from functools import lru_cache

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_AI_CHAT_MODEL = os.getenv("CF_AI_CHAT_MODEL", "@cf/mistralai/mistral-small-3.1-24b-instruct")
# Small fast model that rewrites the user's message as an English search query
# (GLM-4.7 without thinking follows "rewrite, do not answer" far better than Llama 8B).
CF_AI_REWRITE_MODEL = os.getenv("CF_AI_REWRITE_MODEL", "@cf/zai-org/glm-4.7-flash")
CF_AI_EMBED_MODEL = os.getenv("CF_AI_EMBED_MODEL", "@cf/baai/bge-m3")
# Cross-encoder that scores how well a passage answers the question (0-1). It also
# works for French questions on the English documents.
CF_AI_RERANK_MODEL = os.getenv("CF_AI_RERANK_MODEL", "@cf/baai/bge-reranker-base")


def _base_url():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError(
            "CF_ACCOUNT_ID and CF_API_TOKEN must be set to use Cloudflare Workers AI"
        )
    return f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1"


def _chat_client(model, max_tokens, timeout=120, max_retries=2):
    extra_body = {}
    # GLM models reason before answering; for a RAG chat that adds 10-30s and the
    # reasoning tokens count against max_tokens (an empty answer when exhausted).
    # GLM-4 honours the vLLM flag, GLM-5 only the Z.ai-style hint (it still
    # thinks a little, but far less).
    if "/glm-4" in model:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    elif "/glm-5" in model:
        extra_body["thinking"] = {"type": "disabled"}
        max_tokens = max(max_tokens, 1500)
    return ChatOpenAI(
        model=model,
        base_url=_base_url(),
        api_key=CF_API_TOKEN,
        temperature=0.2,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        extra_body=extra_body or None,
    )


# Request budgets: Cloudflare drops a request after 100 s, so the worst case of one chat
# turn (rewrite 8 + query embedding 2x10 + rerank 2x10 + answer 45) stays below that.
@lru_cache(maxsize=1)
def get_llm():
    """Model that writes the final answer."""
    return _chat_client(CF_AI_CHAT_MODEL, 800, timeout=45, max_retries=0)


@lru_cache(maxsize=1)
def get_rewrite_llm():
    """Model that turns the user's message into an English search query. Kept on a short
    leash: when it is slow, the original message is searched instead."""
    return _chat_client(CF_AI_REWRITE_MODEL, 200, timeout=8, max_retries=0)


def _embeddings_client(timeout, max_retries):
    # Workers AI expects raw strings, not tiktoken ids, and caps each request at
    # 60k tokens, so keep batches small.
    return OpenAIEmbeddings(
        model=CF_AI_EMBED_MODEL,
        base_url=_base_url(),
        api_key=CF_API_TOKEN,
        check_embedding_ctx_length=False,
        chunk_size=16,
        timeout=timeout,
        max_retries=max_retries,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Embeds questions while a user waits."""
    return _embeddings_client(timeout=10, max_retries=1)


@lru_cache(maxsize=1)
def get_ingestion_embeddings():
    """Embeds documents at startup, where patience beats a half-built index."""
    return _embeddings_client(timeout=120, max_retries=3)


def rerank(query, texts, attempts=2):
    """Relevance score (0-1) of each text for the query, in the order of `texts`."""
    _base_url()  # fail early with a clear message when credentials are missing
    for attempt in range(attempts):
        try:
            response = httpx.post(
                f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_AI_RERANK_MODEL}",
                headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
                json={"query": query, "contexts": [{"text": t} for t in texts], "top_k": len(texts)},
                timeout=10,
            )
            response.raise_for_status()
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError):
            if attempt == attempts - 1:
                raise
    scores = [0.0] * len(texts)
    for item in response.json()["result"]["response"]:
        scores[item["id"]] = item["score"]
    return scores
