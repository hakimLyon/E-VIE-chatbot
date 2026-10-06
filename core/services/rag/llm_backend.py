"""Cloudflare Workers AI client factories.

Workers AI exposes an OpenAI-compatible API, so the regular LangChain OpenAI
classes work once they are pointed at the account endpoint. Everything is
read from the environment so the backend can be swapped without code changes.
"""

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_AI_CHAT_MODEL = os.getenv("CF_AI_CHAT_MODEL", "@cf/zai-org/glm-4.7-flash")
CF_AI_EMBED_MODEL = os.getenv("CF_AI_EMBED_MODEL", "@cf/baai/bge-m3")


def _base_url():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError(
            "CF_ACCOUNT_ID and CF_API_TOKEN must be set to use Cloudflare Workers AI"
        )
    return f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1"


@lru_cache(maxsize=1)
def get_llm():
    extra_body = {}
    # GLM models reason before answering by default; for a RAG chat that only
    # adds 10-20s of latency and can eat the whole token budget (empty answer).
    if "/glm-" in CF_AI_CHAT_MODEL:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    return ChatOpenAI(
        model=CF_AI_CHAT_MODEL,
        base_url=_base_url(),
        api_key=CF_API_TOKEN,
        temperature=0.2,
        max_tokens=800,
        timeout=120,
        max_retries=2,
        extra_body=extra_body or None,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    # Workers AI expects raw strings, not tiktoken ids, and caps each request at
    # 60k tokens, so keep batches small.
    return OpenAIEmbeddings(
        model=CF_AI_EMBED_MODEL,
        base_url=_base_url(),
        api_key=CF_API_TOKEN,
        check_embedding_ctx_length=False,
        chunk_size=16,
        timeout=120,
        max_retries=2,
    )
