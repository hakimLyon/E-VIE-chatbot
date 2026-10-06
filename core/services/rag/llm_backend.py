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
CF_AI_CHAT_MODEL = os.getenv("CF_AI_CHAT_MODEL", "@cf/mistralai/mistral-small-3.1-24b-instruct")
# Small fast model for the question-rewrite step (GLM-4.7 without thinking follows
# the "rewrite, do not answer" instruction far better than Llama 8B).
CF_AI_REWRITE_MODEL = os.getenv("CF_AI_REWRITE_MODEL", "@cf/zai-org/glm-4.7-flash")
CF_AI_EMBED_MODEL = os.getenv("CF_AI_EMBED_MODEL", "@cf/baai/bge-m3")


def _base_url():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError(
            "CF_ACCOUNT_ID and CF_API_TOKEN must be set to use Cloudflare Workers AI"
        )
    return f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1"


def _chat_client(model, max_tokens):
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
        timeout=120,
        max_retries=2,
        extra_body=extra_body or None,
    )


@lru_cache(maxsize=1)
def get_llm():
    """Model that writes the final answer."""
    return _chat_client(CF_AI_CHAT_MODEL, 800)


@lru_cache(maxsize=1)
def get_rewrite_llm():
    """Model that turns a follow-up into a standalone question."""
    return _chat_client(CF_AI_REWRITE_MODEL, 200)


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
