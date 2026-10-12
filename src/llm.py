"""Thin wrapper over any OpenAI-compatible chat endpoint.

Switch providers by changing LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env, e.g.
  OpenAI : https://api.openai.com/v1
  Gemini : https://generativelanguage.googleapis.com/v1beta/openai/
  Groq   : https://api.groq.com/openai/v1
  Ollama : http://localhost:11434/v1   (any non-empty API key)
"""
import sys
import time

from openai import (APIConnectionError, APITimeoutError, InternalServerError, OpenAI,
                    RateLimitError)

from src import config

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        # Fail fast on stalled requests and do our own retries, so a long run never hangs silently.
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "none",
                         timeout=60.0, max_retries=0)
    return _client


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0) -> dict:
    """Return {'text', 'latency_s', 'prompt_tokens', 'completion_tokens'}."""
    for attempt in range(5):
        t0 = time.perf_counter()
        try:
            resp = client().chat.completions.create(
                model=model or config.LLM_MODEL, messages=messages, temperature=temperature
            )
            break
        except RETRYABLE as e:
            if attempt == 4:
                raise
            wait = 5 * 2 ** attempt  # 5, 10, 20, 40 s
            print(f"  [llm] {type(e).__name__}; retry {attempt + 1}/4 in {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    usage = getattr(resp, "usage", None)
    return {
        "text": resp.choices[0].message.content or "",
        "latency_s": round(time.perf_counter() - t0, 2),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
