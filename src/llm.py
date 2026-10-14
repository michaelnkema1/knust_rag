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
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY.strip() or "none",
                         timeout=60.0, max_retries=0)
    return _client


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0,
         max_retries: int | None = None) -> dict:
    """Return {'text', 'latency_s', 'prompt_tokens', 'completion_tokens'}.

    Retries rate limits / server errors with backoff (5, 10, 20, 40 s). Evals use the default (4);
    the web app sets LLM_MAX_RETRIES low so a visitor isn't left waiting over a minute.
    """
    retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            resp = client().chat.completions.create(
                model=model or config.LLM_MODEL, messages=messages, temperature=temperature
            )
            break
        except RETRYABLE as e:
            print(f"  [llm] {type(e).__name__}: {str(e)[:300]}", file=sys.stderr, flush=True)
            if attempt == retries:
                raise
            wait = 5 * 2 ** attempt
            print(f"  [llm] retry {attempt + 1}/{retries} in {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    usage = getattr(resp, "usage", None)
    return {
        "text": resp.choices[0].message.content or "",
        "latency_s": round(time.perf_counter() - t0, 2),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def chat_stream(messages: list[dict], model: str | None = None, temperature: float = 0.0,
                max_retries: int | None = None):
    """Yield the answer text in pieces as the model produces it.

    Retries (with the same backoff as chat) only happen before the first piece arrives;
    once text has been sent to the user, an error is raised instead of starting over.
    """
    retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    for attempt in range(retries + 1):
        try:
            stream = client().chat.completions.create(
                model=model or config.LLM_MODEL, messages=messages, temperature=temperature, stream=True
            )
            break
        except RETRYABLE as e:
            print(f"  [llm] {type(e).__name__}: {str(e)[:300]}", file=sys.stderr, flush=True)
            if attempt == retries:
                raise
            time.sleep(5 * 2 ** attempt)
    for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            yield event.choices[0].delta.content
