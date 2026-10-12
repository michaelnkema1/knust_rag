"""Thin wrapper over any OpenAI-compatible chat endpoint.

Switch providers by changing LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env, e.g.
  OpenAI : https://api.openai.com/v1
  Groq   : https://api.groq.com/openai/v1
  Ollama : http://localhost:11434/v1   (any non-empty API key)
"""
import time

from openai import OpenAI

from src import config

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "none")
    return _client


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0) -> dict:
    """Return {'text', 'latency_s', 'prompt_tokens', 'completion_tokens'}."""
    t0 = time.perf_counter()
    resp = client().chat.completions.create(
        model=model or config.LLM_MODEL, messages=messages, temperature=temperature
    )
    usage = getattr(resp, "usage", None)
    return {
        "text": resp.choices[0].message.content or "",
        "latency_s": round(time.perf_counter() - t0, 2),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
