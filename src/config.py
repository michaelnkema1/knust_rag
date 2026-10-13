"""Central settings, read from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Chunking (experiment variables)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))       # approx. words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# One index per chunk size, so several can exist side by side for experiments
CHUNKS_PATH = ROOT / "data" / f"chunks_{CHUNK_SIZE}.jsonl"
COLLECTION = os.getenv("COLLECTION", f"knust_regs_{CHUNK_SIZE}")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "10"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "vector")  # vector | bm25 | hybrid

# LLM: any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, ...)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Answer prompt version (see SYSTEM_PROMPTS in src/answer.py)
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")
