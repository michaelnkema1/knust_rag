"""Central settings, read from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"

COLLECTION = os.getenv("COLLECTION", "knust_regs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Chunking (experiment variables)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))       # approx. words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "vector")  # vector | bm25 | hybrid

# LLM: any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, ...)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
