"""Central settings, read from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # source PDFs (not committed)
INDEX_DIR = ROOT / "data" / "index"      # chunks + embeddings (committed, so servers need no rebuild)
MODEL_DIR = ROOT / "data" / "models"     # downloaded embedding model (not committed)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "0")) or None  # None = all cores

# Chunking (experiment variables)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))       # approx. words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# One index per chunk size, so several can exist side by side for experiments
CHUNKS_PATH = INDEX_DIR / f"chunks_{CHUNK_SIZE}.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / f"embeddings_{CHUNK_SIZE}.npy"

# Retrieval
TOP_K = int(os.getenv("TOP_K", "10"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "vector")  # vector | bm25 | hybrid

# LLM: any OpenAI-compatible endpoint (OpenAI, Gemini, Groq, Ollama, ...)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))  # set to 1 on the web server to fail fast
# Model that grades eval answers. Kept fixed so different answer models are judged the same way.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite")

# Answer prompt version (see SYSTEM_PROMPTS in src/answer.py)
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")

# Public copies of the source documents. Citation links redirect here when the PDF isn't on the server.
SOURCE_URLS = {
    "students_guide_2023.pdf":
        "https://www.knust.edu.gh/sites/default/files/2023-06/Students%27%20Guide%20and%20Code%20of%20Conduct.pdf",
}

# Abuse protection for the public demo: max /ask requests per client IP per minute (0 = off)
ASK_RATE_LIMIT = int(os.getenv("ASK_RATE_LIMIT", "8"))
