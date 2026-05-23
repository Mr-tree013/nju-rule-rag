import os
from dotenv import load_dotenv

load_dotenv()

APP_TITLE = os.getenv("APP_TITLE", "NJU Rule RAG")

# LLM
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Embedding
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

# Paths
DATA_DIR = os.getenv("DATA_DIR", "data")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "data/chunks/chunks.jsonl")
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")

# Retrieval
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "5"))
