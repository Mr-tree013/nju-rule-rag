import os

from dotenv import load_dotenv

load_dotenv()


def _get(key, default=""):
    return os.getenv(key, default)


# ============================================================
# App
# ============================================================

APP_TITLE = _get("APP_TITLE", "NJU Rule RAG")

# ============================================================
# LLM
# ============================================================

LLM_API_KEY = _get("LLM_API_KEY")
LLM_BASE_URL = _get("LLM_BASE_URL")
LLM_MODEL = _get("LLM_MODEL")

# ============================================================
# Embedding
# ============================================================

EMBEDDING_API_KEY = _get("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = _get("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = _get("EMBEDDING_MODEL")

# ============================================================
# 数据路径
# ============================================================

DATA_DIR = _get("DATA_DIR", "data")
CHUNKS_FILE = _get("CHUNKS_FILE", "data/chunks/chunks.jsonl")
INDEX_DIR = _get("INDEX_DIR", "data/index")

# ============================================================
# 检索参数
# ============================================================

BM25_TOP_K = int(_get("BM25_TOP_K", "10"))
VECTOR_TOP_K = int(_get("VECTOR_TOP_K", "10"))
HYBRID_TOP_K = int(_get("HYBRID_TOP_K", "5"))


# ============================================================
# 工具函数
# ============================================================

def get_settings():
    """返回全部配置的字典，API Key 脱敏。"""
    return {
        "app_title": APP_TITLE,
        "llm_api_key": "***" if LLM_API_KEY else "(not set)",
        "llm_base_url": LLM_BASE_URL or "(not set)",
        "llm_model": LLM_MODEL or "(not set)",
        "embedding_api_key": "***" if EMBEDDING_API_KEY else "(not set)",
        "embedding_base_url": EMBEDDING_BASE_URL or "(not set)",
        "embedding_model": EMBEDDING_MODEL or "(not set)",
        "data_dir": DATA_DIR,
        "chunks_file": CHUNKS_FILE,
        "index_dir": INDEX_DIR,
        "bm25_top_k": BM25_TOP_K,
        "vector_top_k": VECTOR_TOP_K,
        "hybrid_top_k": HYBRID_TOP_K,
    }


def validate_config():
    """检查必要配置是否齐全，返回警告列表。"""
    warnings = []

    if not LLM_API_KEY:
        warnings.append("LLM_API_KEY 未设置，LLM 调用将失败。请在 .env 中填入 API Key。")
    if not LLM_MODEL:
        warnings.append("LLM_MODEL 未设置，LLM 调用将失败。请在 .env 中填入模型名。")

    return warnings
