"""
Dependency injection — single place where all components are wired together.

Every ``create_*`` function accepts an optional ``Settings`` argument so
tests and scripts can inject custom configuration.
"""

from pathlib import Path

from dotenv import load_dotenv

from app.config import Settings, create_settings
from app.llm_client import LLMClient
from app.pipeline import RAGPipeline
from app.retriever import (
    BM25Retriever,
    HybridRetriever,
    VectorRetriever,
)

# Ensure .env is loaded before constructing Settings.
load_dotenv()


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _get_settings() -> Settings:
    return create_settings()


def create_retriever(settings: Settings | None = None) -> HybridRetriever:
    """Build a ``HybridRetriever`` from *settings* (or the environment)."""
    s = settings or _get_settings()
    root = s.project_root
    chunks = _resolve(root, s.chunks_file)
    index = _resolve(root, s.index_dir)

    bm25 = BM25Retriever(
        chunks_path=chunks,
        index_path=index / "bm25.pkl",
        chunk_lookup_path=index / "chunk_lookup.json",
    )
    vector = VectorRetriever(
        chroma_path=index / "chroma",
        chunks_path=chunks,
        chunk_lookup_path=index / "chunk_lookup.json",
        embedding_model=s.local_embedding_model,
        enable=s.enable_vector,
    )
    retriever = HybridRetriever(
        bm25=bm25,
        vector=vector,
        manifest_path=index / "manifest.json",
        weights=s.retrieval_weights,
        top_k=s.hybrid_top_k,
    )
    status = retriever.status()
    print(
        f"[Retriever] BM25={status['bm25_loaded']}({status['bm25_chunks']} chunks), "
        f"Vector={status['vector_loaded']}"
    )
    return retriever


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build an ``LLMClient`` from *settings* (or the environment)."""
    s = settings or _get_settings()
    return LLMClient(
        api_key=s.llm_api_key,
        base_url=s.llm_base_url,
        model=s.llm_model,
        retry_count=s.retry_count,
        retry_delays=s.retry_delays,
        timeout=s.request_timeout,
    )


def create_pipeline(settings: Settings | None = None) -> RAGPipeline:
    """Build a fully wired ``RAGPipeline`` ready to answer questions."""
    s = settings or _get_settings()
    retriever = create_retriever(s)
    llm = create_llm_client(s)
    pipeline = RAGPipeline(retriever=retriever, llm=llm, settings=s)
    warnings = s.validate()
    if warnings:
        print("[Config] 警告:")
        for w in warnings:
            print(f"  - {w}")
    return pipeline
