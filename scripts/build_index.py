"""
Build BM25 and Chroma vector indices from chunks.jsonl.

Reads data/chunks/chunks.jsonl, builds search indices, and generates:
  data/index/bm25.pkl          — BM25 index + chunk data
  data/index/chunk_lookup.json  — chunk_id → full chunk mapping
  data/index/manifest.json      — build metadata for app/retriever.py
  data/index/chroma/            — Chroma vector store (optional)
"""

import json
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

# Use the shared tokenizer so BM25 behaviour matches the online retriever.
from app.retriever import default_tokenizer as tokenize

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"

BM25_FILE = INDEX_DIR / "bm25.pkl"
LOOKUP_FILE = INDEX_DIR / "chunk_lookup.json"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
CHROMA_DIR = INDEX_DIR / "chroma"

# Read from Settings when available; fall back to env for standalone runs.
try:
    from app.config import create_settings
    _settings = create_settings()
    ENABLE_VECTOR = _settings.enable_vector
    LOCAL_EMBEDDING_MODEL = _settings.local_embedding_model
except ImportError:
    ENABLE_VECTOR = os.getenv("ENABLE_VECTOR", "true").lower() not in ("false", "0", "no")
    LOCAL_EMBEDDING_MODEL = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
    )


# ── helpers ────────────────────────────────────────────────────────

def _resolve_local_model(model_name: str) -> Path | None:
    """Return the local path to *model_name* if cached in data/models/."""
    short = model_name.split("/")[-1]
    for candidate in (
        ROOT / "data" / "models" / model_name.replace("/", "--"),
        ROOT / "data" / "models" / short,
    ):
        if candidate.exists():
            return candidate
    return None


def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ── BM25 ───────────────────────────────────────────────────────────

def build_bm25(chunks):
    print("Building BM25 index...")
    corpus = [c["content"] for c in chunks]
    tokenized = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Save BM25 model + chunk list together for retriever.
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"model": bm25, "chunks": chunks}, f)

    print(f"  BM25 index saved: {BM25_FILE} ({len(chunks)} docs)")
    return bm25


# ── chunk lookup ───────────────────────────────────────────────────

def build_chunk_lookup(chunks):
    """Generate chunk_id → chunk dict for fast lookups."""
    lookup = {c["chunk_id"]: c for c in chunks}

    with open(LOOKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)

    print(f"  Chunk lookup saved: {LOOKUP_FILE} ({len(lookup)} entries)")


# ── Chroma (optional) ──────────────────────────────────────────────

def build_chroma(chunks):
    """Build Chroma vector index.  Returns (embedding_model, status)."""
    embedding_model = None
    status = "ok"

    if not ENABLE_VECTOR:
        print("  ENABLE_VECTOR=false, skipping vector index.")
        return None, "skipped"

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("  ChromaDB not installed, skipping vector index.")
        return None, "skipped"

    # Clean old Chroma data before rebuild so stale UUID dirs don't pile up.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("  Cleaned old Chroma directory.")

    try:
        # Load embedding model — prefer local cache, fall back to HuggingFace.
        from sentence_transformers import SentenceTransformer
        local_model = _resolve_local_model(LOCAL_EMBEDDING_MODEL)
        if local_model:
            print(f"  Loading local model from {local_model} ...")
            embedder = SentenceTransformer(str(local_model), device="cpu")
        else:
            print(f"  Loading model from HuggingFace: {LOCAL_EMBEDDING_MODEL} ...")
            embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device="cpu")
        embedding_model = LOCAL_EMBEDDING_MODEL

        print(f"Building Chroma vector index (model: {LOCAL_EMBEDDING_MODEL})...")
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        collection_name = "nju_rules"
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["content"] for c in batch]
            vectors = embedder.encode(texts).tolist()
            collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=vectors,
                metadatas=[
                    {
                        "source_id": c["source_id"],
                        "title": c["title"],
                        "url": c.get("url", ""),
                        "priority": c["priority"],
                    }
                    for c in batch
                ],
            )
            print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)} documents indexed")

        print(f"  Chroma index saved: {CHROMA_DIR}")
    except Exception as exc:
        print(f"  Chroma index FAILED: {exc}")
        print("  BM25 index is still available.")
        status = f"failed: {exc}"

    return embedding_model, status


# ── manifest ───────────────────────────────────────────────────────

def write_manifest(chunk_count, embedding_model, vector_status):
    # Convert to paths relative to project root for portability.
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    manifest = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunks_file": rel(CHUNKS_FILE),
        "chunk_count": chunk_count,
        "bm25_index": rel(BM25_FILE),
        "chunk_lookup": rel(LOOKUP_FILE),
        "vector_index": rel(CHROMA_DIR) if CHROMA_DIR.exists() else None,
        "embedding_model": embedding_model,
        "status": vector_status,
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  Manifest saved: {MANIFEST_FILE}")


# ── main ───────────────────────────────────────────────────────────

def main():
    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} not found. Run build_chunks.py first.", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # BM25 — always succeeds
    build_bm25(chunks)

    # chunk lookup — always succeeds
    build_chunk_lookup(chunks)

    # Chroma — may fail gracefully
    embedding_model, vector_status = build_chroma(chunks)

    # manifest — always write
    write_manifest(len(chunks), embedding_model, vector_status)

    print(f"\nIndices built successfully in {INDEX_DIR}/")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
