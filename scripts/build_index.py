"""
Build BM25 and Chroma vector indices from chunks.jsonl.

Reads data/chunks/chunks.jsonl, builds search indices,
saves BM25 index and populates Chroma vector store.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import chromadb
import jieba
from chromadb.config import Settings
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
BM25_INDEX_FILE = INDEX_DIR / "bm25_index.pkl"
CHROMA_DIR = INDEX_DIR / "chroma"


def load_chunks():
    """Load all chunks from JSONL file."""
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def tokenize(text):
    """Chinese text tokenization using jieba."""
    return list(jieba.cut(text))


def build_bm25(chunks):
    """Build and save BM25 index."""
    print("Building BM25 index...")
    corpus = [c["content"] for c in chunks]
    tokenized = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump({"model": bm25, "chunks": chunks}, f)

    print(f"  BM25 index saved: {BM25_INDEX_FILE} ({len(chunks)} docs)")
    return bm25


def build_chroma(chunks):
    """Build and persist Chroma vector index."""
    print("Building Chroma vector index...")

    # Use Chroma's default ONNX embedding (all-MiniLM-L6-v2)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = "nju_rules"

    # Remove existing collection if present
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Add documents in batches to avoid memory issues
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[
                {
                    "source_id": c["source_id"],
                    "title": c["title"],
                    "priority": c["priority"],
                    "section": c.get("section", ""),
                }
                for c in batch
            ],
        )
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)} documents indexed")

    print(f"  Chroma index saved: {CHROMA_DIR}")
    return collection


def main():
    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} not found. Run build_chunks.py first.", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    build_bm25(chunks)
    build_chroma(chunks)

    print(f"\nIndices built successfully in {INDEX_DIR}/")


if __name__ == "__main__":
    main()
