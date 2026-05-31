"""
Phase 3.1 — LLM synthetic training data generation.

For each source document, uses Qwen3-8B to generate 3 diverse questions
(formal/colloquial/abbreviated). Retrieves hard negatives from real
BM25+vector top-40. Outputs (query, positive_chunk_id, hard_negative_chunk_ids).

Usage:
    PYTHONPATH=. python scripts/generate_training_pairs.py [--limit 10]
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = ROOT / "data" / "sources.csv"
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
OUTPUT_FILE = ROOT / "data" / "training" / "synthetic_pairs.jsonl"

GEN_PROMPT = """你是一个测试用例生成器。根据下面的校规文档内容，生成3个学生可能会问的问题。

文档内容：
{content}

要求：
1. 第一个问题用正式/书面语风格
2. 第二个问题用口语化/随意的风格（像学生聊天）
3. 第三个问题尽量简短（5-10个字）

每个问题一行，不要编号，不要其他内容。"""


def main():
    limit = 0
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])

    # Load chunks grouped by source
    chunks_by_source: dict[str, list[dict]] = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            sid = c["source_id"]
            if sid not in chunks_by_source:
                chunks_by_source[sid] = []
            chunks_by_source[sid].append(c)

    # Load sources
    sources = {}
    with open(SOURCES_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sources[row["source_id"]] = row

    # Init retriever and LLM
    from app.deps import create_retriever, create_settings
    from app.llm_client import LLMClient

    settings = create_settings()
    retriever = create_retriever(settings)
    llm = LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=120,
    )

    output_dir = OUTPUT_FILE.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    source_ids = list(chunks_by_source.keys())
    if limit:
        source_ids = source_ids[:limit]

    print(f"Generating training pairs for {len(source_ids)} sources...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, sid in enumerate(source_ids):
            src = sources.get(sid, {})
            chunks = chunks_by_source[sid]
            if not chunks:
                continue

            # Pick a representative chunk (longest one for rich content)
            chunk = max(chunks, key=lambda c: len(c.get("content", "")))
            content = chunk["content"][:800]  # limit to 800 chars for prompt

            # Generate questions
            try:
                prompt = GEN_PROMPT.format(content=content)
                response = llm.chat([{"role": "user", "content": prompt}], temperature=0.7)
                questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
                questions = [q.lstrip("1234567890. -–—") for q in questions][:3]
            except Exception as e:
                print(f"  [{i+1}/{len(source_ids)}] {sid}: LLM error: {e}")
                continue

            if len(questions) < 1:
                continue

            # For each question, retrieve hard negatives
            for q in questions:
                if len(q) < 3:
                    continue

                try:
                    candidates = retriever.search(q, top_k=40)
                except Exception:
                    continue

                # Positive: the chunk we generated from
                pos_id = chunk["chunk_id"]

                # Hard negatives: top-40 chunks from DIFFERENT sources
                neg_ids = []
                for c in candidates:
                    cid = c.get("chunk_id", "")
                    csid = c.get("source_id", "")
                    if csid != sid and cid != pos_id:
                        neg_ids.append(cid)
                    if len(neg_ids) >= 5:
                        break

                if not neg_ids:
                    continue

                pair = {
                    "query": q,
                    "positive_chunk_id": pos_id,
                    "hard_negative_chunk_ids": neg_ids,
                    "source_id": sid,
                    "title": src.get("title", ""),
                }
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total += 1

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(source_ids)}] {total} pairs generated...")
            time.sleep(0.5)  # polite throttle

    print(f"\nDone: {total} training pairs written to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
