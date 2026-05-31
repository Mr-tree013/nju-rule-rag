"""
Generate training answers using DeepSeek API (PR #1).

Reads all_pairs_filtered.jsonl, generates 3 types of answers:
  50% full (complete answer from chunk)
  25% partial (hedged answer with missing details)
  20% refusal (no relevant info → refer to department)
  5% general (alpaca-zh style, added separately)

Output: data/training/lora_train.jsonl, lora_holdout.jsonl, lora_meta.json

Usage:
    python scripts/generate_training_answers.py [--spot-check 30]
    python scripts/generate_training_answers.py --dry-run 10
"""

import json, random, re, sys, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "training" / "all_pairs_filtered.jsonl"
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
OUT_TRAIN = ROOT / "data" / "training" / "lora_train.jsonl"
OUT_HOLDOUT = ROOT / "data" / "training" / "lora_holdout.jsonl"
OUT_META = ROOT / "data" / "training" / "lora_meta.json"

HOLDOUT_SIZE = 300
RATIOS = {"full": 0.50, "partial": 0.25, "refusal": 0.20}

# Prompts for each answer type
FULL_PROMPT = """你是一个高校教务问答助手。基于以下资料,用学长的口吻自然回答学生的问题。只用资料里的信息,不要补充资料里没有的内容。

资料: {context}

问题: {query}

请直接输出回答(150-300字):"""

PARTIAL_PROMPT = """你是一个高校教务问答助手。基于以下资料回答学生问题。
注意:资料里只包含部分信息,具体数字/日期/金额可能缺失。
如果有缺失,在那一句末尾自然地加上「具体XX我看到的资料里没写,建议问教务员」。
不要编造任何资料里没有的数字或细节。

资料: {context}

问题: {query}

请直接输出回答(150-300字):"""

REFUSAL_PROMPT = """你是一个高校教务助手。以下资料都和学生的提问不直接相关,你无法从中找到答案。
请用学长的口吻诚实告诉学生:这个问题你在手头的校规资料里没找到具体规定,建议直接联系相关部门咨询。
可以给1-2个可能的联系渠道(如:教务处电话 89681234、教务系统 jw.nju.edu.cn)。

资料(与问题不直接相关): {context}

问题: {query}

请直接输出回答(80-150字):"""


def load_chunks():
    chunks = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c["content"]
    return chunks


def strip_details(text: str) -> str:
    """Remove specific numbers/dates/amounts for partial answer generation."""
    text = re.sub(r'\d+元', 'XX元', text)
    text = re.sub(r'\d+学分', 'XX学分', text)
    text = re.sub(r'\d+月\d+日', 'X月X日', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}', 'XXXX-XX-XX', text)
    text = re.sub(r'\d{4}年', 'XXXX年', text)
    text = re.sub(r'\d+%', 'XX%', text)
    text = re.sub(r'第\d+条', '第X条', text)
    text = re.sub(r'\d+\.\s', 'X. ', text)
    return text


def call_deepseek(prompt: str, api_key: str, max_retries: int = 3) -> str:
    """Call DeepSeek API with retry."""
    import requests
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:100]}", flush=True)
                time.sleep(1)
        except Exception as e:
            print(f"  Request error: {e}", flush=True)
            time.sleep(1)
    return ""


def main():
    dry_run = 0
    spot_check = 0
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--dry-run" and i + 1 < len(sys.argv):
            dry_run = int(sys.argv[i + 1])
        elif arg == "--spot-check" and i + 1 < len(sys.argv):
            spot_check = int(sys.argv[i + 1])

    # Load API key from .env
    env_file = ROOT / ".env"
    api_key = ""
    if env_file.exists():
        for line in open(env_file):
            if line.startswith("FALLBACK_LLM_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
            elif line.startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
    if not api_key:
        print("ERROR: No DeepSeek API key found. Set FALLBACK_LLM_API_KEY in .env")
        return 1

    # Load data
    chunks = load_chunks()
    pairs = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["positive_chunk_id"] in chunks:
                pairs.append(p)

    if dry_run:
        pairs = random.sample(pairs, min(dry_run, len(pairs)))
        print(f"DRY RUN: {len(pairs)} samples")

    # Shuffle and split by source_id
    random.seed(42)
    random.shuffle(pairs)
    
    # Collect unique sources for holdout split
    all_sources = list(set(p["source_id"] for p in pairs))
    random.shuffle(all_sources)
    
    # Pick sources for holdout (aim for ~300 pairs)
    holdout_sources = set()
    holdout_count = 0
    for src in all_sources:
        holdout_sources.add(src)
        holdout_count += sum(1 for p in pairs if p["source_id"] == src)
        if holdout_count >= HOLDOUT_SIZE:
            break
    
    train_pairs = [p for p in pairs if p["source_id"] not in holdout_sources]
    holdout_pairs = [p for p in pairs if p["source_id"] in holdout_sources]
    
    print(f"Train: {len(train_pairs)}, Holdout: {len(holdout_pairs)} (from {len(holdout_sources)} sources)")

    # Assign types to each pair
    type_counts = Counter()
    typed_pairs = []
    
    for p in train_pairs + holdout_pairs:
        # Weighted random assignment
        r = random.random()
        if r < RATIOS["full"]:
            ptype = "full"
        elif r < RATIOS["full"] + RATIOS["partial"]:
            ptype = "partial"
        else:
            ptype = "refusal"
        type_counts[ptype] += 1
        typed_pairs.append((p, ptype))

    print(f"Types: {dict(type_counts)}")

    # Generate answers
    total = len(typed_pairs)
    for i, (pair, ptype) in enumerate(typed_pairs):
        query = pair["query"]
        pos_id = pair["positive_chunk_id"]
        pos_content = chunks[pos_id]
        
        if ptype == "full":
            ctx = pos_content[:1200]
            prompt = FULL_PROMPT.format(context=ctx, query=query)
        elif ptype == "partial":
            ctx = strip_details(pos_content)[:1200]
            prompt = PARTIAL_PROMPT.format(context=ctx, query=query)
        else:  # refusal
            neg_ids = pair.get("hard_negative_chunk_ids", [])[:3]
            neg_texts = [chunks.get(cid, "")[:400] for cid in neg_ids]
            ctx = "\n---\n".join(neg_texts) if neg_texts else "（无相关资料）"
            prompt = REFUSAL_PROMPT.format(context=ctx, query=query)
        
        answer = call_deepseek(prompt, api_key)
        if not answer:
            print(f"  [{i+1}/{total}] FAILED: {query[:50]}...", flush=True)
            continue
        
        pair["answer"] = answer
        pair["type"] = ptype
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] {dict(Counter(p['type'] for p,_ in typed_pairs[:i+1]))}", flush=True)
        time.sleep(0.3)  # Rate limit

    # Split back into train/holdout
    train_final = [p for p, _ in typed_pairs if p["source_id"] not in holdout_sources]
    holdout_final = [p for p, _ in typed_pairs if p["source_id"] in holdout_sources]

    # Write outputs
    for path, data in [(OUT_TRAIN, train_final), (OUT_HOLDOUT, holdout_final)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in data:
                if "answer" in p:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Meta
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pairs": len(pairs),
        "train_count": len([p for p in train_final if "answer" in p]),
        "holdout_count": len([p for p in holdout_final if "answer" in p]),
        "type_distribution": dict(type_counts),
        "holdout_sources": sorted(holdout_sources),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Train: {meta['train_count']}, Holdout: {meta['holdout_count']}")
    print(f"Output: {OUT_TRAIN}, {OUT_HOLDOUT}, {OUT_META}")

    # Spot check
    if spot_check:
        print(f"\n=== SPOT CHECK ({spot_check} samples) ===")
        samples = random.sample([p for p, _ in typed_pairs if "answer" in p], min(spot_check, meta['train_count']))
        for i, p in enumerate(samples):
            print(f"\n[{i+1}] Type={p['type']} Q: {p['query'][:60]}")
            print(f"   A: {p['answer'][:200]}")
            print(f"   ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
