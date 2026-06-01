"""Classify F<=2 evaluation questions into failure categories.

Usage:
    PYTHONPATH=. python scripts/error_taxonomy.py

Reads data/eval/gen_scores.csv + data/eval/results.csv, outputs
data/eval/error_taxonomy.csv with columns:
  id, question, F, category, detail

Categories:
  F1 - Retrieval failure (gold chunk not in top-5, answer has no references)
  F2 - Answer ignores retrieval (answer exists but doesn't use retrieved content)
  F3 - fact_check false positive (answer modified by fc, key info deleted)
  F4 - Tier downgrade too aggressive (downgraded to T3 unnecessarily)
  F5 - Judge noise (answer looks reasonable, judge may have erred)
"""

import csv, json, sys, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
GEN_CSV = ROOT / "data" / "eval" / "gen_scores.csv"
RESULTS_CSV = ROOT / "data" / "eval" / "results.csv"


def load_chunks() -> dict[str, str]:
    chunks_file = ROOT / "data" / "chunks" / "chunks.jsonl"
    lookup = {}
    if not chunks_file.exists():
        return lookup
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            try:
                c = json.loads(line)
                lookup[c.get("chunk_id", "")] = c.get("content", "")
            except json.JSONDecodeError:
                continue
    return lookup


def classify(gen_row: dict, result_row: dict, chunk_lookup: dict) -> tuple[str, str]:
    """Returns (category, detail)."""
    f_score = int(float(gen_row.get("faithfulness", 3)))
    if f_score > 2:
        return "F_OK", ""

    answer = result_row.get("answer", "")
    question = result_row.get("question", "")

    # Parse debug from results
    debug = {}
    try:
        debug = json.loads(result_row.get("debug", "{}"))
    except (json.JSONDecodeError, TypeError):
        pass

    fc = debug.get("fact_check", {})
    tier = debug.get("confidence_tier", "?")
    sources = []
    try:
        sources = json.loads(result_row.get("sources", "[]"))
    except (json.JSONDecodeError, TypeError):
        # Fallback: extract chunk_ids via regex for truncated JSON
        cids = re.findall(r'"chunk_id"\s*:\s*"([^"]+)"', result_row.get("sources", ""))
        sources = [{"chunk_id": cid} for cid in cids]

    # F3: fact_check modified answer
    fc_modified = fc.get("hedged", 0) > 0 or fc.get("removed", 0) > 0
    fc_tier_change = fc.get("tier_after_check", tier) != tier

    # F4: Tier 3 downgrade (too aggressive)
    if tier == "3" or fc.get("tier_after_check") == "3":
        # Check if answer is just a referral template
        if "我手头的校规资料里没有足够的信息" in answer:
            return "F4", "Tier 3 referral — may be too aggressive"

    # F3: fact_check modified but maybe incorrectly
    if fc_modified and fc_tier_change:
        return "F3", f"FC modified: hedged={fc.get('hedged',0)} removed={fc.get('removed',0)} unverified={fc.get('unverified',[])}"

    # F1: Retrieval failure — no sources or very few
    if len(sources) <= 1:
        return "F1", f"Only {len(sources)} sources retrieved"

    # Check if answer has substantive content vs just hedging
    answer_len = len(answer)
    hedge_phrases = [
        "我看到的资料里没写", "建议问教务员", "建议你直接问", "资料里没有",
        "资料里没明说", "我看到的资料里确实没"
    ]
    hedge_count = sum(1 for p in hedge_phrases if p in answer)

    # F2: Answer exists but heavy hedging = model not using retrieval well
    if answer_len < 100 or hedge_count >= 2:
        return "F2", f"Answer too short ({answer_len} chars) or heavy hedging ({hedge_count}x)"

    # Default: might be judge noise
    return "F5", f"Answer looks substantive ({answer_len} chars, {len(sources)} sources) — possible judge error"


def main():
    if not GEN_CSV.exists():
        print(f"Missing {GEN_CSV}. Run eval_generation.py first.")
        return 1
    if not RESULTS_CSV.exists():
        print(f"Missing {RESULTS_CSV}. Run eval_rag.py first.")
        return 1

    # Load data
    with open(GEN_CSV, encoding="utf-8") as f:
        gen_scores = list(csv.DictReader(f))
    with open(RESULTS_CSV, encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    # Build result lookup by id
    result_by_id = {r["id"]: r for r in results}
    chunk_lookup = load_chunks()

    cats = Counter()
    rows = []
    for g in gen_scores:
        rid = g.get("id", "")
        f_val = int(float(g.get("faithfulness", 3)))
        if f_val > 2:
            continue
        res = result_by_id.get(rid, {})
        cat, detail = classify(g, res, chunk_lookup)
        cats[cat] += 1
        rows.append({
            "id": rid,
            "question": g.get("question", "")[:80],
            "F": f_val,
            "category": cat,
            "detail": detail,
        })

    # Write output
    out_path = ROOT / "data" / "eval" / "error_taxonomy.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "F", "category", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(cats.values()) or 1
    print(f"Classified {sum(cats.values())} F<=2 questions:")
    for cat in ["F1", "F2", "F3", "F4", "F5"]:
        count = cats.get(cat, 0)
        bar = "█" * (count * 40 // total)
        print(f"  {cat}: {count:3d} ({count/total*100:5.1f}%) {bar}")
    print(f"\nOutput: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
