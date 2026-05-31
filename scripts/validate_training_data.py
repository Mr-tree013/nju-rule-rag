"""
Validate LoRA training data (PR #1).

Checks:
  - answer length in [20, 500]
  - refusal answers contain keywords
  - partial answers contain hedging phrases
  - token length distribution
  - type distribution matches expected ratios

Usage:
    python scripts/validate_training_data.py [--file data/training/lora_train.jsonl]
"""

import json, sys, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_REFUSAL_WORDS = ["没找到", "资料里没", "建议联系", "咨询", "教务处", "教务员"]
REQUIRED_PARTIAL_WORDS = ["具体", "没写", "没明说", "确认下", "教务员"]
EXPECTED_TYPES = {"full": (0.40, 0.60), "partial": (0.15, 0.35), "refusal": (0.12, 0.30)}


def validate(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs from {path}")
    errors = 0
    warnings = 0

    # 1. Type distribution
    types = Counter(p.get("type", "?") for p in pairs)
    total = len(pairs)
    print(f"\nType distribution:")
    for t, expected_range in EXPECTED_TYPES.items():
        count = types.get(t, 0)
        ratio = count / total if total else 0
        lo, hi = expected_range
        status = "OK" if lo <= ratio <= hi else "WARN"
        print(f"  {t}: {count} ({ratio:.1%}) [{lo:.0%}-{hi:.0%}] {status}")
        if status == "WARN":
            warnings += 1

    # 2. Answer length
    too_short = 0
    too_long = 0
    empty = 0
    for p in pairs:
        ans = p.get("answer", "")
        if not ans:
            empty += 1
        elif len(ans) < 20:
            too_short += 1
        elif len(ans) > 500:
            too_long += 1

    print(f"\nAnswer length:")
    print(f"  Empty: {empty} {'ERROR' if empty else 'OK'}")
    print(f"  Too short (<20): {too_short}")
    print(f"  Too long (>500): {too_long}")
    if empty:
        errors += empty
    if too_short > total * 0.05:
        warnings += 1

    # 3. Refusal keywords
    refusal_no_keyword = 0
    for p in pairs:
        if p.get("type") == "refusal":
            ans = p.get("answer", "")
            if not any(w in ans for w in REQUIRED_REFUSAL_WORDS):
                refusal_no_keyword += 1

    print(f"\nRefusal answers missing keywords: {refusal_no_keyword}/{types.get('refusal',0)}")
    if refusal_no_keyword > types.get('refusal', 0) * 0.2:
        errors += 1

    # 4. Partial hedging keywords
    partial_no_hedge = 0
    for p in pairs:
        if p.get("type") == "partial":
            ans = p.get("answer", "")
            if not any(w in ans for w in REQUIRED_PARTIAL_WORDS):
                partial_no_hedge += 1

    print(f"Partial answers missing hedge: {partial_no_hedge}/{types.get('partial',0)}")
    if partial_no_hedge > types.get('partial', 0) * 0.3:
        warnings += 1

    # 5. Sample print
    print(f"\nSample answers:")
    for t in ["full", "partial", "refusal"]:
        samples = [p for p in pairs if p.get("type") == t][:1]
        for p in samples:
            print(f"  [{t}] Q: {p['query'][:60]}")
            print(f"       A: {p['answer'][:150]}")
            print()

    print(f"\n{'PASSED' if errors == 0 else 'FAILED'}: {errors} errors, {warnings} warnings")
    return 0 if errors == 0 else 1


def main():
    path = ROOT / "data" / "training" / "lora_train.jsonl"
    for arg in sys.argv[1:]:
        if arg.startswith("--file="):
            path = Path(arg.split("=", 1)[1])
    return validate(path)


if __name__ == "__main__":
    sys.exit(main())
