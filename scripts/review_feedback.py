"""Review negative feedback entries for manual labeling.

Usage:
    python scripts/review_feedback.py              # all dates
    python scripts/review_feedback.py 2026-06-01   # specific date
    python scripts/review_feedback.py --stats      # summary only
"""
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_DIR = ROOT / "data" / "feedback"


def main():
    stats_only = "--stats" in sys.argv
    dates = [a for a in sys.argv[1:] if not a.startswith("--")]
    files = sorted(FEEDBACK_DIR.glob("*.jsonl"))
    if dates:
        files = [f for f in files if f.stem in dates]
    if not files:
        print("No feedback files found.")
        return 1

    ratings = Counter()
    total = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    ratings[e.get("rating", "?")] += 1
                    total += 1
                    if not stats_only and e.get("rating") == "down":
                        print(f"\n{'='*60}")
                        print(f"[{e.get('ts','?')}] Q: {e.get('question','')[:100]}")
                        print(f"A: {e.get('answer','')[:200]}")
                        print(f"Comment: {e.get('comment','(none)')}")
                except json.JSONDecodeError:
                    continue

    print(f"\n{'='*60}")
    print(f"Total: {total} | Up: {ratings.get('up',0)} | Down: {ratings.get('down',0)} | Other: {ratings.get('other','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
