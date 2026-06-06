"""Export feedback down-votes as LoRA training pairs.

Reviews all down-voted answers, extracts (question, answer, sources)
triples, and outputs them in the training pair format for manual curation.

Usage:
    python scripts/export_training_pairs.py                 # all feedback
    python scripts/export_training_pairs.py --since 2026-06 # from date
    python scripts/export_training_pairs.py --output data/training/real_pairs.jsonl
"""
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_DIR = ROOT / "data" / "feedback"


def main():
    since = None
    output = None
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    for f in flags:
        if f.startswith("--since") and "=" not in f:
            idx = flags.index(f)
            if idx + 1 < len(args):
                since = args[idx]
        elif f.startswith("--since="):
            since = f.split("=", 1)[1]
        elif f.startswith("--output") and "=" not in f:
            idx = flags.index(f)
            if idx + 1 < len(args):
                output = args[idx]
        elif f.startswith("--output="):
            output = f.split("=", 1)[1]

    files = sorted(FEEDBACK_DIR.glob("*.jsonl"))
    if not files:
        print("No feedback files found in", FEEDBACK_DIR)
        return 1

    if since:
        files = [f for f in files if f.stem >= since]

    pairs = []
    stats = Counter()
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    stats[e.get("rating", "?")] += 1
                    if e.get("rating") == "down":
                        pair = {
                            "question": e.get("question", ""),
                            "bad_answer": e.get("answer", ""),
                            "sources": e.get("sources", ""),
                            "request_id": e.get("request_id", ""),
                            "comment": e.get("comment", ""),
                            "ts": e.get("ts", ""),
                            # Leave answer field empty for manual fill
                            "good_answer": "",
                        }
                        pairs.append(pair)
                except json.JSONDecodeError:
                    continue

    out_path = Path(output) if output else ROOT / "data" / "training" / "real_downvotes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Feedback: {sum(stats.values())} total | Up: {stats.get('up',0)} | Down: {stats.get('down',0)}")
    print(f"Exported {len(pairs)} down-voted pairs to {out_path}")
    print(f"Next: manually fill 'good_answer' field, then merge into lora_train.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
