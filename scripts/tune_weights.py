"""
Grid-search hybrid retrieval weights to maximise MRR + recall@5.

Usage:
    python scripts/tune_weights.py [--step 0.05]

Outputs the top-10 weight combinations ranked by a composite score.
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

from app.config import RetrievalWeights, create_settings
from app.deps import create_retriever


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def evaluate_weights(
    retriever,
    questions: list[dict],
    weights: RetrievalWeights,
    top_k: int = 40,
) -> dict[str, float]:
    """Evaluate a single weight combination. Mutates retriever._weights."""
    retriever._weights = weights

    mrr_vals: list[float] = []
    r5_vals: list[float] = []
    r10_vals: list[float] = []

    for q in questions:
        gold_str = q.get("gold_source_ids", "").strip()
        if not gold_str or gold_str == "unknown":
            continue
        gold_ids = set(g.strip() for g in gold_str.split(";") if g.strip())
        if not gold_ids:
            continue

        chunks = retriever.search(q["question"], top_k=top_k)
        sources = [c.get("source_id", "") for c in chunks]

        # MRR
        for rank, src in enumerate(sources, start=1):
            if src in gold_ids:
                mrr_vals.append(1.0 / rank)
                break
        else:
            mrr_vals.append(0.0)

        # recall@5, recall@10
        r5_vals.append(1.0 if gold_ids & set(sources[:5]) else 0.0)
        r10_vals.append(1.0 if gold_ids & set(sources[:10]) else 0.0)

    n = len(mrr_vals) or 1
    mrr = sum(mrr_vals) / n
    r5 = sum(r5_vals) / n
    r10 = sum(r10_vals) / n

    # Composite: MRR × 0.4 + recall@5 × 0.3 + recall@10 × 0.3
    composite = mrr * 0.4 + r5 * 0.3 + r10 * 0.3

    return {"mrr": mrr, "recall@5": r5, "recall@10": r10, "composite": composite}


def main() -> int:
    step = 0.05
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--step" and i + 1 < len(sys.argv) - 1:
            step = float(sys.argv[i + 1])

    questions_path = ROOT / "data" / "eval" / "questions.csv"
    questions = load_questions(questions_path)
    # Filter to questions with gold sources
    questions = [
        q for q in questions
        if q.get("gold_source_ids", "").strip() and q["gold_source_ids"].strip() != "unknown"
    ]
    print(f"Loaded {len(questions)} questions with gold sources")

    print("Loading retriever...")
    retriever = create_retriever()

    # Grid search
    baseline = RetrievalWeights(bm25=0.45, vector=0.35, priority=0.20)
    results: list[tuple[RetrievalWeights, dict]] = []

    total = 0
    t0 = time.time()

    for bm25 in frange(0.0, 1.0 + step / 2, step):
        for vec in frange(0.0, 1.0 - bm25 + step / 2, step):
            priority = round(1.0 - bm25 - vec, 6)
            if priority < 0 or priority > 0.3:
                continue
            w = RetrievalWeights(bm25=bm25, vector=vec, priority=priority)
            if abs(w.bm25 + w.vector + w.priority - 1.0) > 0.02:
                continue

            metrics = evaluate_weights(retriever, questions, w)
            results.append((w, metrics))
            total += 1

            if total % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {total} combos evaluated ({elapsed:.0f}s) ...")

    # Sort by composite score
    results.sort(key=lambda x: x[1]["composite"], reverse=True)

    elapsed = time.time() - t0
    print(f"\nEvaluated {total} weight combinations in {elapsed:.0f}s")

    # Print top 10
    print(f"\n{'─' * 70}")
    print(f"{'Rank':<5} {'BM25':<8} {'Vector':<8} {'Priority':<10} {'MRR':<8} {'R@5':<8} {'R@10':<8} {'Score':<8}")
    print(f"{'─' * 70}")

    for rank, (w, m) in enumerate(results[:10], start=1):
        marker = " ← baseline" if (w.bm25 == baseline.bm25 and w.vector == baseline.vector) else ""
        print(f"{rank:<5} {w.bm25:<8.2f} {w.vector:<8.2f} {w.priority:<10.2f} "
              f"{m['mrr']:<8.4f} {m['recall@5']:<8.4f} {m['recall@10']:<8.4f} "
              f"{m['composite']:<8.4f}{marker}")

    # Print baseline for comparison
    base_metrics = evaluate_weights(retriever, questions, baseline)
    print(f"\nBaseline (0.45/0.35/0.20): MRR={base_metrics['mrr']:.4f} "
          f"R@5={base_metrics['recall@5']:.4f} R@10={base_metrics['recall@10']:.4f} "
          f"Score={base_metrics['composite']:.4f}")

    best_w, best_m = results[0]
    improvement = (best_m["composite"] - base_metrics["composite"]) / base_metrics["composite"] * 100
    print(f"\nBest: {best_w.bm25:.2f}/{best_w.vector:.2f}/{best_w.priority:.2f} "
          f"(+{improvement:.1f}% vs baseline)")

    # Write full results
    output = ROOT / "data" / "eval" / "weight_search.csv"
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bm25", "vector", "priority", "mrr", "recall@5", "recall@10", "composite"])
        for w, m in results:
            writer.writerow([w.bm25, w.vector, w.priority,
                             m["mrr"], m["recall@5"], m["recall@10"], m["composite"]])
    print(f"\nFull results: {output}")

    return 0


def frange(start, stop, step):
    """Float range generator."""
    while start < stop:
        yield round(start, 6)
        start += step


if __name__ == "__main__":
    sys.exit(main())
