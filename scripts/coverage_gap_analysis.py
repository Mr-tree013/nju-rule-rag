"""
Coverage gap analysis — identify which topics need more source documents.

Computes per-topic metrics (faithfulness, recall@5, question count, source count)
and ranks topics by ROI priority for document acquisition.

Usage:
    PYTHONPATH=. python scripts/coverage_gap_analysis.py

Output: data/eval/coverage_gaps.csv (sorted by priority, highest first)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    # ── Load data ──────────────────────────────────────────────────
    questions = load_csv(ROOT / "data" / "eval" / "questions.csv")
    gen_scores = load_csv(ROOT / "data" / "eval" / "gen_scores.csv")
    retrieval = load_csv(ROOT / "data" / "eval" / "retrieval_results_rerank.csv")

    # Index gen_scores and retrieval by question id
    gen_by_id = {r["id"]: r for r in gen_scores}
    ret_by_id = {r["id"]: r for r in retrieval}

    # ── Topic → source mapping from config ─────────────────────────
    from app.config import create_settings
    settings = create_settings()
    topic_source_map: dict[str, set[str]] = defaultdict(set)
    for topic, src_ids in settings.topic_route_map.items():
        for sid in src_ids:
            topic_source_map[topic].add(sid)

    # ── Aggregate per topic ────────────────────────────────────────
    topic_data: dict[str, dict] = defaultdict(lambda: {
        "questions": 0,
        "faithfulness_sum": 0.0,
        "relevance_sum": 0.0,
        "recall5_sum": 0.0,
        "gold_rank_sum": 0.0,
        "mrr_sum": 0.0,
    })

    for q in questions:
        qid = q.get("id", "")
        topic = q.get("topic", "").strip()
        if not topic:
            continue

        topic_data[topic]["questions"] += 1

        gen = gen_by_id.get(qid)
        if gen:
            try:
                topic_data[topic]["faithfulness_sum"] += float(gen["faithfulness"])
                topic_data[topic]["relevance_sum"] += float(gen["relevance"])
            except (ValueError, KeyError):
                pass

        ret = ret_by_id.get(qid)
        if ret:
            try:
                topic_data[topic]["recall5_sum"] += float(ret["recall@5"])
                topic_data[topic]["gold_rank_sum"] += float(ret.get("gold_chunk_rank", 0))
                topic_data[topic]["mrr_sum"] += float(ret["mrr"])
            except (ValueError, KeyError):
                pass

    # ── Compute per-topic metrics ──────────────────────────────────
    rows = []
    for topic, data in sorted(topic_data.items()):
        n = data["questions"]
        if n == 0:
            continue

        source_count = len(topic_source_map.get(topic, set()))
        avg_f = data["faithfulness_sum"] / n if n else 0
        avg_r = data["relevance_sum"] / n if n else 0
        avg_recall5 = data["recall5_sum"] / n if n else 0
        avg_mrr = data["mrr_sum"] / n if n else 0
        avg_gold_rank = data["gold_rank_sum"] / n if n else 0

        # Priority: higher = more urgent to add sources.
        # Weighted by question count, penalized by low faithfulness,
        # inversely proportional to existing source coverage.
        unfaithfulness = max(0, 5 - avg_f)  # how much room to improve
        priority = round(n * unfaithfulness / max(source_count, 1), 2)

        rows.append({
            "topic": topic,
            "questions": n,
            "sources": source_count,
            "avg_faithfulness": round(avg_f, 2),
            "avg_relevance": round(avg_r, 2),
            "avg_recall5": round(avg_recall5, 3),
            "avg_mrr": round(avg_mrr, 3),
            "avg_gold_chunk_rank": round(avg_gold_rank, 1),
            "f_leq_2_pct": f"{round(data['faithfulness_sum']/n*100) if n else 0}%",
            "priority_score": priority,
            "recommendation": "",
        })

    # Sort by priority descending
    rows.sort(key=lambda r: r["priority_score"], reverse=True)

    # ── Generate recommendations ───────────────────────────────────
    for r in rows:
        if r["priority_score"] >= 8:
            r["recommendation"] = "URGENT: add 5+ sources immediately"
        elif r["priority_score"] >= 5:
            r["recommendation"] = "HIGH: add 3-5 sources this week"
        elif r["priority_score"] >= 3:
            r["recommendation"] = "MEDIUM: add 1-3 sources when possible"
        else:
            r["recommendation"] = "LOW: current coverage adequate"

    # ── Write output ───────────────────────────────────────────────
    out_path = ROOT / "data" / "eval" / "coverage_gaps.csv"
    fieldnames = [
        "topic", "questions", "sources", "avg_faithfulness", "avg_relevance",
        "avg_recall5", "avg_mrr", "avg_gold_chunk_rank", "f_leq_2_pct",
        "priority_score", "recommendation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Print summary ──────────────────────────────────────────────
    print("Coverage Gap Analysis — Priority Ranking")
    print("=" * 78)
    print(f"{'Rank':<5} {'Topic':<16} {'Q':>3} {'Src':>4} {'F':>5} {'R@5':>6} {'Priority':>9}  Recommendation")
    print("-" * 78)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:<5} {r['topic']:<16} {r['questions']:>3} {r['sources']:>4} "
            f"{r['avg_faithfulness']:>5.2f} {r['avg_recall5']:>6.3f} "
            f"{r['priority_score']:>9.1f}  {r['recommendation']}"
        )

    print(f"\nOutput: {out_path}")
    print(f"Top 5 topics to address: {', '.join(r['topic'] for r in rows[:5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
