"""
Evaluate retrieval quality using gold source annotations.

Usage:
    python scripts/eval_retrieval.py [--output-dir data/eval]

Requires gold_source_ids column in questions.csv (task 2.1).
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

from app.deps import create_retriever, create_settings


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compute_metrics(
    questions: list[dict],
    output_dir: Path,
    use_reranker: bool = False,
    use_rewrite: bool = False,
) -> dict[str, Any]:
    """Run retrieval eval and return summary dict."""
    print("Loading retriever...")
    settings = create_settings()
    retriever = create_retriever(settings)

    rewriter = None
    if use_rewrite:
        from app.query_rewriter import QueryRewriter
        from app.llm_client import LLMClient
        print("Loading query rewriter...")
        llm = LLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
        rewriter = QueryRewriter(llm)

    reranker = None
    if use_reranker:
        from app.reranker import CrossEncoderReranker
        print("Loading reranker...")
        reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    k_values = [1, 3, 5, 10, 20, 40]
    eval_top_k = settings.rerank_candidate_k if use_reranker else 50
    total = len(questions)
    skipped = 0

    # Per-question results
    recall_at_k: dict[int, list[float]] = {k: [] for k in k_values}
    mrr_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []  # context recall
    no_gold_count = 0

    for i, q in enumerate(questions, start=1):
        qid = q.get("id", str(i))
        question = q.get("question", "").strip()
        gold_str = q.get("gold_source_ids", "").strip()

        if not question:
            continue

        if not gold_str or gold_str == "unknown":
            skipped += 1
            continue

        gold_ids = set(g.strip() for g in gold_str.split(";") if g.strip())
        if not gold_ids:
            skipped += 1
            continue

        # Optional query rewrite
        search_query = question
        if rewriter:
            search_query = rewriter.rewrite(question)

        # Retrieve with larger candidate pool for proper eval
        try:
            chunks = retriever.search(search_query, top_k=eval_top_k)
        except Exception as exc:
            print(f"  [{qid}] retrieval error: {exc}")
            for k in k_values:
                recall_at_k[k].append(0.0)
            mrr_values.append(0.0)
            precision_values.append(0.0)
            recall_values.append(0.0)
            continue

        # Optional reranker pass
        if reranker:
            chunks = reranker.rerank(search_query, chunks, top_k=settings.rerank_top_k)

        # Extract source_ids from retrieved chunks (preserve order)
        retrieved_sources = [c.get("source_id", "") for c in chunks]

        # ── recall@k ──
        for k in k_values:
            retrieved_k = set(retrieved_sources[:k])
            if gold_ids & retrieved_k:
                # At least one gold source found in top-k
                recall_at_k[k].append(1.0)
            else:
                recall_at_k[k].append(0.0)

        # ── MRR ──
        for rank, src in enumerate(retrieved_sources, start=1):
            if src in gold_ids:
                mrr_values.append(1.0 / rank)
                break
        else:
            mrr_values.append(0.0)

        # ── Context precision / recall ── (use top 10 as reference)
        retrieved_n = set(retrieved_sources[:10])
        hits = gold_ids & retrieved_n
        precision = len(hits) / min(10, len(retrieved_sources[:10])) if retrieved_sources[:10] else 0.0
        recall = len(hits) / len(gold_ids) if gold_ids else 0.0
        precision_values.append(precision)
        recall_values.append(recall)

        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] ...")

    evaluated = total - skipped

    # ── Aggregate ──
    def avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    metrics: dict[str, Any] = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total,
        "evaluated": evaluated,
        "skipped_no_gold": skipped,
    }
    for k in k_values:
        metrics[f"recall@{k}"] = avg(recall_at_k[k])
    metrics["mrr"] = avg(mrr_values)
    metrics["context_precision@10"] = avg(precision_values)
    metrics["context_recall@10"] = avg(recall_values)

    suffix = ""
    if use_rewrite:
        suffix += "_rewrite"
    if use_reranker:
        suffix += "_rerank"
    results_csv = output_dir / f"retrieval_results{suffix}.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "gold_source_ids", "mrr"] +
                         [f"recall@{k}" for k in k_values] +
                         ["precision@10", "recall@10"])
        idx = 0
        for q in questions:
            qid = q.get("id", "")
            gold_str = q.get("gold_source_ids", "").strip()
            if not gold_str or gold_str == "unknown":
                continue
            gold_ids = set(g.strip() for g in gold_str.split(";") if g.strip())
            if not gold_ids:
                continue
            row = [
                qid, q.get("question", ""), gold_str,
                mrr_values[idx] if idx < len(mrr_values) else 0,
            ]
            for k in k_values:
                row.append(recall_at_k[k][idx] if idx < len(recall_at_k[k]) else 0)
            row.append(precision_values[idx] if idx < len(precision_values) else 0)
            row.append(recall_values[idx] if idx < len(recall_values) else 0)
            writer.writerow(row)
            idx += 1

    summary_json = output_dir / f"retrieval_summary{suffix}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n  Results: {results_csv}")
    print(f"  Summary: {summary_json}")
    return metrics


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_reranker = "--rerank" in sys.argv
    use_rewrite = "--rewrite" in sys.argv
    output_dir = Path(args[0]) if args else ROOT / "data" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_path = ROOT / "data" / "eval" / "questions.csv"
    if not questions_path.exists():
        print(f"Error: {questions_path} not found. Run task 2.1 first.", file=sys.stderr)
        return 1

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")
    if use_reranker:
        print("Reranker: ENABLED")
    if use_rewrite:
        print("Query Rewrite: ENABLED")

    metrics = compute_metrics(questions, output_dir, use_reranker=use_reranker, use_rewrite=use_rewrite)

    print()
    print("─" * 50)
    print("Retrieval Evaluation Summary")
    print("─" * 50)
    print(f"  Evaluated:               {metrics['evaluated']}/{metrics['total_questions']}")
    for k in [1, 3, 5, 10, 20, 40]:
        key = f"recall@{k}"
        print(f"  {key:<24} {metrics[key]:.3f}")
    print(f"  MRR:                      {metrics['mrr']:.4f}")
    print(f"  Context Precision@10:     {metrics['context_precision@10']:.4f}")
    print(f"  Context Recall@10:        {metrics['context_recall@10']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
