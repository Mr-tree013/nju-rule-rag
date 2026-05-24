"""
Auto-evaluation script for the NJU Rule RAG Bot.

Calls POST /ask for every question in data/eval/questions.csv
and writes data/eval/results.csv + data/eval/summary.json.
"""

import csv
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_CSV = ROOT / "data" / "eval" / "questions.csv"
RESULTS_CSV = ROOT / "data" / "eval" / "results.csv"
SUMMARY_JSON = ROOT / "data" / "eval" / "summary.json"

ASK_URL = "http://127.0.0.1:8000/ask"
REQUEST_TIMEOUT = 60  # seconds


# ── helpers ────────────────────────────────────────────────────────

def is_refused(answer: str, sources: list) -> bool:
    """判断系统是否拒答。"""
    if not sources or len(sources) == 0:
        return True
    refuse_keywords = [
        "没有找到足够可靠的依据",
        "没有找到与您问题相关的足够可靠",
        "暂时不可用",
    ]
    for kw in refuse_keywords:
        if kw in answer:
            return True
    return False


def keyword_hit(expected_keywords: str, answer: str, sources: list) -> bool:
    """判断期望关键词是否出现在回答或来源标题中。"""
    if not expected_keywords or not expected_keywords.strip():
        return False
    keywords = [kw.strip() for kw in expected_keywords.replace("；", ";").replace("，", ",").split(",") if kw.strip()]
    if not keywords:
        # Try splitting by spaces or treating the whole string as one keyword
        keywords = [expected_keywords.strip()]

    # Check in answer
    for kw in keywords:
        if kw in answer:
            return True

    # Check in source titles
    for src in sources:
        title = src.get("title", "")
        for kw in keywords:
            if kw in title:
                return True

    return False


# ── main ───────────────────────────────────────────────────────────

def main() -> int:
    if not QUESTIONS_CSV.exists():
        print(f"Error: {QUESTIONS_CSV} not found.", file=sys.stderr)
        return 1

    with open(QUESTIONS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    if not questions:
        print("Error: questions.csv is empty.", file=sys.stderr)
        return 1

    print(f"Loaded {len(questions)} questions from {QUESTIONS_CSV}")
    print(f"Target: {ASK_URL}")
    print()

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    results = []
    success_count = 0
    error_count = 0

    for i, row in enumerate(questions, start=1):
        qid = row.get("id", str(i))
        question = row.get("question", "").strip()
        topic = row.get("topic", "").strip()
        expected_risk = row.get("risk_level", "").strip()
        expected_keywords = row.get("expected_source_keyword", "").strip()
        should_refuse = row.get("should_refuse", "").strip().lower() == "true"

        if not question:
            print(f"[{i}/{len(questions)}] SKIP: empty question (id={qid})")
            continue

        print(f"[{i}/{len(questions)}] {question[:50]}...", end=" ", flush=True)

        t_start = time.time()
        try:
            resp = requests.post(
                ASK_URL,
                json={"question": question},
                timeout=REQUEST_TIMEOUT,
            )
            latency = round(time.time() - t_start, 2)

            if resp.status_code == 200:
                data = resp.json()
                actual_risk = data.get("risk_level", "unknown")
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                source_count = len(sources)
                refused = is_refused(answer, sources)
                has_source = source_count > 0 and not refused
                kw_hit = keyword_hit(expected_keywords, answer, sources)

                result = {
                    "id": qid,
                    "question": question,
                    "topic": topic,
                    "expected_risk_level": expected_risk,
                    "actual_risk_level": actual_risk,
                    "should_refuse": should_refuse,
                    "answer": answer[:200],
                    "sources": json.dumps(sources, ensure_ascii=False)[:300],
                    "source_count": source_count,
                    "latency": latency,
                    "has_source": has_source,
                    "keyword_hit": kw_hit,
                    "refused": refused,
                    "error": "",
                }
                success_count += 1
                print(f"{latency}s risk={actual_risk} src={source_count} kw={'Y' if kw_hit else 'N'}")
            else:
                latency = round(time.time() - t_start, 2)
                result = {
                    "id": qid,
                    "question": question,
                    "topic": topic,
                    "expected_risk_level": expected_risk,
                    "actual_risk_level": "",
                    "should_refuse": should_refuse,
                    "answer": "",
                    "sources": "",
                    "source_count": 0,
                    "latency": latency,
                    "has_source": False,
                    "keyword_hit": False,
                    "refused": True,
                    "error": f"HTTP {resp.status_code}",
                }
                error_count += 1
                print(f"HTTP {resp.status_code}")
        except requests.Timeout:
            latency = round(time.time() - t_start, 2)
            result = {
                "id": qid, "question": question, "topic": topic,
                "expected_risk_level": expected_risk, "actual_risk_level": "",
                "should_refuse": should_refuse,
                "answer": "", "sources": "", "source_count": 0,
                "latency": latency, "has_source": False,
                "keyword_hit": False, "refused": True, "error": "timeout",
            }
            error_count += 1
            print("TIMEOUT")
        except requests.ConnectionError:
            latency = round(time.time() - t_start, 2)
            result = {
                "id": qid, "question": question, "topic": topic,
                "expected_risk_level": expected_risk, "actual_risk_level": "",
                "should_refuse": should_refuse,
                "answer": "", "sources": "", "source_count": 0,
                "latency": latency, "has_source": False,
                "keyword_hit": False, "refused": True, "error": "connection error",
            }
            error_count += 1
            print("CONNECTION ERROR")
        except Exception as exc:
            latency = round(time.time() - t_start, 2)
            result = {
                "id": qid, "question": question, "topic": topic,
                "expected_risk_level": expected_risk, "actual_risk_level": "",
                "should_refuse": should_refuse,
                "answer": "", "sources": "", "source_count": 0,
                "latency": latency, "has_source": False,
                "keyword_hit": False, "refused": True, "error": str(exc)[:100],
            }
            error_count += 1
            print(f"ERROR: {exc}")

        results.append(result)

    # ── write results.csv ──────────────────────────────────────────

    fieldnames = [
        "id", "question", "topic", "expected_risk_level", "actual_risk_level",
        "should_refuse", "answer", "sources", "source_count",
        "latency", "has_source", "keyword_hit", "refused", "error",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── summary ────────────────────────────────────────────────────

    total = len(results)
    has_source_count = sum(1 for r in results if r["has_source"])
    kw_hit_count = sum(1 for r in results if r["keyword_hit"])
    refused_count = sum(1 for r in results if r["refused"])
    avg_latency = round(sum(r["latency"] for r in results) / total, 2) if total else 0

    # High-risk conservative handling: high-risk questions that were NOT refused
    # should ideally have need_human_confirm or high actual_risk_level
    high_should_refuse = [r for r in results if r["should_refuse"] is True]
    high_refused = sum(1 for r in high_should_refuse if r["refused"])
    high_not_refused = sum(1 for r in high_should_refuse if not r["refused"])

    # Should refuse but didn't (bad — giving answers to high-risk)
    should_refuse_but_answered = [
        r for r in results
        if r["should_refuse"] is True and not r["refused"]
    ]

    # Should NOT refuse but did (bad — refusing answers to safe questions)
    should_answer_but_refused = [
        r for r in results
        if r["should_refuse"] is False and r["refused"]
    ]

    summary = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total,
        "success_count": success_count,
        "error_count": error_count,
        "avg_latency": avg_latency,
        "has_source_ratio": round(has_source_count / total, 3) if total else 0,
        "keyword_hit_ratio": round(kw_hit_count / total, 3) if total else 0,
        "high_risk_total": len(high_should_refuse),
        "high_risk_refused": high_refused,
        "high_risk_not_refused": high_not_refused,
        "should_refuse_but_answered": len(should_refuse_but_answered),
        "should_answer_but_refused": len(should_answer_but_refused),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── console output ─────────────────────────────────────────────

    print()
    print("─" * 50)
    print("Evaluation Summary")
    print("─" * 50)
    print(f"  Total questions:         {total}")
    print(f"  Success:                 {success_count}")
    print(f"  Errors:                  {error_count}")
    print(f"  Avg latency:             {avg_latency}s")
    print(f"  Has source ratio:        {has_source_count}/{total} ({summary['has_source_ratio']:.1%})")
    print(f"  Keyword hit ratio:       {kw_hit_count}/{total} ({summary['keyword_hit_ratio']:.1%})")
    print(f"  High-risk refused:       {high_refused}/{len(high_should_refuse)}")
    print(f"  Should-refuse answered:  {summary['should_refuse_but_answered']}  ← BAD if > 0")
    print(f"  Should-answer refused:   {summary['should_answer_but_refused']}  ← BAD if > 0")
    print()
    print(f"  Results: {RESULTS_CSV}")
    print(f"  Summary: {SUMMARY_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
