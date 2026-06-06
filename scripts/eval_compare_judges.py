"""Compare two LLM judges on a sample of eval questions.

Runs two judges (default: deepseek-chat + qwen3:8b-nothink-v2) on the
same 30 questions and computes Spearman correlation between their scores.
A correlation < 0.6 indicates significant judge noise — the eval system
needs calibration.

Usage:
    python scripts/eval_compare_judges.py                    # default 30Q
    python scripts/eval_compare_judges.py --n 50             # 50 questions
    python scripts/eval_compare_judges.py --judge1 deepseek --judge2 gpt4
"""
import json, sys, csv, random
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "data" / "eval" / "results.csv"
GEN_SCORES = ROOT / "data" / "eval" / "gen_scores.csv"


def load_judge_scores(path):
    """Load F/R scores from a gen_scores CSV."""
    scores = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            qid = row.get("id", row.get("question_id", ""))
            try:
                f_score = int(float(row.get("faithfulness", 0)))
                r_score = int(float(row.get("relevance", 0)))
                scores[qid] = {"F": f_score, "R": r_score}
            except (ValueError, TypeError):
                continue
    return scores


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    n = 30
    for f in flags:
        if f.startswith("--n="):
            n = int(f.split("=", 1)[1])
        elif f == "--n" and len(args) > 0:
            n = int(args.pop(0))

    # Load existing DeepSeek scores
    deepseek = load_judge_scores(GEN_SCORES)
    if not deepseek:
        print("No DeepSeek scores found. Run eval_generation.py --deepseek first.")
        return 1

    # Sample N questions
    qids = list(deepseek.keys())
    if len(qids) > n:
        random.seed(42)
        qids = random.sample(qids, n)

    print(f"Comparing judges on {len(qids)} questions")
    print(f"Judge 1: deepseek-chat (existing scores)")

    # Load qwen3 judge scores if available
    qwen_path = ROOT / "data" / "eval" / "gen_qwen3-8b_scores.csv"
    if qwen_path.exists():
        qwen = load_judge_scores(qwen_path)
        common = [q for q in qids if q in qwen]
        if len(common) >= 10:
            ds_f = [deepseek[q]["F"] for q in common]
            qw_f = [qwen[q]["F"] for q in common]
            ds_r = [deepseek[q]["R"] for q in common]
            qw_r = [qwen[q]["R"] for q in common]

            r_f, p_f = spearmanr(ds_f, qw_f)
            r_r, p_r = spearmanr(ds_r, qw_r)

            print(f"Judge 2: qwen3:8b-nothink-v2 ({len(common)} common questions)")
            print(f"\nSpearman correlation:")
            print(f"  Faithfulness: r={r_f:.3f} (p={p_f:.3f}) {'OK' if abs(r_f) >= 0.6 else 'LOW — judge noise suspected'}")
            print(f"  Relevance:    r={r_r:.3f} (p={p_r:.3f}) {'OK' if abs(r_r) >= 0.6 else 'LOW — judge noise suspected'}")

            if abs(r_f) < 0.6 or abs(r_r) < 0.6:
                print(f"\nRecommendation: add a 3rd judge (GPT-4o-mini) and average scores.")
                print(f"  python scripts/eval_generation.py --judge gpt4o-mini")
        else:
            print(f"Only {len(common)} common questions — need >= 10 for correlation")
    else:
        print("No qwen3 judge scores found. Run:")
        print("  PYTHONPATH=. python scripts/eval_generation.py")
        print("  (without --deepseek, which uses the local qwen3 judge)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
