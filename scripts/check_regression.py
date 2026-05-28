"""
CI regression gate — compare latest eval results against baseline.

Usage:
    python scripts/check_regression.py [--run-eval] [--tolerance 0.05]

    --run-eval   Run eval_rag.py and eval_retrieval.py before checking
    --tolerance  Maximum allowed metric degradation (default 0.05 = 5%)

Exit 0 if all metrics pass, 1 if any metric regresses beyond tolerance.
"""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"

# Metrics to check: (latest_file, baseline_file, metric_key, direction, label)
# direction: "higher_better" or "lower_better"
CHECKS = [
    # ── Generation (from /ask eval) ──
    ("summary.json", "summary_baseline.json", "has_source_ratio", "higher_better", "有来源比例"),
    ("summary.json", "summary_baseline.json", "keyword_hit_ratio", "higher_better", "关键词命中率"),
    ("summary.json", "summary_baseline.json", "should_refuse_but_answered", "lower_better", "应拒未拒数"),
    ("summary.json", "summary_baseline.json", "should_answer_but_refused", "lower_better", "不应拒却拒数"),
    # ── Retrieval ──
    ("retrieval_summary.json", "retrieval_summary_baseline.json", "recall@5", "higher_better", "Recall@5"),
    ("retrieval_summary.json", "retrieval_summary_baseline.json", "recall@10", "higher_better", "Recall@10"),
    ("retrieval_summary.json", "retrieval_summary_baseline.json", "mrr", "higher_better", "MRR"),
]


def load_json(name: str) -> dict[str, Any]:
    path = EVAL_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_metric(
    latest: dict, baseline: dict, key: str, direction: str, label: str, tolerance: float
) -> tuple[bool, str]:
    """Check one metric. Returns (passed, message)."""
    cur = latest.get(key)
    base = baseline.get(key)

    if cur is None:
        return False, f"{label}: 当前值缺失"
    if base is None:
        return True, f"{label}: 基线值缺失，跳过 ({cur})"

    if direction == "higher_better":
        delta = cur - base
        threshold = base * tolerance
        if delta < -threshold:
            return False, f"{label}: {base:.4f} → {cur:.4f} ({delta:+.4f}) ⬇ 超过容忍度 {tolerance:.0%}"
        return True, f"{label}: {base:.4f} → {cur:.4f} ({delta:+.4f}) OK"
    else:  # lower_better
        delta = cur - base
        threshold = base * tolerance if base > 0 else tolerance
        if delta > threshold:
            return False, f"{label}: {base} → {cur} ({delta:+d}) ⬆ 超过容忍度 {tolerance:.0%}"
        return True, f"{label}: {base} → {cur} ({delta:+d}) OK"


def main() -> int:
    args = sys.argv[1:]
    tolerance = 0.05

    for i, arg in enumerate(args):
        if arg == "--tolerance":
            tolerance = float(args[i + 1]) if i + 1 < len(args) else tolerance
        elif arg == "--run-eval":
            import subprocess
            print("Running eval_rag.py...")
            subprocess.run([sys.executable, "scripts/eval_rag.py"], check=True, cwd=ROOT)
            print("Running eval_retrieval.py...")
            subprocess.run([sys.executable, "scripts/eval_retrieval.py"], check=True, cwd=ROOT,
                           env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})

    print(f"Regression check (tolerance={tolerance:.0%})\n")

    failures = 0
    for latest_file, baseline_file, key, direction, label in CHECKS:
        latest = load_json(latest_file)
        baseline = load_json(baseline_file)

        passed, msg = check_metric(latest, baseline, key, direction, label, tolerance)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")
        if not passed:
            failures += 1

    print()
    if failures == 0:
        print("All metrics pass — no regression detected.")
        return 0
    else:
        print(f"{failures} metric(s) regressed beyond tolerance — see above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
