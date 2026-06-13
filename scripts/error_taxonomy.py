"""Classify F<=2 evaluation questions into failure categories.

Usage:
    PYTHONPATH=. python scripts/error_taxonomy.py

Reads data/eval/gen_scores.csv + data/eval/results.csv, outputs
data/eval/error_taxonomy.csv with columns:
  id, question, F, category, detail

Categories:
  F1 - Retrieval failure (gold chunk not in top-5; genuine knowledge gap)
  F2 - Fabrication / ignores retrieval (model wrote plausible-but-false content)
  F3 - fact_check false positive (FC modified answer but shouldn't have)
  F4 - Tier 3 too aggressive (template-refused but sources had the answer)
  F5 - Judge noise (answer looks reasonable, judge likely erred)
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
    """Returns (category, detail) based primarily on judge notes."""
    f_score = int(float(gen_row.get("faithfulness", 3)))
    if f_score > 2:
        return "F_OK", ""

    notes = gen_row.get("notes", "").strip()
    question = result_row.get("question", "")
    answer = result_row.get("answer", "")

    # Parse debug from results
    debug = {}
    try:
        debug = json.loads(result_row.get("debug", "{}"))
    except (json.JSONDecodeError, TypeError):
        pass

    fc = debug.get("fact_check", {})
    tier = str(debug.get("confidence_tier", "?"))

    # Count sources
    sources = []
    try:
        sources = json.loads(result_row.get("sources", "[]"))
    except (json.JSONDecodeError, TypeError):
        cids = re.findall(r'"chunk_id"\s*:\s*"([^"]+)"', result_row.get("sources", ""))
        sources = [{"chunk_id": cid} for cid in cids]

    fc_modified = fc.get("hedged", 0) > 0 or fc.get("removed", 0) > 0
    fc_tier_changed = str(fc.get("tier_after_check", tier)) != tier

    # ── Classification based on judge notes ────────────────────────

    # F4: Tier 3 template refusal when sources HAD the answer
    # Notes pattern: "拒答" + "资料有/明确有/充分"
    if re.search(r"拒答", notes) and re.search(r"资料(有|明确|充分|明确写)", notes):
        # Check if answer is a template refusal
        if "没有足够的信息" in answer or "建议直接联系相关部门" in answer:
            return "F4", f"Tier 3 refusal but sources had info: {notes[:120]}"

    # F4 variant: "拒答不当" even without explicit template
    if re.search(r"拒答不当|错误拒答|却拒答", notes):
        return "F4", f"Wrong refusal — sources had info: {notes[:120]}"

    # F3: fact_check modified answer but maybe incorrectly
    # (rare — only flag if FC acted AND notes don't clearly say "编造")
    if fc_modified and fc_tier_changed:
        if not re.search(r"编造|脱离资料|未基于资料|完全编造|与参考资料无关", notes):
            return "F3", f"FC modified (hedged={fc.get('hedged',0)} removed={fc.get('removed',0)}) but notes don't confirm fabrication: {notes[:120]}"

    # F2: Fabrication — model wrote content not from sources
    # Notes patterns: "编造", "脱离参考资料", "未基于资料", "完全编造", "超出资料范围"
    if re.search(r"编造|脱离参考|未基于参考|完全编造|与参考资料无关|超出资料|未引用资料|未充分利用资料|不完整|未有效回答|未准确引用", notes):
        detail = notes[:150]
        return "F2", f"Fabrication/ignore-retrieval: {detail}"

    # F2 variant: sources had clear info, model claimed ignorance
    # e.g. "资料明确写无级差，回答却称没写清楚"
    if re.search(r"资料明确写.*(?:回答却|却称|却未|回答却称)", notes):
        return "F2", f"Sources had clear answer, model claimed ignorance: {notes[:150]}"

    # F2 variant: "遗漏关键" — missing key info from sources
    if re.search(r"遗漏关键|遗漏.*规则|遗漏.*流程", notes):
        return "F2", f"Missing key info: {notes[:150]}"

    # F2 variant: "回答模糊" — vague answer when sources have specifics
    if re.search(r"回答模糊|未引用资料中", notes):
        return "F2", f"Vague answer: {notes[:150]}"

    # F1: Genuine retrieval gap — sources legitimately don't cover this
    # Notes pattern: "拒答" without "资料有/明确有" modifier
    if re.search(r"拒答", notes):
        # Sources didn't have info → correct refusal, but F is low
        # This could be an eval question issue (should_refuse not set correctly)
        if len(sources) <= 2:
            return "F1", f"Genuine retrieval gap — correct refusal: {notes[:120]}"
        else:
            return "F1", f"Has sources but refused: {notes[:120]}"

    # ── Fallback heuristics (when notes are empty or uninformative) ──

    # Fallback F4: Tier 3 template without judge notes confirming fabrication
    if tier == "3" or fc.get("tier_after_check") == "3":
        if "没有足够的信息" in answer or "建议直接联系相关部门" in answer:
            return "F4", f"Tier 3 referral template (fallback heuristic, no judge notes)"

    # Fallback F1: Very few sources
    if len(sources) <= 1:
        return "F1", f"Only {len(sources)} sources retrieved (fallback heuristic)"

    # Fallback: check answer quality heuristics
    answer_len = len(answer)
    hedge_phrases = [
        "我看到的资料里没写", "建议问教务员", "建议你直接问", "资料里没有",
        "资料里没明说", "我看到的资料里确实没"
    ]
    hedge_count = sum(1 for p in hedge_phrases if p in answer)

    if answer_len < 100 or hedge_count >= 2:
        return "F2", f"Short ({answer_len} chars) or heavy hedging ({hedge_count}x) — fallback"

    # If we get here with no notes, it might be judge noise
    if not notes:
        return "F5", f"No judge notes, answer substantive ({answer_len} chars, {len(sources)} sources) — possible noise"

    # Notes exist but don't match any known failure pattern → uncertain, flag as F5
    return "F5", f"Notes don't match failure patterns: {notes[:150]}"


def main():
    if not GEN_CSV.exists():
        print(f"Missing {GEN_CSV}. Run eval_generation.py first.")
        return 1
    if not RESULTS_CSV.exists():
        print(f"Missing {RESULTS_CSV}. Run eval_rag.py first.")
        return 1

    with open(GEN_CSV, encoding="utf-8") as f:
        gen_scores = list(csv.DictReader(f))
    with open(RESULTS_CSV, encoding="utf-8") as f:
        results = list(csv.DictReader(f))

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
    print(f"Classified {sum(cats.values())} F<=2 questions (of {len(gen_scores)} total):\n")
    for cat in ["F1", "F2", "F3", "F4", "F5"]:
        count = cats.get(cat, 0)
        bar = "█" * max(1, count * 40 // total)
        print(f"  {cat}: {count:3d} ({count/total*100:5.1f}%) {bar}")
    print(f"\nOutput: {out_path}")

    # Print ROADMAP decision table
    print("\n── ROADMAP §1.3 决策表 ──")
    thresholds = {"F1": (0.30, "§3 检索增强"), "F2": (0.30, "§4 LoRA-v3"),
                  "F3": (0.20, "§2.1 fact_check调优"), "F4": (0.20, "§2.2 Tier阈值调优"),
                  "F5": (0.15, "§6 评测升级")}
    for cat in ["F1", "F2", "F3", "F4", "F5"]:
        pct = cats.get(cat, 0) / total
        thresh, action = thresholds[cat]
        triggered = "✅ TRIGGERED" if pct >= thresh else "—"
        print(f"  {cat}: {pct*100:5.1f}% (阈值≥{thresh*100:.0f}%) → {action} {triggered}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
