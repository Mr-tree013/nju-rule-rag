"""
NER-level fact checking for LLM output (Phase 1.1).

Extracts verifiable entities (numbers, dates, URLs, emails, phones, proper nouns)
from the LLM answer and checks each against the retrieved source chunks.

Three-tier response:
  - Zero-tolerance (numbers, amounts, URLs, phones): remove sentence if not found
  - Soft (dates, proper nouns): hedge with "具体的X资料里没写"
  - Severe: >=3 failures or any zero-tolerance → downgrade to Tier 3
"""

import re
from typing import Any

# ── Extraction patterns ──────────────────────────────────────────

# Numbers with units: "500元", "20学分", "3次", "50%", "4人间", "14个学分"
# Note: excludes 天/周/月/年 (handled by DATE_RE) and 人/个 (too common)
AMOUNT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:元|块|学分|次|%|万元|千元|百元)"
)

# Time patterns: "12:00", "9月15日", "2024年", "第3周", "3-8周"
DATE_RE = re.compile(
    r"(?:\d{4}[-/年])?\d{1,2}[-/月]\d{1,2}[日号]?|"
    r"第\s*\d+\s*[周学期]|"
    r"\d+:\d{2}|"
    r"\d{4}-\d{2}-\d{2}"
)

# URLs
URL_RE = re.compile(r"https?://[^\s,。；，]+")

# Email
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")

# Phone numbers: 7-11 digits
PHONE_RE = re.compile(r"(?<!\d)(?:\(?\d{3,4}\)?-?\s?)?\d{7,8}(?!\d)")

# Proper nouns in quotes or brackets: 「...」, 《...》, "..." (system names, document titles)
PROPER_NOUN_RE = re.compile(r"[「《\"]([^》」\"]{3,30})[》」\"]")


def extract_facts(text: str) -> list[dict[str, str]]:
    """Extract all verifiable fact items from answer text.

    Returns list of {type, value, sentence}.
    """
    facts: list[dict[str, str]] = []

    # Split into sentences for context
    sentences = re.split(r"[。！？\n]", text)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Zero-tolerance items
        for m in AMOUNT_RE.finditer(sent):
            facts.append({"type": "amount", "value": m.group(), "sentence": sent[:120]})
        for m in URL_RE.finditer(sent):
            facts.append({"type": "url", "value": m.group(), "sentence": sent[:120]})
        for m in EMAIL_RE.finditer(sent):
            facts.append({"type": "email", "value": m.group(), "sentence": sent[:120]})
        for m in PHONE_RE.finditer(sent):
            val = m.group().strip()
            if len(val) >= 7:
                facts.append({"type": "phone", "value": val, "sentence": sent[:120]})

        # Soft items
        for m in DATE_RE.finditer(sent):
            facts.append({"type": "date", "value": m.group(), "sentence": sent[:120]})
        for m in PROPER_NOUN_RE.finditer(sent):
            facts.append({"type": "proper_noun", "value": m.group(1), "sentence": sent[:120]})

    return facts


def _is_zero_tolerance(fact_type: str) -> bool:
    return fact_type in ("amount", "url", "email", "phone")


def verify_facts(
    facts: list[dict[str, str]],
    chunks: list[dict],
) -> list[dict[str, Any]]:
    """Verify each fact against the retrieved chunks.

    Returns facts with 'verified' field added.
    """
    # Build search corpus from all chunks
    corpus = "\n".join(c.get("content", "") for c in chunks)

    results = []
    for f in facts:
        val = f["value"]
        ftype = f["type"]

        if ftype in ("url", "email", "phone"):
            # Exact match required
            verified = val in corpus
        elif ftype == "amount":
            # Exact match (数字+单位必须逐字出现)
            verified = val in corpus
        elif ftype == "date":
            # Allow ±1 char tolerance for full-width half-width differences
            verified = val in corpus or _fuzzy_date_match(val, corpus)
        elif ftype == "proper_noun":
            # Must appear as a substring
            verified = val in corpus
        else:
            verified = val in corpus

        results.append({**f, "verified": verified})

    return results


def _fuzzy_date_match(date_str: str, corpus: str) -> bool:
    """Allow minor variations like 9月15日 vs 9月15日 (full-width digits)."""
    # Normalize digits
    normalized = date_str.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return normalized in corpus


def apply_fact_check(
    answer: str,
    chunks: list[dict],
    current_tier: str,
) -> dict[str, Any]:
    """Full fact-check pipeline. Returns modified answer and tier.

    Returns dict with:
      - answer: modified answer text
      - tier: new confidence tier (may be downgraded)
      - debug: {extracted, verified, removed, hedged, tier_after_check}
    """
    facts = extract_facts(answer)
    if not facts:
        return {
            "answer": answer,
            "tier": current_tier,
            "debug": {
                "extracted_facts": 0,
                "verified": 0,
                "removed": 0,
                "hedged": 0,
                "tier_after_check": current_tier,
            },
        }

    verified_facts = verify_facts(facts, chunks)
    failed = [f for f in verified_facts if not f["verified"]]

    removed_count = 0
    hedged_count = 0
    new_tier = current_tier

    if not failed:
        return {
            "answer": answer,
            "tier": current_tier,
            "debug": {
                "extracted_facts": len(facts),
                "verified": len(facts),
                "removed": 0,
                "hedged": 0,
                "unverified": [],
                "tier_after_check": current_tier,
            },
        }

    # Classify failures
    hard_fails = [f for f in failed if _is_zero_tolerance(f["type"])]
    soft_fails = [f for f in failed if not _is_zero_tolerance(f["type"])]

    # Severe: >=3 total failures OR any zero-tolerance → Tier 3
    if len(failed) >= 3 or len(hard_fails) > 0:
        # Downgrade to Tier 3: replace answer with short referral
        new_tier = "3"
        short_answer = (
            "这个问题我手头的校规资料里没有足够的信息来准确回答，"
            "建议直接联系相关部门：\n"
            "  - 教务处 (025) 8968-1234\n"
            "  - 或通过教务系统 jw.nju.edu.cn 在线咨询"
        )
        removed_count = len(failed)
        debug_info = {
            "extracted_facts": len(facts),
            "verified": len(facts) - len(failed),
            "removed": removed_count,
            "hedged": 0,
            "hard_failures": len(hard_fails),
            "soft_failures": len(soft_fails),
            "unverified": [f"{f['type']}:{f['value']}" for f in failed[:5]],
            "tier_after_check": "3",
        }
        return {"answer": short_answer, "tier": "3", "debug": debug_info}

    # Soft failures only (1-2 items): hedge them
    modified = answer
    for f in soft_fails:
        # Replace the specific value with a hedge
        val = f["value"]
        ftype = f["type"]
        hedge_map = {
            "date": f"(具体时间我看到的资料里没写，建议问教务员)",
            "proper_noun": f"(具体的{val}资料里没明说，你确认下)",
        }
        replacement = hedge_map.get(ftype, f"(具体的{val}资料里没写)")
        # Replace the value in the answer
        modified = modified.replace(val, replacement, 1)
        hedged_count += 1

    # Downgrade to Tier 2 if hedging
    if hedged_count > 0 and current_tier == "1":
        new_tier = "2"

    debug_info = {
        "extracted_facts": len(facts),
        "verified": len(facts) - len(failed),
        "removed": 0,
        "hedged": hedged_count,
        "hard_failures": 0,
        "soft_failures": len(soft_fails),
        "unverified": [f"{f['type']}:{f['value']}" for f in soft_fails],
        "tier_after_check": new_tier,
    }

    return {"answer": modified, "tier": new_tier, "debug": debug_info}
