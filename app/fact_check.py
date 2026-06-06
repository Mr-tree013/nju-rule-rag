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


def _build_normalized_corpus(chunks: list[dict]) -> str:
    """Build a normalized version of the corpus where amounts/phones/URLs
    are normalized in-place so that normalized fact values can be found
    via substring search."""
    corpus = "\n".join(c.get("content", "") for c in chunks)
    # Normalize all phone-like patterns (digit groups with any separators)
    corpus = re.sub(r'(?<!\d)(\d{3,4})[-\s]?(\d{3,4})[-\s]?(\d{0,4})(?!\d)', r'\1\2\3', corpus)
    corpus = re.sub(r'(?<!\d)(\d{2,4})[-\s](\d{6,8})(?!\d)', r'\1\2', corpus)
    # Normalize full-width digits in numbers
    corpus = corpus.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    # Normalize Chinese numerals in context
    corpus = _normalize_chinese_numerals(corpus)
    # Normalize amounts: "2.0 学分" → "2学分", "2.00 元" → "2元"
    corpus = re.sub(r'(?<=\d)\.0+\s*(?=[^0-9])', '', corpus)
    corpus = re.sub(r'(?<=\d)\.00\s*(?=[^0-9])', '', corpus)
    # Remove spaces between digits and amount units: "500 元" → "500元"
    corpus = re.sub(r'(?<=\d)\s+(?=(?:元|块|学分|次|%|万元|千元|百元))', '', corpus)
    # URL normalization: strip protocol, www, trailing slash for comparison
    corpus = re.sub(r'https?://(?:www\.)?', '', corpus)
    corpus = re.sub(r'/+$', '', corpus)
    return corpus


def _extract_corpus_numbers(corpus: str) -> list[str]:
    """Extract all number-like substrings from corpus for fuzzy matching."""
    patterns = [
        r'\d+(?:\.\d+)?\s*(?:元|块|学分|次|%|万元|千元|百元)',
        r'\d+(?:\.\d+)?',
    ]
    results = []
    for pat in patterns:
        results.extend(re.findall(pat, corpus))
    return results


def _similarity(a: str, b: str) -> float:
    """Compute similarity between two strings, number-aware for amounts."""
    if a == b:
        return 1.0
    # Normalize whitespace and fullwidth
    na = re.sub(r'[\s　]', '', a)
    nb = re.sub(r'[\s　]', '', b)
    if na == nb:
        return 1.0
    # Number-aware: if both contain numbers, compare numeric closeness
    nums_a = re.findall(r'\d+(?:\.\d+)?', na)
    nums_b = re.findall(r'\d+(?:\.\d+)?', nb)
    if nums_a and nums_b:
        try:
            n_a = float(nums_a[0])
            n_b = float(nums_b[0])
            # Ratio-based similarity: 500 vs 501 = 0.998, 500 vs 600 = 0.833
            if max(abs(n_a), abs(n_b)) > 0:
                num_sim = 1.0 - min(abs(n_a - n_b) / max(abs(n_a), abs(n_b)), 1.0)
                # Blend: 90% number similarity + 10% string similarity
                str_sim = _bigram_sim(na, nb)
                return 0.9 * num_sim + 0.1 * str_sim
        except ValueError:
            pass
    return _bigram_sim(na, nb)


def _bigram_sim(a: str, b: str) -> float:
    """Character bigram overlap similarity."""
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1)) if len(s) >= 2 else {s}
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / max(len(ba), len(bb))


def _best_corpus_match(val: str, norm_val: str, corpus: str, norm_corpus: str) -> tuple[float, str]:
    """Find the best matching substring in the corpus for a fact value.
    Returns (similarity, best_matched_substring)."""
    best_sim = 0.0
    best_match = ""
    # Search in both original and normalized corpus
    for c in [corpus, norm_corpus]:
        # Try to find the normalized value or nearby text
        idx = c.find(norm_val)
        if idx >= 0:
            return (1.0, norm_val)
        # Try partial: extract a window around where the value might appear
        # by searching for parts of the value
        search_val = norm_val
        # Try progressively shorter prefixes
        for end in range(len(search_val), max(len(search_val) // 2, 2), -1):
            sub = search_val[:end]
            idx = c.find(sub)
            if idx >= 0:
                # Extract the surrounding context
                start = max(0, idx - 2)
                end_pos = min(len(c), idx + len(search_val) + 5)
                match_text = c[start:end_pos]
                sim = _similarity(search_val, match_text)
                if sim > best_sim:
                    best_sim = sim
                    best_match = match_text.strip()[:80]
                break
    return (best_sim, best_match)


_SIMILARITY_THRESHOLD = 0.85


def verify_facts(
    facts: list[dict[str, str]],
    chunks: list[dict],
) -> list[dict[str, Any]]:
    """Verify each fact against the retrieved chunks.

    Returns facts with 'verified', 'match_quality' fields added.
    match_quality: 'exact' | 'partial' (>85% similarity) | 'none'
    """
    corpus = "\n".join(c.get("content", "") for c in chunks)
    norm_corpus = _build_normalized_corpus(chunks)
    corpus_numbers = _extract_corpus_numbers(corpus)

    results = []
    for f in facts:
        val = f["value"]
        ftype = f["type"]
        verified = False
        match_quality = "none"
        best_sim = 0.0
        best_match = ""
        # Apply Chinese numeral normalization to fact values too
        val_norm = _normalize_chinese_numerals(val)

        if ftype == "url":
            norm_val = _normalize_url(val_norm)
            verified = val_norm in corpus or norm_val in corpus or norm_val in norm_corpus
            if not verified:
                best_sim, best_match = _best_corpus_match(val_norm, norm_val, corpus, norm_corpus)
        elif ftype == "phone":
            norm_val = _normalize_phone(val_norm)
            verified = val_norm in corpus or norm_val in corpus or norm_val in norm_corpus
            if not verified:
                best_sim, best_match = _best_corpus_match(val_norm, norm_val, corpus, norm_corpus)
        elif ftype == "amount":
            norm_val = _normalize_amount(val_norm)
            verified = val_norm in corpus or norm_val in corpus or norm_val in norm_corpus
            if not verified:
                # Try fuzzy matching against corpus numbers
                for cnum in corpus_numbers:
                    sim = _similarity(norm_val, _normalize_amount(cnum))
                    if sim > best_sim:
                        best_sim = sim
                        best_match = cnum
                if best_sim >= _SIMILARITY_THRESHOLD:
                    match_quality = "partial"
                    # stays verified=False → treated as soft_fail (hedge) not hard_fail (delete)
        elif ftype == "date":
            verified = val_norm in corpus or _fuzzy_date_match(val_norm, corpus)
            if not verified:
                norm_date = val_norm.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
                verified = norm_date in norm_corpus
        elif ftype == "email":
            verified = val_norm in corpus
        elif ftype == "proper_noun":
            verified = val_norm in corpus
        else:
            verified = val_norm in corpus

        # For non-amount types, check partial match if not verified
        if not verified and ftype != "amount":
            if best_sim >= _SIMILARITY_THRESHOLD:
                match_quality = "partial"

        results.append({
            **f,
            "verified": verified,
            "match_quality": match_quality,
            "best_similarity": round(best_sim, 3),
            "best_match": best_match[:80],
        })

    return results


# Chinese numeral → Arabic mapping
_CN_NUM_MAP = str.maketrans({
    "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
})
# Compound Chinese numerals: "二十"→"20", "三十"→"30", "十二"→"12", etc.
_CN_NUM_PATTERNS = [
    (re.compile(r"(?:[零一二三四五六七八九]?[十百千万亿])+[零一二三四五六七八九]?"), _CN_NUM_MAP),
    (re.compile(r"第\s*([一二三四五六七八九十]+)\s*(?=[周学期年])"), None),  # will be handled separately
]


def _normalize_chinese_numerals(text: str) -> str:
    """Convert Chinese numerals in context patterns to Arabic.
    Handles: "第三学年"→"第3学年", "二十学分"→"20学分", "二万元"→"2万元".
    """
    s = text
    # Compound teens: 十一→11, 十二→12, ..., 十九→19
    s = re.sub(r'(?<![二三四五六七八九])十\s*([一二三四五六七八九])',
               lambda m: f'1{_cn_digit_to_arabic(m.group(1))}', s)
    # Bare 十 before time classifiers → 10 (but not preceded by Chinese digit like 五十)
    s = re.sub(r'(?<![一二三四五六七八九\d])十\s*(?=(?:周|年|届|学期|学年))', '10', s)
    # "第X" patterns: 第一→第1, 第十→第10
    s = re.sub(r'第\s*([一二三四五六七八九])\s*(?=(?:周|年|届|学期|学年|条|章|节))',
               lambda m: f"第{_cn_digit_to_arabic(m.group(1))}", s)
    s = re.sub(r'第\s*十\s*(?=(?:周|年|届|学期|学年|条|章|节))', "第10", s)
    # "二十"→"20", "三十"→"30" etc when followed by common units
    _units = r'(?:元|块|学分|次|%|万元|千元|百元|万|年|天|周|月|人|间|个)'
    for tens_char, tens_num in [("二", "2"), ("三", "3"), ("四", "4"),
                                  ("五", "5"), ("六", "6"), ("七", "7"),
                                  ("八", "8"), ("九", "9")]:
        s = re.sub(rf'{tens_char}十\s*(?={_units})', f'{tens_num}0', s)
    # Single digit before 万/千/百: 二万→2万, 三千→3千 (not preceded by 十)
    for digit_char, digit_num in [("一", "1"), ("二", "2"), ("三", "3"),
                                    ("四", "4"), ("五", "5"), ("六", "6"),
                                    ("七", "7"), ("八", "8"), ("九", "9")]:
        s = re.sub(rf'(?<![十]){digit_char}\s*(?=[万千百])', digit_num, s)
    # Single digit before time classifiers: 一届→1届, 三周→3周 (not preceded by 十)
    s = re.sub(r'(?<![十])([一二三四五六七八九])\s*(?=(?:周|年|届|学期|学年))',
               lambda m: _cn_digit_to_arabic(m.group(1)), s)
    return s


def _cn_digit_to_arabic(cn: str) -> str:
    """Convert single Chinese digit to Arabic. "三"→"3", "十"→"10"."""
    mapping = {"零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
               "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
    return mapping.get(cn, cn)


def _normalize_amount(val: str) -> str:
    """Normalize amounts: strip spaces, convert full-width/Chinese digits, handle decimal zero."""
    s = val.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    s = s.replace(" ", "").replace("　", "")
    s = _normalize_chinese_numerals(s)
    # "2.0学分" and "2学分" should match
    s = re.sub(r'(?<=\d)\.0+(?=[^0-9])', '', s)
    # "2.00元" and "2元"
    s = re.sub(r'(?<=\d)\.00(?=[^0-9])', '', s)
    return s


def _normalize_phone(val: str) -> str:
    """Strip all non-digits, handle common separators."""
    return re.sub(r'[^\d]', '', val)


def _normalize_url(val: str) -> str:
    """Strip trailing slash, www prefix, normalize protocol."""
    s = val.lower()
    s = re.sub(r'^https?://(?:www\.)?', '', s)
    s = s.rstrip('/')
    return s


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
    # Partial matches: close enough to hedge rather than delete
    partial_matches = [f for f in verified_facts if f.get("match_quality") == "partial"]

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

    # Classify failures: truly unverified (match_quality='none') vs partial matches
    hard_fails = [f for f in failed
                  if _is_zero_tolerance(f["type"]) and f.get("match_quality") != "partial"]
    soft_fails = [f for f in failed
                  if not _is_zero_tolerance(f["type"]) or f.get("match_quality") == "partial"]
    # Truly unverified: neither exact match nor partial match
    truly_unverified = [f for f in failed if f.get("match_quality") != "partial"]

    # Severe: >=3 truly-unverified failures OR >=2 hard failures → Tier 3
    if len(truly_unverified) >= 3 or len(hard_fails) >= 2:
        new_tier = "3"
        short_answer = (
            "这个问题我手头的校规资料里没有足够的信息来准确回答，"
            "建议直接联系相关部门：\n"
            "  - 教务处 (025) 8968-1234\n"
            "  - 或通过教务系统 jw.nju.edu.cn 在线咨询"
        )
        return {
            "answer": short_answer, "tier": "3",
            "debug": {
                "extracted_facts": len(facts),
                "verified": len(facts) - len(failed),
                "removed": len(failed),
                "hedged": 0,
                "partial_matches": len(partial_matches),
                "hard_failures": len(hard_fails),
                "soft_failures": len(soft_fails),
                "unverified": [f"{f['type']}:{f['value']}" for f in failed[:5]],
                "tier_after_check": "3",
            },
        }

    # 1 hard_fail + optional soft_fails: hedge everything, keep answer
    modified = answer
    hedged_count = 0
    all_to_hedge = hard_fails + soft_fails  # at most 1 hard + 1-2 soft

    for f in all_to_hedge:
        val = f["value"]
        ftype = f["type"]
        is_partial = f.get("match_quality") == "partial"
        # Partial match: softer hedge (info might exist, just doesn't match exactly)
        # Truly unverified: stronger hedge (info absent from sources)
        if is_partial:
            hedge_map = {
                "amount": f"(具体数字可能有出入，建议跟教务员确认)",
                "date": f"(具体时间可能有出入，建议跟教务员确认)",
                "url": f"(具体网址可能有变动)",
                "email": f"(具体联系方式可能有变动)",
                "phone": f"(具体电话可能有变动，建议确认)",
                "proper_noun": f"(具体的{val}资料里说法可能不同，你确认下)",
            }
        else:
            hedge_map = {
                "amount": f"(具体费用我看到的资料里没写，建议问教务员)",
                "date": f"(具体时间我看到的资料里没写，建议问教务员)",
                "url": f"(具体网址我看到的资料里没写)",
                "email": f"(具体联系方式我看到的资料里没写)",
                "phone": f"(具体电话我看到的资料里没写)",
                "proper_noun": f"(具体的{val}资料里没明说，你确认下)",
            }
        replacement = hedge_map.get(ftype, f"(具体的{val}资料里没写)")
        modified = modified.replace(val, replacement, 1)
        hedged_count += 1

    # Downgrade to Tier 2 if hedging
    if hedged_count > 0 and current_tier == "1":
        new_tier = "2"

    return {
        "answer": modified, "tier": new_tier,
        "debug": {
            "extracted_facts": len(facts),
            "verified": len(facts) - len(failed),
            "removed": 0,
            "hedged": hedged_count,
            "partial_matches": len(partial_matches),
            "hard_failures": len(hard_fails),
            "soft_failures": len(soft_fails),
            "unverified": [f"{f['type']}:{f['value']}" for f in all_to_hedge],
            "tier_after_check": new_tier,
        },
    }
