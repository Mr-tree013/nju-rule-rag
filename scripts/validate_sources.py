#!/usr/bin/env python3
"""Validate data/sources.csv for the NJU Rule RAG project.

Checks:
  1. Required fields exist (including E.2 crawl metadata columns)
  2. source_id is unique
  3. priority is 1–5
  4. source_type is html / pdf / markdown / other
  5. url starts with http:// or https:// (if non-empty)
  6. crawl_method is valid (static/render/pdf/manual/wechat)
  7. crawl_frequency is valid (weekly/monthly/yearly/on_demand)
  8. content_hash is present (16-char hex)
  9. chunk_strategy is valid (article/heading/qa/table_row/fixed)
 10. stale_after_days is positive integer

Prints statistics and exits with code 1 on severe errors.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCES_PATH = DATA_DIR / "sources.csv"

REQUIRED_FIELDS = [
    "source_id",
    "title",
    "url",
    "source_type",
    "department",
    "scope",
    "priority",
    "need_login",
    "update_frequency",
    # E.2 crawl metadata
    "topics",
    "crawl_method",
    "crawl_frequency",
    "last_crawled_at",
    "content_hash",
    "stale_after_days",
    "auth_required",
    "chunk_strategy",
]

VALID_SOURCE_TYPES = {"html", "pdf", "markdown", "other"}
VALID_PRIORITIES = {1, 2, 3, 4, 5}
VALID_NEED_LOGIN = {"yes", "no"}
VALID_CRAWL_METHODS = {"static", "render", "pdf", "manual", "wechat"}
VALID_CRAWL_FREQUENCIES = {"weekly", "monthly", "yearly", "semester", "on_demand"}
VALID_CHUNK_STRATEGIES = {"article", "heading", "qa", "table_row", "fixed"}


# ── helpers ─────────────────────────────────────────────────────────

def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


# ── main ────────────────────────────────────────────────────────────

def validate_sources(path: Path) -> int:
    """Run all validations.  Returns 1 if severe errors found, 0 otherwise."""

    # ── 0.  read CSV ────────────────────────────────────────────────

    if not path.exists():
        print(_red(f"[FATAL] File not found: {path}"))
        return 1

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(_red("[FATAL] CSV file is empty."))
        return 1

    fields = reader.fieldnames or []

    print(f"Loaded {len(rows)} rows from {path}")
    print(f"Columns found ({len(fields)}): {', '.join(fields)}")
    print()

    severe = 0
    warnings = 0

    # ── 1.  required fields ─────────────────────────────────────────

    missing_fields = [f for f in REQUIRED_FIELDS if f not in fields]
    if missing_fields:
        print(_red(f"[ERROR] Missing required fields: {missing_fields}"))
        print()
        severe += 1
    else:
        print(_green("[OK] All required fields present."))

    # ── collect per-row issues ──────────────────────────────────────

    seen_ids: set[str] = set()
    dup_ids: list[str] = []

    type_issues: list[tuple[int, str]] = []       # (row#, value)
    priority_issues: list[tuple[int, str]] = []
    url_issues: list[tuple[int, str]] = []
    login_issues: list[tuple[int, str]] = []
    empty_urls: list[int] = []     # row# — warning, local files are ok
    missing_field_cells: list[tuple[int, str]] = []  # (row#, field_name)

    stats_priority: Counter[str] = Counter()
    stats_department: Counter[str] = Counter()
    stats_type: Counter[str] = Counter()
    stats_need_login: Counter[str] = Counter()
    stats_crawl_method: Counter[str] = Counter()
    stats_chunk_strategy: Counter[str] = Counter()
    stats_auth_required: Counter[str] = Counter()

    # New-column issues
    crawl_method_issues: list[tuple[int, str]] = []
    crawl_freq_issues: list[tuple[int, str]] = []
    content_hash_missing: list[int] = []
    content_hash_bad: list[tuple[int, str]] = []
    stale_after_issues: list[tuple[int, str]] = []
    auth_required_issues: list[tuple[int, str]] = []
    chunk_strategy_issues: list[tuple[int, str]] = []
    topics_empty: list[int] = []

    for i, row in enumerate(rows, start=2):  # line 1 is header
        # check for empty required cells (url is exempt — local files are ok)
        for field in REQUIRED_FIELDS:
            val = row.get(field, "").strip()
            if not val and field != "url":
                missing_field_cells.append((i, field))

        sid = row.get("source_id", "").strip()
        stype = row.get("source_type", "").strip().lower()
        prio_str = row.get("priority", "").strip()
        url = row.get("url", "").strip()
        need_login = row.get("need_login", "").strip().lower()
        dept = row.get("department", "").strip()

        # 2.  duplicate source_id
        if sid:
            if sid in seen_ids:
                dup_ids.append(sid)
            else:
                seen_ids.add(sid)

        # 3.  priority
        if prio_str:
            try:
                prio = int(prio_str)
                if prio not in VALID_PRIORITIES:
                    priority_issues.append((i, prio_str))
                else:
                    stats_priority[str(prio)] += 1
            except ValueError:
                priority_issues.append((i, prio_str))

        # 4.  source_type
        if stype and stype not in VALID_SOURCE_TYPES:
            type_issues.append((i, stype))
        if stype:
            stats_type[stype] += 1

        # 5.  url — track empty (warning) vs malformed (warning)
        if not url:
            empty_urls.append(i)
        elif not (url.startswith("http://") or url.startswith("https://")):
            url_issues.append((i, url))

        # 6.  need_login
        if need_login and need_login not in VALID_NEED_LOGIN:
            login_issues.append((i, need_login))
        if need_login:
            stats_need_login[need_login] += 1

        # E.2 — crawl_method
        cm = row.get("crawl_method", "").strip().lower()
        if cm and cm not in VALID_CRAWL_METHODS:
            crawl_method_issues.append((i, cm))
        if cm:
            stats_crawl_method[cm] += 1

        # E.2 — crawl_frequency
        cf = row.get("crawl_frequency", "").strip().lower()
        if cf and cf not in VALID_CRAWL_FREQUENCIES:
            crawl_freq_issues.append((i, cf))

        # E.2 — content_hash
        ch = row.get("content_hash", "").strip()
        if not ch:
            content_hash_missing.append(i)
        elif not (len(ch) == 16 and all(c in "0123456789abcdef" for c in ch)):
            content_hash_bad.append((i, ch))

        # E.2 — stale_after_days
        sad = row.get("stale_after_days", "").strip()
        if sad:
            try:
                if int(sad) <= 0:
                    stale_after_issues.append((i, sad))
            except ValueError:
                stale_after_issues.append((i, sad))

        # E.2 — auth_required
        ar = row.get("auth_required", "").strip().lower()
        if ar and ar not in ("true", "false"):
            auth_required_issues.append((i, ar))
        if ar:
            stats_auth_required[ar] += 1

        # E.2 — chunk_strategy
        cs = row.get("chunk_strategy", "").strip().lower()
        if cs and cs not in VALID_CHUNK_STRATEGIES:
            chunk_strategy_issues.append((i, cs))
        if cs:
            stats_chunk_strategy[cs] += 1

        # E.2 — topics
        tp = row.get("topics", "").strip()
        if not tp:
            topics_empty.append(i)

        # stats
        if dept:
            stats_department[dept] += 1

    # ── print errors ────────────────────────────────────────────────

    # missing field cells
    if missing_field_cells:
        for row_num, field in missing_field_cells:
            print(_red(f"[ERROR] Row {row_num}: '{field}' is empty."))
        print()
        severe += 1

    # duplicate ids
    if dup_ids:
        for sid in dup_ids:
            print(_red(f"[ERROR] Duplicate source_id: {sid}"))
        print()
        severe += 1
    else:
        print(_green("[OK] All source_id values are unique."))

    # priority
    if priority_issues:
        for row_num, val in priority_issues:
            print(_red(f"[ERROR] Row {row_num}: invalid priority '{val}' (must be 1–5)."))
        print()
        severe += 1
    else:
        print(_green("[OK] All priority values are in 1–5."))

    # source_type
    if type_issues:
        for row_num, val in type_issues:
            print(_yellow(f"[WARN] Row {row_num}: unknown source_type '{val}' (expected html/pdf/markdown/other)."))
        print()
        warnings += len(type_issues)

    # url — empty (warning)
    if empty_urls:
        print(_yellow(f"[WARN] {len(empty_urls)} row(s) have empty url (local file, this is normal)."))
        print()
        warnings += 1

    # url — malformed (warning)
    if url_issues:
        for row_num, val in url_issues:
            print(_yellow(f"[WARN] Row {row_num}: url '{val}' does not start with http:// or https://."))
        print()
        warnings += len(url_issues)

    # need_login
    if login_issues:
        for row_num, val in login_issues:
            print(_yellow(f"[WARN] Row {row_num}: unexpected need_login '{val}' (expected yes/no)."))
        print()
        warnings += len(login_issues)

    # E.2 — crawl_method
    if crawl_method_issues:
        for row_num, val in crawl_method_issues:
            print(_red(f"[ERROR] Row {row_num}: invalid crawl_method '{val}' (expected {VALID_CRAWL_METHODS})."))
        print()
        severe += 1

    # E.2 — crawl_frequency
    if crawl_freq_issues:
        for row_num, val in crawl_freq_issues:
            print(_yellow(f"[WARN] Row {row_num}: unexpected crawl_frequency '{val}' (expected {VALID_CRAWL_FREQUENCIES})."))
        print()
        warnings += len(crawl_freq_issues)

    # E.2 — content_hash
    if content_hash_missing:
        print(_red(f"[ERROR] {len(content_hash_missing)} row(s) have empty content_hash. Run: python scripts/validate_sources.py --hash"))
        print()
        severe += 1
    if content_hash_bad:
        for row_num, val in content_hash_bad:
            print(_red(f"[ERROR] Row {row_num}: invalid content_hash '{val}' (expected 16 hex chars)."))
        print()
        severe += 1

    # E.2 — stale_after_days
    if stale_after_issues:
        for row_num, val in stale_after_issues:
            print(_yellow(f"[WARN] Row {row_num}: invalid stale_after_days '{val}' (expected positive integer)."))
        print()
        warnings += len(stale_after_issues)

    # E.2 — auth_required
    if auth_required_issues:
        for row_num, val in auth_required_issues:
            print(_yellow(f"[WARN] Row {row_num}: invalid auth_required '{val}' (expected true/false)."))
        print()
        warnings += len(auth_required_issues)

    # E.2 — chunk_strategy
    if chunk_strategy_issues:
        for row_num, val in chunk_strategy_issues:
            print(_red(f"[ERROR] Row {row_num}: invalid chunk_strategy '{val}' (expected {VALID_CHUNK_STRATEGIES})."))
        print()
        severe += 1

    # E.2 — topics empty
    if topics_empty:
        print(_yellow(f"[WARN] {len(topics_empty)} row(s) have empty topics field."))
        print()
        warnings += 1

    # ── statistics ──────────────────────────────────────────────────

    print("─" * 50)
    print("Statistics")
    print("─" * 50)
    print(f"  Total sources:   {len(rows)}")

    print(f"  By priority:")
    for p in sorted(stats_priority, key=int):
        bar = "█" * stats_priority[p]
        print(f"    priority {p}: {stats_priority[p]:>3}  {bar}")

    print(f"  By department:")
    for dept, count in stats_department.most_common():
        print(f"    {dept}: {count}")

    print(f"  By source_type:")
    for t, count in stats_type.most_common():
        print(f"    {t}: {count}")

    print(f"  need_login = yes:  {stats_need_login.get('yes', 0)}")
    print(f"  need_login = no:   {stats_need_login.get('no', 0)}")

    print(f"  By crawl_method:")
    for m, count in stats_crawl_method.most_common():
        print(f"    {m}: {count}")

    print(f"  By chunk_strategy:")
    for s, count in stats_chunk_strategy.most_common():
        print(f"    {s}: {count}")

    print(f"  auth_required = true:  {stats_auth_required.get('true', 0)}")
    print(f"  auth_required = false: {stats_auth_required.get('false', 0)}")

    print()
    if severe:
        print(_red(f"✗ {severe} severe error(s) found."))
    if warnings:
        print(_yellow(f"! {warnings} warning(s) found."))
    if not severe and not warnings:
        print(_green("✓ All checks passed."))

    return 1 if severe > 0 else 0


if __name__ == "__main__":
    sys.exit(validate_sources(SOURCES_PATH))
