#!/usr/bin/env python3
"""Validate data/sources.csv for the NJU Rule RAG project.

Checks:
  1. Required fields exist
  2. source_id is unique
  3. priority is 1–5
  4. source_type is html / pdf / markdown / other
  5. url starts with http:// or https:// (if non-empty)

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
]

VALID_SOURCE_TYPES = {"html", "pdf", "markdown", "other"}
VALID_PRIORITIES = {1, 2, 3, 4, 5}
VALID_NEED_LOGIN = {"yes", "no"}


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
