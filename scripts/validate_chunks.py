#!/usr/bin/env python3
"""Validate data/chunks/chunks.jsonl for the NJU Rule RAG project.

Checks:
  1. Every chunk has all required fields
  2. chunk_id is unique
  3. priority is 1–5 as integer
  4. content is non-empty and has enough Chinese characters

Prints statistics and exits with code 1 on severe errors.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks" / "chunks.jsonl"

# Fields the app side expects (from dev contract, see docs/dev_contract.md).
CONTRACT_FIELDS = [
    "chunk_id",
    "source_id",
    "title",
    "url",
    "department",
    "scope",
    "priority",
    "article",
    "content",
    "fetched_at",
]

# Fields whose value must be non-empty (key must exist AND value truthy).
MUST_HAVE_VALUE = {
    "chunk_id",
    "source_id",
    "title",
    "department",
    "scope",
    "priority",
    "content",
    "article",
}

# Fields that may be empty strings.
CAN_BE_EMPTY = {"url", "fetched_at"}

# Deprecated / legacy field — kept for backward compat.
RECOMMENDED_FIELDS = ["section"]

VALID_PRIORITIES = {1, 2, 3, 4, 5}

# Chunks shorter than this (Chinese characters) get a warning.
MIN_CHINESE_CHARS = 20


# ── helpers ─────────────────────────────────────────────────────────

def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def _count_chinese(text: str) -> int:
    """Count CJK characters in *text*."""
    return sum(1 for ch in text if "一" <= ch <= "鿿")


# ── main ────────────────────────────────────────────────────────────

def validate_chunks(path: Path) -> int:
    """Run all validations.  Returns 1 if severe errors found, 0 otherwise."""

    if not path.exists():
        print(_red(f"[FATAL] File not found: {path}"))
        return 1

    chunks: list[dict] = []
    read_errors: list[tuple[int, str]] = []

    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunks.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                read_errors.append((i, str(exc)))

    if read_errors:
        for row_num, msg in read_errors:
            print(_red(f"[ERROR] Line {row_num}: invalid JSON — {msg}"))

    if not chunks:
        print(_red("[FATAL] chunks.jsonl is empty or has no valid JSON lines."))
        return 1

    print(f"Loaded {len(chunks)} chunks from {path}")
    print()

    warnings = 0

    # ── collect per-chunk issues ─────────────────────────────────────

    seen_ids: set[str] = set()
    dup_ids: set[str] = set()

    missing_field_keys: Counter[str] = Counter()      # key absent from dict
    empty_value_fields: Counter[str] = Counter()       # key present but value empty
    missing_recommended: Counter[str] = Counter()
    empty_content_ids: list[str] = []
    short_content: list[tuple[str, int]] = []
    bad_priority: list[tuple[str, object]] = []

    stats_per_source: Counter[str] = Counter()
    stats_priority: Counter[str] = Counter()

    for chunk in chunks:
        cid = chunk.get("chunk_id", "") or "(missing)"

        # 1a.  contract fields — key existence
        for f in CONTRACT_FIELDS:
            if f not in chunk:
                missing_field_keys[f] += 1

        # 1b.  contract fields — value emptiness (only for MUST_HAVE_VALUE)
        for f in MUST_HAVE_VALUE:
            if f in chunk and not chunk.get(f):
                empty_value_fields[f] += 1

        # 1c.  recommended fields
        for f in RECOMMENDED_FIELDS:
            if f not in chunk or not chunk.get(f):
                missing_recommended[f] += 1

        # Skip further per-chunk checks only if core fields are broken.
        core_broken = (
            "chunk_id" not in chunk or not chunk.get("chunk_id")
            or "content" not in chunk
        )
        if core_broken:
            continue

        # 2.  duplicate chunk_id
        if cid in seen_ids:
            dup_ids.add(cid)
        else:
            seen_ids.add(cid)

        # 3.  priority
        prio = chunk.get("priority")
        if not isinstance(prio, int) or prio not in VALID_PRIORITIES:
            bad_priority.append((cid, prio))
        else:
            stats_priority[str(prio)] += 1

        # 4.  content
        content = chunk.get("content", "")
        if not content.strip():
            empty_content_ids.append(cid)
        else:
            cn_chars = _count_chinese(content)
            if cn_chars < MIN_CHINESE_CHARS:
                short_content.append((cid, cn_chars))

        # stats
        sid = chunk.get("source_id", "")
        if sid:
            stats_per_source[sid] += 1

    # ── print errors ─────────────────────────────────────────────────

    # missing field keys (key not in dict at all) — severe
    if missing_field_keys:
        for field, count in missing_field_keys.most_common():
            print(_red(f"[ERROR] {count} chunk(s) missing field key '{field}'."))
        print()

    # empty value fields (key exists but value is empty) — severe
    if empty_value_fields:
        for field, count in empty_value_fields.most_common():
            print(_red(f"[ERROR] {count} chunk(s) have empty value for '{field}'."))
        print()

    # empty content — severe
    if empty_content_ids:
        for cid in empty_content_ids[:5]:
            print(_red(f"[ERROR] Chunk '{cid}': content is empty."))
        if len(empty_content_ids) > 5:
            print(_red(f"      ... and {len(empty_content_ids) - 5} more."))
        print()

    # duplicate ids — severe
    if dup_ids:
        for cid in sorted(dup_ids):
            print(_red(f"[ERROR] Duplicate chunk_id: {cid}"))
        print()
    else:
        print(_green("[OK] All chunk_id values are unique."))

    # bad priority — severe
    if bad_priority:
        for cid, val in bad_priority[:5]:
            print(_red(f"[ERROR] Chunk '{cid}': invalid priority {val!r} (must be 1–5)."))
        if len(bad_priority) > 5:
            print(_red(f"      ... and {len(bad_priority) - 5} more."))
        print()
    else:
        print(_green("[OK] All priority values are in 1–5."))

    # missing recommended — warning
    if missing_recommended:
        for field, count in missing_recommended.most_common():
            print(_yellow(f"[WARN] {count} chunk(s) missing recommended field '{field}'."))
        print()
        warnings += 1

    # short content — warning
    if short_content:
        for cid, cn in short_content[:5]:
            print(_yellow(f"[WARN] Chunk '{cid}': only {cn} Chinese chars (min {MIN_CHINESE_CHARS})."))
        if len(short_content) > 5:
            print(_yellow(f"      ... and {len(short_content) - 5} more."))
        print()
        warnings += 1

    # ── determine result ────────────────────────────────────────────

    has_severe = bool(
        missing_field_keys
        or empty_value_fields
        or dup_ids
        or bad_priority
        or empty_content_ids
        or read_errors
    )

    # ── statistics ──────────────────────────────────────────────────

    print("─" * 50)
    print("Statistics")
    print("─" * 50)
    print(f"  Total chunks:          {len(chunks)}")
    print(f"  Unique chunk_ids:      {len(seen_ids)}")
    print(f"  Missing field keys:    {sum(missing_field_keys.values())}")
    print(f"  Empty value fields:    {sum(empty_value_fields.values())}")
    print(f"  Empty content:         {len(empty_content_ids)}")
    print(f"  Short content (<{MIN_CHINESE_CHARS} cn): {len(short_content)}")

    print(f"  By priority:")
    for p in sorted(stats_priority, key=int):
        bar = "█" * max(1, stats_priority[p] // 4)
        print(f"    priority {p}: {stats_priority[p]:>4}  {bar}")

    print(f"  By source_id (top {min(10, len(stats_per_source))}):")
    for sid, count in stats_per_source.most_common(10):
        print(f"    {sid}: {count}")
    if len(stats_per_source) > 10:
        print(f"    ... and {len(stats_per_source) - 10} more sources.")

    print()
    if has_severe:
        print(_red("✗ Validation FAILED"))
    elif warnings:
        print(_yellow(f"✓ Validation PASSED with {warnings} warning(s)."))
    else:
        print(_green("✓ All checks passed."))

    return 1 if has_severe else 0


if __name__ == "__main__":
    sys.exit(validate_chunks(CHUNKS_PATH))
