"""
Scheduled crawl runner (F.1).

Filters sources by crawl_frequency and runs change detection.
Designed to be called from cron. Skips sources marked 'manual'
or 'on_demand' — only processes sources with an active frequency.

Usage:
    python scripts/crawl_scheduled.py              # run all active sources
    python scripts/crawl_scheduled.py --frequency weekly   # only weekly
    python scripts/crawl_scheduled.py --dry-run    # list what would run
    python scripts/crawl_scheduled.py --limit 5    # test with first 5

Cron examples:
    0 9 * * 1  cd /path/to/project && PYTHONPATH=. python scripts/crawl_scheduled.py
    0 9 1 * *  cd /path/to/project && PYTHONPATH=. python scripts/crawl_scheduled.py --frequency monthly
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = ROOT / "data" / "sources.csv"

ACTIVE_FREQUENCIES = {"daily", "weekly", "monthly", "semester", "yearly"}


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    limit = 0
    freq_filter = None

    for arg in sys.argv[1:]:
        if arg.startswith("--frequency="):
            freq_filter = arg.split("=", 1)[1].strip().lower()
        elif arg.startswith("--limit="):
            try:
                limit = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    if not SOURCES_CSV.exists():
        print(f"Error: {SOURCES_CSV} not found")
        return 1

    with open(SOURCES_CSV, encoding="utf-8-sig") as f:
        sources = list(csv.DictReader(f))

    # Filter to active sources
    targets = []
    for row in sources:
        sid = row.get("source_id", "").strip()
        url = row.get("url", "").strip()
        need_login = row.get("need_login", "").strip().lower()
        crawl_freq = row.get("crawl_frequency", "").strip().lower()

        if not sid or not url or need_login == "yes":
            continue
        if crawl_freq not in ACTIVE_FREQUENCIES:
            continue
        if freq_filter and crawl_freq != freq_filter:
            continue

        targets.append((sid, url, crawl_freq))

    if dry_run:
        print(f"Would crawl {len(targets)} source(s):")
        for sid, url, freq in targets[:limit or len(targets)]:
            print(f"  [{freq}] {sid}: {url[:100]}")
        return 0

    if not targets:
        print("No active sources to crawl.")
        return 0

    if limit:
        targets = targets[:limit]

    print(f"Crawling {len(targets)} source(s)...")

    from app.crawl.fetcher import fetch_and_stage, _load_source_hashes
    known = _load_source_hashes()

    stats = {"total": 0, "unchanged": 0, "staged": 0, "new": 0, "error": 0}
    for sid, url, freq in targets:
        stats["total"] += 1
        print(f"[{stats['total']}/{len(targets)}] [{freq}] {sid}: {url[:80]}...")
        result = fetch_and_stage(
            url=url, source_id=sid,
            known_hash=known.get(sid, ""),
        )
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        print(f"  → {status}: {result.get('note', result.get('error', ''))}")

    print(f"\nDone: {stats['total']} crawled, {stats['unchanged']} unchanged, "
          f"{stats['staged']} changed, {stats['new']} new, {stats['error']} errors")
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
