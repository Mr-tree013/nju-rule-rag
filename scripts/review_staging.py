"""
Interactive staging review CLI (F.3).

Scans data/staging/ for new crawled documents, shows a summary,
and lets the reviewer accept, reject, edit, or skip each one.

Usage:
    python scripts/review_staging.py              # interactive mode
    python scripts/review_staging.py --list       # list staged files only
    python scripts/review_staging.py --auto       # auto-accept all
"""

import csv
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = ROOT / "data" / "staging"
SOURCES_CSV = ROOT / "data" / "sources.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARCHIVE_DIR = ROOT / "data" / "archive"


def _load_sources() -> tuple[list[dict], list[str]]:
    with open(SOURCES_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def _save_sources(rows: list[dict], fieldnames: list[str]) -> None:
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_staged() -> list[dict[str, Any]]:
    if not STAGING_DIR.exists():
        return []
    items = []
    for meta_file in sorted(STAGING_DIR.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            md_file = meta.get("file_path", "")
            if md_file and Path(md_file).exists():
                meta["_meta_file"] = str(meta_file)
                meta["_md_file"] = md_file
                items.append(meta)
        except (json.JSONDecodeError, KeyError):
            pass
    return items


def _show_item(item: dict, index: int, total: int) -> None:
    sid = item.get("source_id", "?")
    url = item.get("url", "")
    status = item.get("status", "?")
    note = item.get("note", "")
    md_file = item.get("_md_file", "")

    print(f"\n{'='*70}")
    print(f"[{index}/{total}] {sid}")
    print(f"  URL:     {url[:100]}")
    print(f"  Status:  {status}")
    print(f"  Note:    {note}")
    print(f"  File:    {md_file}")

    if md_file:
        try:
            content = Path(md_file).read_text(encoding="utf-8")
            lines = content.split("\n")
            print(f"\n  -- Preview (first 20 lines, {len(lines)} total, {len(content)} chars) --")
            for line in lines[:20]:
                print(f"  {line[:130]}")
        except Exception:
            pass


def _accept(item: dict) -> bool:
    sid = item.get("source_id", "")
    md_file = item.get("_md_file", "")
    new_hash = item.get("content_hash", "")

    if not sid or not md_file:
        print("  [ERROR] Missing source_id or file path")
        return False

    src_path = Path(md_file)
    if not src_path.exists():
        print(f"  [ERROR] Source file not found: {md_file}")
        return False

    dst_path = PROCESSED_DIR / src_path.name
    shutil.copy(src_path, dst_path)
    print(f"  [OK] Copied to {dst_path}")

    rows, fieldnames = _load_sources()
    updated = False
    today = date.today().isoformat()
    for row in rows:
        if row.get("source_id", "").strip() == sid:
            row["content_hash"] = new_hash
            row["last_crawled_at"] = today
            old_filename = row.get("filename", "")
            if old_filename and old_filename != dst_path.name:
                # Archive old file
                old_path = PROCESSED_DIR / old_filename
                if old_path.exists():
                    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_path), str(ARCHIVE_DIR / old_filename))
                    print(f"  [OK] Archived old: {old_filename}")
            row["filename"] = dst_path.name
            updated = True
            break

    if updated:
        _save_sources(rows, fieldnames)
        print(f"  [OK] Updated sources.csv (hash={new_hash[:12]}...)")
    else:
        print(f"  [WARN] source_id '{sid}' not found in sources.csv")

    # Clean up staging
    meta_path = Path(item.get("_meta_file", ""))
    if meta_path.exists():
        meta_path.unlink()
    if src_path.exists():
        src_path.unlink()

    return True


def _reject(item: dict) -> None:
    for p in [item.get("_md_file", ""), item.get("_meta_file", "")]:
        if p and Path(p).exists():
            Path(p).unlink()
    print(f"  [OK] Rejected")


def interactive_review() -> int:
    items = _find_staged()
    if not items:
        print("No staged documents to review.")
        return 0

    total = len(items)
    stats = {"accepted": 0, "rejected": 0, "skipped": 0}

    for i, item in enumerate(items, 1):
        _show_item(item, i, total)
        while True:
            choice = input("\n  [a]ccept  [r]eject  [s]kip  [q]uit ? ").strip().lower()
            if choice in ("a", "accept"):
                if _accept(item):
                    stats["accepted"] += 1
                break
            elif choice in ("r", "reject"):
                _reject(item)
                stats["rejected"] += 1
                break
            elif choice in ("s", "skip"):
                stats["skipped"] += 1
                break
            elif choice in ("q", "quit"):
                print(f"\nQuit. {stats['accepted']} accepted, {stats['rejected']} rejected, "
                      f"{stats['skipped']} skipped, {total - i} remaining.")
                return 0
            else:
                print("  Enter a/r/s/q")

    print(f"\nDone: {stats['accepted']} accepted, {stats['rejected']} rejected, "
          f"{stats['skipped']} skipped")
    return 0


def list_staged() -> int:
    items = _find_staged()
    if not items:
        print("No staged documents.")
        return 0
    print(f"{len(items)} staged document(s):\n")
    for item in items:
        print(f"  [{item.get('status', '?')}] {item.get('source_id', '?')}")
        print(f"    URL:  {item.get('url', '')[:100]}")
        print(f"    Note: {item.get('note', '')}")
        print()
    return 0


def auto_accept() -> int:
    items = _find_staged()
    if not items:
        print("No staged documents.")
        return 0
    accepted = 0
    for item in items:
        print(f"Auto-accepting: {item.get('source_id', '?')}...")
        if _accept(item):
            accepted += 1
    print(f"\nAccepted {accepted}/{len(items)}")
    return 0


def main() -> int:
    if "--list" in sys.argv:
        return list_staged()
    elif "--auto" in sys.argv:
        return auto_accept()
    else:
        return interactive_review()


if __name__ == "__main__":
    sys.exit(main())
